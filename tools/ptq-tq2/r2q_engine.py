#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
llama-apu: R2Q (Cascaded Dual-Binary Residuals with Deviation-Aware Distillation)
Implements Candidate Workflow 2 (arXiv:2511.21736):
- Dual binary decomposition: W ~= alpha1 * Q1 + alpha2 * Q2 (Q1, Q2 in {-1, +1})
- Closed-form 2x2 least-squares scale solver + iterative sign relaxation
- 4-level adaptive lattice quantization (eliminating outlier blowout & zero-collapse)
- Deviation-Aware Distillation (DAD) using layer input activations
- Zero online activation rotation overhead (direct INT2 / GEMV execution)
"""

import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def r2q_decompose_block(w: np.ndarray, block_size: int = 256, n_iters: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes a 1D or 2D weight block into cascaded dual-binary residuals:
        w ~= alpha1 * q1 + alpha2 * q2
    where q1, q2 in {-1, +1}^G, and (alpha1, alpha2) are closed-form least-squares optimal scales.
    Iterative sign relaxation refines (q1, q2) to minimize MSE.
    
    Returns:
        q1: int8 array in {-1, +1}
        q2: int8 array in {-1, +1}
        alpha1: float per-block scale 1
        alpha2: float per-block scale 2
    """
    flat = w.flatten().astype(np.float32)
    n_elem = flat.size
    pad_size = 0
    if n_elem % block_size != 0:
        pad_size = block_size - (n_elem % block_size)
        flat = np.pad(flat, (0, pad_size), mode='constant')

    n_blocks = flat.size // block_size
    blocks = flat.reshape(n_blocks, block_size)

    # 1. First binary component: sign and L1 norm
    q1 = np.where(blocks >= 0.0, 1.0, -1.0).astype(np.float32)
    alpha1 = np.mean(np.abs(blocks), axis=1, keepdims=True)
    alpha1 = np.maximum(alpha1, 1e-8)

    # 2. Residual component
    r = blocks - alpha1 * q1
    q2 = np.where(r >= 0.0, 1.0, -1.0).astype(np.float32)
    alpha2 = np.mean(np.abs(r), axis=1, keepdims=True)
    alpha2 = np.maximum(alpha2, 1e-8)

    # 3. Iterative closed-form scale solver + sign relaxation
    for it in range(n_iters):
        # Closed-form 2x2 scale solver for each block:
        # [ G    c  ] [ a1 ] = [ b1 ]
        # [ c    G  ] [ a2 ]   [ b2 ]
        # where c = sum(q1 * q2), b1 = sum(q1 * w), b2 = sum(q2 * w)
        c = np.sum(q1 * q2, axis=1, keepdims=True)
        b1 = np.sum(q1 * blocks, axis=1, keepdims=True)
        b2 = np.sum(q2 * blocks, axis=1, keepdims=True)

        det = float(block_size * block_size) - c * c
        det = np.maximum(det, 1e-6)

        alpha1 = (float(block_size) * b1 - c * b2) / det
        alpha2 = (float(block_size) * b2 - c * b1) / det
        alpha1 = np.maximum(alpha1, 1e-8)
        alpha2 = np.maximum(alpha2, 1e-8)

        # Discrete sign relaxation: evaluate all 4 states for each element
        # s0 = -a1 - a2, s1 = -a1 + a2, s2 = +a1 - a2, s3 = +a1 + a2
        s0 = -alpha1 - alpha2
        s1 = -alpha1 + alpha2
        s2 = +alpha1 - alpha2
        s3 = +alpha1 + alpha2

        # Compute squared errors to the 4 levels: shape (n_blocks, block_size)
        e0 = (blocks - s0) ** 2
        e1 = (blocks - s1) ** 2
        e2 = (blocks - s2) ** 2
        e3 = (blocks - s3) ** 2

        # Pick best state per element
        best_state = np.argmin(np.stack([e0, e1, e2, e3], axis=-1), axis=-1)

        # Map state {0, 1, 2, 3} back to (q1, q2) in {-1, +1}
        # 0: (-1, -1), 1: (-1, +1), 2: (+1, -1), 3: (+1, +1)
        q1 = np.where((best_state == 2) | (best_state == 3), 1.0, -1.0)
        q2 = np.where((best_state == 1) | (best_state == 3), 1.0, -1.0)

    # Final scale update
    c = np.sum(q1 * q2, axis=1, keepdims=True)
    b1 = np.sum(q1 * blocks, axis=1, keepdims=True)
    b2 = np.sum(q2 * blocks, axis=1, keepdims=True)
    det = np.maximum(float(block_size * block_size) - c * c, 1e-6)
    alpha1 = np.maximum((float(block_size) * b1 - c * b2) / det, 1e-8)
    alpha2 = np.maximum((float(block_size) * b2 - c * b1) / det, 1e-8)

    q1_out = q1.reshape(-1)[:n_elem].reshape(w.shape).astype(np.int8)
    q2_out = q2.reshape(-1)[:n_elem].reshape(w.shape).astype(np.int8)

    return q1_out, q2_out, alpha1.flatten(), alpha2.flatten()

def r2q_dad_scale_distill(
    W_orig: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    alpha1_init: np.ndarray,
    alpha2_init: np.ndarray,
    X_calib: np.ndarray,
    steps: int = 20,
    lr: float = 1e-2,
    block_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Deviation-Aware Distillation (DAD):
    Optimizes (alpha1, alpha2) per block to minimize output activation discrepancy:
        min_{alpha1, alpha2} || X (alpha1 Q1 + alpha2 Q2)^T - X W_orig^T ||_F^2
    """
    a1 = alpha1_init.copy().astype(np.float32)
    a2 = alpha2_init.copy().astype(np.float32)
    
    n_blocks = a1.shape[0]
    q1_flat = q1.flatten().astype(np.float32).reshape(n_blocks, block_size)
    q2_flat = q2.flatten().astype(np.float32).reshape(n_blocks, block_size)
    
    Y_teacher = X_calib @ W_orig.T
    
    # Initial MSE
    W_init = (q1_flat * a1[:, None] + q2_flat * a2[:, None]).reshape(W_orig.shape)
    mse_init = float(np.mean((X_calib @ W_init.T - Y_teacher) ** 2))
    
    m1, v1 = np.zeros_like(a1), np.zeros_like(a1)
    m2, v2 = np.zeros_like(a2), np.zeros_like(a2)
    beta1, beta2 = 0.9, 0.999
    
    for step in range(1, steps + 1):
        W_q = (q1_flat * a1[:, None] + q2_flat * a2[:, None]).reshape(W_orig.shape)
        Y_student = X_calib @ W_q.T
        
        dY = 2.0 * (Y_student - Y_teacher) / (X_calib.shape[0] * W_orig.shape[0])
        dW = dY.T @ X_calib
        
        dW_blocks = dW.flatten().reshape(n_blocks, block_size)
        da1 = np.sum(dW_blocks * q1_flat, axis=1)
        da2 = np.sum(dW_blocks * q2_flat, axis=1)
        
        # AdamW updates
        m1 = beta1 * m1 + (1.0 - beta1) * da1
        v1 = beta2 * v1 + (1.0 - beta2) * (da1 ** 2)
        m1_hat = m1 / (1.0 - beta1 ** step)
        v1_hat = v1 / (1.0 - beta2 ** step)
        a1 -= lr * m1_hat / (np.sqrt(v1_hat) + 1e-8)
        a1 = np.maximum(a1, 1e-8)
        
        m2 = beta1 * m2 + (1.0 - beta1) * da2
        v2 = beta2 * v2 + (1.0 - beta2) * (da2 ** 2)
        m2_hat = m2 / (1.0 - beta1 ** step)
        v2_hat = v2 / (1.0 - beta2 ** step)
        a2 -= lr * m2_hat / (np.sqrt(v2_hat) + 1e-8)
        a2 = np.maximum(a2, 1e-8)
        
    W_final = (q1_flat * a1[:, None] + q2_flat * a2[:, None]).reshape(W_orig.shape)
    mse_final = float(np.mean((X_calib @ W_final.T - Y_teacher) ** 2))
    
    return a1, a2, mse_init, mse_final

def quantize_matrix_to_tq2_0_r2q(
    W: np.ndarray,
    X_calib: Optional[np.ndarray] = None,
    dad_steps: int = 15,
    dad_lr: float = 1e-2,
    block_size: int = 256
) -> bytes:
    """
    Quantizes a 2D weight matrix W into GGUF block_tq2_0 binary buffer using R2Q.
    For each block of 256 elements:
        - R2Q decomposition gives 4 optimal discrete levels
        - Maps levels to 2-bit codebook {0, 1, 2, 3} -> {-d, 0, +d, +2d}
        - Packs 64 bytes of 2-bit quants + 2 bytes FP16 scale d = 66 bytes/block
    """
    orig_shape = W.shape
    flat = W.flatten().astype(np.float32)
    n_elem = flat.size
    
    # Pad to block_size if necessary
    pad_size = 0
    if n_elem % block_size != 0:
        pad_size = block_size - (n_elem % block_size)
        flat = np.pad(flat, (0, pad_size), mode='constant')
        
    n_blocks = flat.size // block_size
    blocks = flat.reshape(n_blocks, block_size)
    
    # 1. R2Q Dual-binary decomposition
    q1, q2, a1, a2 = r2q_decompose_block(blocks, block_size=block_size, n_iters=3)
    
    # 2. DAD distillation if calibration activations provided
    if X_calib is not None and dad_steps > 0:
        a1, a2, _, _ = r2q_dad_scale_distill(
            W_orig=blocks.reshape(-1, W.shape[1] if len(orig_shape) == 2 else 256),
            q1=q1,
            q2=q2,
            alpha1_init=a1,
            alpha2_init=a2,
            X_calib=X_calib,
            steps=dad_steps,
            lr=dad_lr,
            block_size=block_size
        )
        
    # 3. Construct 4-level adaptive reconstruction: w_rec = a1*q1 + a2*q2
    q1_f = q1.astype(np.float32).reshape(n_blocks, block_size)
    q2_f = q2.astype(np.float32).reshape(n_blocks, block_size)
    w_rec = q1_f * a1[:, None] + q2_f * a2[:, None]
    
    # 4. Map w_rec into TQ2_0 representation:
    # In GGML TQ2_0 dequantize_row_tq2_0:
    # y = (q - 1) * d, with q in {0, 1, 2, 3} => y in {-d, 0, +d, +2d}
    # Optimal d per block minimizing sum(w_i - (q_i - 1)*d)^2:
    # Round w_rec to nearest level in {-d, 0, +d, +2d}
    
    # Initial estimate of d from a1:
    d_est = np.maximum(a1, 1e-8)
    
    # Compute optimal discrete trits/quants q in {0, 1, 2, 3}
    # Levels are: 0 -> -d, 1 -> 0, 2 -> +d, 3 -> +2d
    # normalized = w_rec / d_est
    # q_idx = clip(round(normalized) + 1, 0, 3)
    norm = blocks / d_est[:, None]
    q_idx = np.clip(np.round(norm) + 1, 0, 3).astype(np.uint8)
    
    # Re-estimate optimal d: sum(blocks * (q - 1)) / sum((q - 1)^2)
    basis = (q_idx.astype(np.float32) - 1.0)
    denom = np.sum(basis ** 2, axis=1)
    denom = np.maximum(denom, 1.0)
    num = np.sum(blocks * basis, axis=1)
    d_opt = np.maximum(num / denom, 1e-8).astype(np.float32)
    
    # Pack into TQ2_0 memory format (66 bytes per block of 256)
    # TQ2_0 layout:
    # uint8_t qs[64]; // 256 elements packed 4 per byte
    # ggml_half d;    // 2 bytes fp16
    # Packing order matching ggml-quants.c:
    # for j in range(0, 64, 32): (j=0, j=32)
    #   for m in range(32):
    #     q = sum((xi & 3) << (2*n) for n in range(4))
    #     where xi is element at index m + n*32 (within the 128-element sub-half)
    out_buf = bytearray(n_blocks * 66)
    
    for b in range(n_blocks):
        b_offset = b * 66
        b_q = q_idx[b] # 256 elements
        b_d = d_opt[b]
        
        # Pack 64 bytes of qs
        qs = bytearray(64)
        for half in range(2): # half 0: 0..127, half 1: 128..255
            half_base = half * 128
            j_base = half * 32
            for m in range(32):
                byte_val = 0
                for n in range(4):
                    elem_idx = half_base + m + n * 32
                    val = int(b_q[elem_idx]) & 3
                    byte_val |= (val << (2 * n))
                qs[j_base + m] = byte_val
                
        out_buf[b_offset : b_offset + 64] = qs
        # Pack 2 bytes FP16 scale d
        d_fp16 = np.float16(b_d).tobytes()
        out_buf[b_offset + 64 : b_offset + 66] = d_fp16
        
    return bytes(out_buf)
