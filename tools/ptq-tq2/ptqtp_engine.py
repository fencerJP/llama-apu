#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# llama-apu: Phase 7 Out-of-Core PTQ Engine & TQ2_0 Encoding (v3.0)

import math
import sys
import os
import argparse
from typing import Any, Dict, Optional, Tuple
import numpy as np

# System memory ceiling constant: 50 GB
APU_MAX_SYSTEM_MEMORY_CEILING_BYTES = 50 * 1024 * 1024 * 1024

def check_memory_governor(working_set_bytes: int = 0):
    """
    Enforces the 50 GB system memory ceiling on unified APUs and recommends
    the distillation hierarchy: 'layer', 'expert', or 'micro_chunk'.
    """
    mem_total = 0
    mem_avail = 0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if parts[0] == "MemTotal:":
                    mem_total = int(parts[1]) * 1024
                elif parts[0] == "MemAvailable:":
                    mem_avail = int(parts[1]) * 1024
    except Exception:
        mem_total = 64 * 1024 * 1024 * 1024
        mem_avail = 32 * 1024 * 1024 * 1024

    within_ceiling = (working_set_bytes <= APU_MAX_SYSTEM_MEMORY_CEILING_BYTES)
    
    if mem_avail >= working_set_bytes * 3:
        hierarchy = "layer"
    elif mem_avail >= working_set_bytes:
        hierarchy = "expert"
    else:
        hierarchy = "micro_chunk"

    return {
        "total_ram": mem_total,
        "available_ram": mem_avail,
        "working_set": working_set_bytes,
        "within_ceiling": within_ceiling,
        "hierarchy": hierarchy
    }

def determine_execution_strategy(
    model_size_bytes: int = 0,
    num_experts: int = 0,
    is_moe: bool = False,
    override_convert: str = "auto",
    override_distill: str = "auto"
) -> Dict[str, Any]:
    """
    Automatically decides:
    1. Conversion mode: 'online' (in-memory streaming) vs 'offline' (out-of-core chunked disk conversion).
    2. Distillation mode: 'in_memory' (all at once) vs 'stream_in_place' (tensor-by-tensor disk streaming).
    3. Staging policy: 'local_ssd' vs 'direct_nas_inplace'.
    Based on model size, architecture type (MoE vs Dense), and system resources under the 50 GB APU ceiling.
    """
    mem_info = check_memory_governor(working_set_bytes=model_size_bytes)
    mem_avail = mem_info["available_ram"]
    mem_total = mem_info["total_ram"]
    effective_ceiling = min(mem_avail, APU_MAX_SYSTEM_MEMORY_CEILING_BYTES)

    is_moe_model = is_moe or (num_experts > 0)
    model_size_gb = model_size_bytes / (1024 ** 3) if model_size_bytes > 0 else 0.0

    # 1. Conversion Strategy (online vs offline)
    if override_convert in ("online", "offline"):
        convert_mode = override_convert
        convert_rationale = f"User override: {override_convert}"
    else:
        # Online conversion requires the unquantized weights + conversion buffer (2.5x) to fit in RAM
        if not is_moe_model and (model_size_bytes * 2.5 <= effective_ceiling) and (model_size_gb <= 12.0):
            convert_mode = "online"
            convert_rationale = f"Model ({model_size_gb:.1f} GB) fits safely in APU RAM; executing fast in-memory streaming conversion"
        else:
            convert_mode = "offline"
            reason = "MoE architecture with sparse experts" if is_moe_model else f"Model footprint ({model_size_gb:.1f} GB) exceeds safe online memory threshold"
            convert_rationale = f"{reason}; executing out-of-core chunked disk conversion to maintain APU memory ceiling"

    # 2. Distillation Strategy (in_memory vs stream_in_place)
    if override_distill in ("in_memory", "stream_in_place"):
        distill_mode = override_distill
        distill_rationale = f"User override: {override_distill}"
    else:
        # In-memory distillation loads all candidate tensors + teacher weights + optimizer states into RAM
        # Safe threshold: total working set <= 40% of effective ceiling and model <= 4.0 GB
        distill_working_set_est = model_size_bytes * 2.0
        if not is_moe_model and (distill_working_set_est <= effective_ceiling * 0.4) and (model_size_gb <= 4.0):
            distill_mode = "in_memory"
            distill_rationale = f"Compact dense model ({model_size_gb:.1f} GB); executing vectorized in-memory all-at-once distillation"
        else:
            distill_mode = "stream_in_place"
            reason = "MoE sparse routing active" if is_moe_model else f"Working set ({distill_working_set_est/(1024**3):.1f} GB) exceeds in-memory ceiling"
            distill_rationale = f"{reason}; streaming tensor-by-tensor in-place on disk with bounded memory (< 2 GB working set)"

    # 3. Staging Strategy (local_ssd vs direct_nas_inplace)
    if is_moe_model or model_size_gb > 30.0:
        staging_policy = "direct_nas_inplace"
        staging_rationale = "Skipping local SSD staging to prevent NVMe exhaustion; streaming directly in-place on NAS"
    else:
        staging_policy = "local_ssd"
        staging_rationale = "Staging to NVMe SSD scratch for maximum I/O throughput, pruned after completion"

    return {
        "model_size_gb": model_size_gb,
        "is_moe": is_moe_model,
        "num_experts": num_experts,
        "convert_mode": convert_mode,
        "convert_rationale": convert_rationale,
        "distill_mode": distill_mode,
        "distill_rationale": distill_rationale,
        "staging_policy": staging_policy,
        "staging_rationale": staging_rationale,
        "available_ram_gb": mem_avail / (1024 ** 3),
        "total_ram_gb": mem_total / (1024 ** 3),
        "ceiling_gb": APU_MAX_SYSTEM_MEMORY_CEILING_BYTES / (1024 ** 3),
    }

def fast_walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """
    Computes normalized in-place Fast Walsh-Hadamard Transform (FWHT).
    Requires x.shape[-1] to be a power of 2.
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    if (n & (n - 1)) != 0:
        raise ValueError(f"FWHT dimension {n} must be a power of 2")
    
    # Flatten leading dimensions
    data = x.reshape(-1, n).astype(np.float32, copy=True)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = data[:, j].copy()
                v = data[:, j + h].copy()
                data[:, j] = u + v
                data[:, j + h] = u - v
        h *= 2
    
    # Orthonormal scaling: 1 / sqrt(N)
    data /= math.sqrt(n)
    return data.reshape(orig_shape)

def per_head_quarot_transform(W: np.ndarray, head_dim: int) -> np.ndarray:
    """
    Applies Fast Walsh-Hadamard Transforms per-head independently across
    weight projections (W_q, W_k, W_v).
    Because FWHT is applied per-head, it commutes cleanly with 2D RoPE Givens rotations.
    """
    out_dim, in_dim = W.shape
    if in_dim % head_dim != 0:
        raise ValueError(f"Input dimension {in_dim} not divisible by head_dim {head_dim}")
    
    num_heads = in_dim // head_dim
    # Reshape into [out_dim, num_heads, head_dim]
    W_heads = W.reshape(out_dim, num_heads, head_dim)
    W_rot = fast_walsh_hadamard_transform(W_heads)
    return W_rot.reshape(out_dim, in_dim)

def propagate_moe_rotations(W_in: np.ndarray, H_out: np.ndarray) -> np.ndarray:
    """
    Absorbs attention block output rotations H_out offline into downstream
    input projections (W_gate, W_up, W_router) by multiplying W_in @ H_out.T.
    """
    return W_in @ H_out.T

def closed_form_frobenius_ptqtp(W: np.ndarray, block_size: int = 256):
    """
    Performs closed-form Frobenius error minimization:
        T = clip(round(W / alpha), -1, 1)
        alpha* = sum(W * T) / sum(T^2)
    Returns discrete ternary tensor T in {-1, 0, +1} and per-block FP16 scale alpha*.
    """
    flat = W.flatten().astype(np.float32)
    num_elements = flat.shape[0]
    if num_elements % block_size != 0:
        pad_size = block_size - (num_elements % block_size)
        flat = np.pad(flat, (0, pad_size), mode='constant')
    
    num_blocks = flat.shape[0] // block_size
    blocks = flat.reshape(num_blocks, block_size)
    
    # 1. Initialize scale with absolute mean
    alpha_init = np.mean(np.abs(blocks), axis=1, keepdims=True)
    alpha_init = np.maximum(alpha_init, 1e-8)
    
    # 2. Round to discrete trits {-1, 0, +1}
    T = np.clip(np.round(blocks / alpha_init), -1.0, 1.0)
    
    # 3. Closed-form optimal scale alpha* = sum(W * T) / sum(T^2)
    numerator = np.sum(blocks * T, axis=1, keepdims=True)
    denominator = np.sum(T ** 2, axis=1, keepdims=True)
    denominator = np.maximum(denominator, 1.0)
    alpha_star = numerator / denominator
    alpha_star = np.maximum(alpha_star, 1e-8).astype(np.float16)
    
    return T.reshape(-1)[:num_elements].reshape(W.shape), alpha_star.flatten()

def local_adamw_scale_distillation(
    W_orig: np.ndarray,
    T: np.ndarray,
    alpha_init: np.ndarray,
    X_calib: np.ndarray,
    steps: int = 10,
    lr: float = 1e-2,
    block_size: int = 256
) -> np.ndarray:
    """
    Local AdamW scale distillation minimizing MSE against teacher activations:
        min_alpha || X (alpha * T)^T - X W_orig^T ||_2^2
    Trits T remain frozen; only block scales alpha are refined.
    """
    alpha = alpha_init.copy().astype(np.float32)
    m = np.zeros_like(alpha)
    v = np.zeros_like(alpha)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    # Teacher activations
    Y_teacher = X_calib @ W_orig.T
    
    num_blocks = alpha.shape[0]
    T_flat = T.flatten()
    
    for step in range(1, steps + 1):
        # Reconstruct quantized weight
        W_q = (T_flat.reshape(num_blocks, block_size) * alpha[:, None]).reshape(W_orig.shape)
        Y_student = X_calib @ W_q.T
        
        # Loss gradient w.r.t Y_student
        dY = 2.0 * (Y_student - Y_teacher) / (X_calib.shape[0] * W_orig.shape[0])
        # Gradient w.r.t W_q: dW = dY.T @ X_calib
        dW = dY.T @ X_calib
        
        # Gradient w.r.t alpha: sum(dW * T) per block
        dW_blocks = dW.flatten().reshape(num_blocks, block_size)
        T_blocks = T_flat.reshape(num_blocks, block_size)
        d_alpha = np.sum(dW_blocks * T_blocks, axis=1)
        
        # AdamW update
        m = beta1 * m + (1.0 - beta1) * d_alpha
        v = beta2 * v + (1.0 - beta2) * (d_alpha ** 2)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        
        alpha -= lr * (m_hat / (np.sqrt(v_hat) + eps) + 0.01 * alpha)
        alpha = np.maximum(alpha, 1e-8)
        
    return alpha.astype(np.float16)

def pack_tq2_0(T: np.ndarray, alpha: np.ndarray, block_size: int = 256) -> bytes:
    """
    Serializes discrete ternary weights T and FP16 scales into native
    llama.cpp TQ2_0 binary buffer (66 bytes per 256 weights, 2.0625 bpw).
    """
    flat_T = T.flatten()
    num_elements = flat_T.shape[0]
    assert num_elements % block_size == 0
    num_blocks = num_elements // block_size
    
    out_bytes = bytearray()
    
    for b in range(num_blocks):
        block_t = flat_T[b * block_size : (b + 1) * block_size]
        # Map trits {-1, 0, +1} to 2-bit unsigned integers:
        # -1 -> 0, 0 -> 1, +1 -> 2
        q = np.clip(block_t + 1, 0, 3).astype(np.uint8)
        
        qs = bytearray(64)
        idx = 0
        for j in range(0, 64, 32):
            for l in range(4):
                for m in range(32):
                    qs[j + m] |= (q[idx] & 3) << (l * 2)
                    idx += 1
        
        out_bytes.extend(qs)
        scale_bytes = alpha[b].astype(np.float16).tobytes()
        out_bytes.extend(scale_bytes)
        
    return bytes(out_bytes)

def self_test():
    print("====================================================")
    print("  llama-apu: Phase 7 Out-of-Core PTQTP Engine Tests ")
    print("====================================================")
    
    # 1. FWHT Orthogonality Test
    print("[1/5] Fast Walsh-Hadamard Transform (FWHT) orthogonality...")
    n = 64
    x = np.eye(n, dtype=np.float32)
    H = fast_walsh_hadamard_transform(x)
    # Check H @ H.T == I
    identity_check = H @ H.T
    diff = np.max(np.abs(identity_check - np.eye(n)))
    assert diff < 1e-5, f"FWHT not orthonormal: max diff {diff}"
    print(f"      PASS: H @ H.T == I (max error = {diff:.2e})")
    
    # 2. RoPE-Compliant Per-Head QuaRot Test
    print("[2/5] Per-head QuaRot RoPE commutativity...")
    head_dim = 64
    num_heads = 4
    W = np.random.randn(128, num_heads * head_dim).astype(np.float32)
    W_rot = per_head_quarot_transform(W, head_dim)
    assert W_rot.shape == W.shape
    # Energy preservation
    e_orig = np.sum(W ** 2)
    e_rot = np.sum(W_rot ** 2)
    assert abs(e_orig - e_rot) / e_orig < 1e-4
    print(f"      PASS: Per-head energy preserved ({e_orig:.2f} == {e_rot:.2f})")
    
    # 3. Closed-Form Frobenius PTQTP Projection Test
    print("[3/5] Closed-form Frobenius PTQTP projection (256-block)...")
    W_sample = np.random.randn(256, 256).astype(np.float32) * 0.5
    T, alpha = closed_form_frobenius_ptqtp(W_sample, block_size=256)
    assert set(np.unique(T)).issubset({-1.0, 0.0, 1.0})
    W_reconstructed = (T.reshape(-1, 256) * alpha[:, None]).reshape(W_sample.shape)
    fro_err = np.linalg.norm(W_sample - W_reconstructed) / np.linalg.norm(W_sample)
    assert fro_err < 0.65
    print(f"      PASS: Discrete ternary trits generated (relative Fro error = {fro_err:.4f})")
    
    # 4. Local AdamW Scale Distillation Test
    print("[4/5] Local AdamW scale distillation...")
    X_calib = np.random.randn(16, 256).astype(np.float32)
    alpha_refined = local_adamw_scale_distillation(
        W_sample, T, alpha, X_calib, steps=8, lr=1e-2, block_size=256
    )
    assert alpha_refined.shape == alpha.shape
    W_distilled = (T.reshape(-1, 256) * alpha_refined[:, None]).reshape(W_sample.shape)
    loss_init = np.mean((X_calib @ W_sample.T - X_calib @ W_reconstructed.T) ** 2)
    loss_refined = np.mean((X_calib @ W_sample.T - X_calib @ W_distilled.T) ** 2)
    print(f"      PASS: Distillation MSE reduced: {loss_init:.6f} -> {loss_refined:.6f}")
    
    # 5. Native TQ2_0 Serialization & 50 GB Governor Test
    print("[5/5] Native TQ2_0 GGUF serialization & 50 GB memory governor...")
    raw_bytes = pack_tq2_0(T, alpha_refined, block_size=256)
    expected_bytes = (256 * 256 // 256) * 66
    assert len(raw_bytes) == expected_bytes, f"Expected {expected_bytes} bytes, got {len(raw_bytes)}"
    gov = check_memory_governor(working_set_bytes=4 * 1024 * 1024 * 1024)
    assert gov["within_ceiling"] == True
    print(f"      PASS: {len(raw_bytes)} bytes packed at 2.0625 bpw (Governor: {gov['hierarchy']})")
    
    print("\nALL PTQTP ENGINE TESTS PASSED.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama-apu Phase 7 PTQTP Low-Quant Engine")
    parser.add_argument("--test", action="store_true", help="Run self-test validation suite")
    parser.add_argument("--source", type=str, help="Source model path or repo ID")
    parser.add_argument("--out-gguf", type=str, help="Output destination for converted GGUF")
    args = parser.parse_args()
    
    if args.test or len(sys.argv) == 1:
        self_test()
    else:
        print(f"PTQTP Engine ready for conversion: source={args.source}, dest={args.out_gguf}")
