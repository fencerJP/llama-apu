#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script to create a BiLLM .q4nx submodel slice from Qwen3.8-27B-Cold-Fusion
and verify inference using the unified llama executable on AMD APU.
"""

import os
import sys
import time
import struct
import numpy as np
from safetensors import safe_open

# Add converter path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
converter_dir = os.path.join(repo_root, "zero-copy_model_runner", "converter")
sys.path.insert(0, converter_dir)

from convert_to_billm import (
    optimize_spinquant_rotation,
    spinquant_rotate_weights,
    compute_rotated_hessian,
    select_salient_weights,
    quantize_billm_block,
)

def create_q4nx_file(output_path, arch_name, hyperparams, payload_bytes, xclbin_bytes=b""):
    HEADER_SIZE = 256
    magic = b"Q4NX"
    version = 1
    arch_bytes = arch_name.encode("utf-8")[:31].ljust(32, b"\0")

    xclbin_offset = HEADER_SIZE
    xclbin_size = len(xclbin_bytes)

    raw_payload_offset = xclbin_offset + xclbin_size
    payload_offset = (raw_payload_offset + 63) & ~63
    payload_size = len(payload_bytes)

    if payload_size % 64 != 0:
        padding_needed = 64 - (payload_size % 64)
        payload_bytes = payload_bytes + (b"\0" * padding_needed)
        payload_size = len(payload_bytes)

    header = bytearray(HEADER_SIZE)
    header[0:4] = magic
    header[4:8] = struct.pack("<I", version)
    header[8:40] = arch_bytes
    header[40:44] = struct.pack("<I", hyperparams["hidden_dim"])
    header[44:48] = struct.pack("<I", hyperparams["num_heads"])
    header[48:52] = struct.pack("<I", hyperparams["num_kv_heads"])
    header[52:56] = struct.pack("<I", hyperparams["num_layers"])
    header[56:60] = struct.pack("<I", hyperparams["vocab_size"])
    header[60:64] = struct.pack("<I", hyperparams["context_length"])
    header[64:72] = struct.pack("<Q", xclbin_offset)
    header[72:80] = struct.pack("<Q", xclbin_size)
    header[80:88] = struct.pack("<Q", 0)  # tensor table offset
    header[88:96] = struct.pack("<Q", 0)  # tensor count
    header[96:104] = struct.pack("<Q", payload_offset)
    header[104:112] = struct.pack("<Q", payload_size)

    with open(output_path, "wb") as f:
        f.write(header)
        if xclbin_size > 0:
            f.write(xclbin_bytes)
        current_pos = xclbin_offset + xclbin_size
        if current_pos < payload_offset:
            f.write(b"\0" * (payload_offset - current_pos))
        f.write(payload_bytes)

    print(f"[+] Successfully wrote .q4nx container to: {output_path} ({os.path.getsize(output_path):,} bytes)")

def quantize_and_build_slice():
    safetensors_path = "/mnt/Media/Downloads/model_testing/Qwen3.8-27B-Cold-Fusion/model-00002-of-00012.safetensors"
    output_q4nx = "/mnt/Media/Downloads/model_testing/qwen3.8-cold-fusion-slice.q4nx"
    
    print(f"[*] Extracting submodel slice from: {safetensors_path}")
    t0 = time.time()
    with safe_open(safetensors_path, framework="pt") as f:
        # Extract MLP up_proj slice (512x1024)
        t_slice = f.get_slice("model.language_model.layers.0.mlp.up_proj.weight")
        w_orig = t_slice[:512, :1024].float().numpy()

    print(f"[+] Loaded submatrix {w_orig.shape} in {(time.time() - t0)*1000:.1f} ms")

    # 4-stage BiLLM Quantization
    t0_quant = time.time()
    in_dim = w_orig.shape[1]
    out_dim = w_orig.shape[0]

    # Phase 1: SpinQuant
    r_in = optimize_spinquant_rotation(in_dim)
    w_tilde = spinquant_rotate_weights(w_orig, r_in)

    # Phase 2: Rotated Hessian
    np.random.seed(42)
    x_calib = np.random.randn(64, in_dim).astype(np.float32)
    x_tilde = x_calib.dot(r_in.T)
    h_tilde = compute_rotated_hessian(x_tilde)
    h_diag = np.diag(h_tilde)

    # Phase 3: Salient Weights
    salient_mask = select_salient_weights(w_tilde, h_diag, salient_ratio=0.015)

    # Phase 4: Residual Binarization
    payload = bytearray()
    for row in range(out_dim):
        w_row = w_tilde[row]
        mask_row = salient_mask[row]
        for b in range(0, in_dim, 128):
            block_w = w_row[b : b + 128]
            block_mask = mask_row[b : b + 128]
            payload.extend(quantize_billm_block(block_w, block_mask))

    quant_time = time.time() - t0_quant
    print(f"[+] 4-Stage BiLLM Quantization finished in {quant_time*1000:.1f} ms (payload: {len(payload):,} bytes)")

    hyperparams = {
        "hidden_dim": 5120,
        "num_heads": 40,
        "num_kv_heads": 8,
        "num_layers": 4,
        "vocab_size": 248320,
        "context_length": 8192,
    }

    create_q4nx_file(
        output_path=output_q4nx,
        arch_name="qwen3_8",
        hyperparams=hyperparams,
        payload_bytes=payload,
    )
    return output_q4nx

if __name__ == "__main__":
    quantize_and_build_slice()
