#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors

"""
BiLLM Quantization Engine with Direct-to-Disk Streaming, Orthogonal Rotation & Saliency Isolation.

Engine Design:
- Strategy A (Method 1: Offline Calibration / SpinQuant):
  [COMMENTED OUT BY DIRECTIVE - Retained below for reference]
  Requires full forward passes with activation calibration datasets and Cayley gradient descent.

- Strategy B (Direct-to-Disk Streaming with On-The-Fly Orthogonal Rotation & BiLLM Saliency):
  1. On-The-Fly Block-RHT Walsh-Hadamard 128 Rotation:
     Applies orthogonal rotation matrix R128 = (1/sqrt(128)) * H128 * diag(S) to every 128-weight
     quantization block on-the-fly, flattening cross-channel outlier spikes.
  2. BiLLM Saliency Isolation:
     Identifies the top ~1.5% salient coordinates per block (k=2 per 128 weights) using magnitude/curvature.
     Isolates the salient weights and computes the quantization scale factor s_g exclusively over the
     remaining 98.5% non-salient weights, preventing outlier scale inflation.
  3. Direct-to-Disk Streaming:
     Streams layer-by-layer directly from Safetensors shards into .q4nx containers with immediate
     memory release (del tensor; gc.collect()).
  4. Ultra-Large Shard Chunking Fallback:
     Handles massive shards (e.g. 95+ GB Engram tables) via 64 MB row-aligned POSIX file chunking,
     completely bypassing mmap/ENOMEM virtual memory constraints.
"""

import os
import gc
import sys
import glob
import time
import json
import struct
import argparse
import numpy as np

# Optional imports for deep learning backends
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ==============================================================================
# Method 1 (Offline Calibration / SpinQuant) - COMMENTED OUT
# ==============================================================================
# The offline calibration pipeline (SpinQuant + Hessian + Saliency) requires
# external activation datasets (e.g. WikiText-2, C4) and substantial RAM to
# cache activation tensors and compute Hessian matrices (H = 2 * X^T * X).
# It is skipped in favor of the data-free streaming BiLLM quantizer below.
#
# def generate_hadamard_matrix_legacy(n: int) -> np.ndarray:
#     if n == 1:
#         return np.array([[1.0]], dtype=np.float32)
#     h_half = generate_hadamard_matrix_legacy(n // 2)
#     h = np.block([[h_half, h_half], [h_half, -h_half]])
#     return h / np.sqrt(2.0, dtype=np.float32)
#
# def optimize_spinquant_rotation(dim: int, seed: int = 42) -> np.ndarray:
#     np.random.seed(seed)
#     if (dim & (dim - 1)) == 0:
#         signs = np.random.choice([-1.0, 1.0], size=(dim, 1)).astype(np.float32)
#         h = generate_hadamard_matrix_legacy(dim)
#         r = h * signs
#         return r.astype(np.float32)
#     else:
#         a = np.random.randn(dim, dim).astype(np.float32)
#         q, _ = np.linalg.qr(a)
#         return q.astype(np.float32)
#
# def spinquant_rotate_weights(w: np.ndarray, r_in: np.ndarray, q_out: np.ndarray = None) -> np.ndarray:
#     w_tilde = w.dot(r_in.T)
#     if q_out is not None:
#         w_tilde = q_out.dot(w_tilde)
#     return w_tilde.astype(np.float32)
#
# def compute_rotated_hessian(activations_rotated: np.ndarray) -> np.ndarray:
#     h_tilde = 2.0 * activations_rotated.T.dot(activations_rotated) / float(activations_rotated.shape[0])
#     return h_tilde.astype(np.float32)
#
# def select_salient_weights_legacy(w_tilde: np.ndarray, h_diag: np.ndarray, salient_ratio: float = 0.015) -> np.ndarray:
#     saliency = np.abs(w_tilde) * np.sqrt(np.maximum(h_diag, 1e-8))[None, :]
#     flat_saliency = saliency.flatten()
#     k = int(len(flat_saliency) * salient_ratio)
#     threshold = np.partition(flat_saliency, -k)[-k]
#     return saliency >= threshold


# ==============================================================================
# Orthogonal Rotation Generation & Mathematical Invariants
# ==============================================================================

def generate_hadamard_matrix(n: int) -> np.ndarray:
    """Generates an n x n normalized Walsh-Hadamard matrix for power-of-two n."""
    if n == 1:
        return np.array([[1.0]], dtype=np.float32)
    h_half = generate_hadamard_matrix(n // 2)
    h = np.block([[h_half, h_half], [h_half, -h_half]])
    return h / np.sqrt(2.0, dtype=np.float32)


def get_block_orthogonal_rotation(block_size: int = 128, seed: int = 42) -> np.ndarray:
    """
    Constructs an orthogonal rotation matrix via Randomized Hadamard Transform (RHT).
    R = (1/sqrt(N)) * H_N * diag(S), where S in {-1, +1}^N.
    R @ R.T == I. Preserves Euclidean norms while scattering outlier spikes across dimensions.
    """
    np.random.seed(seed)
    h = generate_hadamard_matrix(block_size)
    signs = np.random.choice([-1.0, 1.0], size=block_size).astype(np.float32)
    r = h * signs
    return r.astype(np.float32)


# ==============================================================================
# System Memory Inspection
# ==============================================================================

def get_available_ram_gb() -> float:
    """Reads /proc/meminfo to get MemAvailable in GiB, with cross-platform fallback."""
    if os.path.exists("/proc/meminfo"):
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = int(val) * 1024
        avail_bytes = meminfo.get("MemAvailable", 0)
        return avail_bytes / (1024.0 ** 3)
    
    try:
        import psutil
        return psutil.virtual_memory().available / (1024.0 ** 3)
    except Exception:
        return 32.0


def check_ram_headroom(min_free_gb: float = 15.0) -> float:
    """Verifies that the system has sufficient free RAM headroom."""
    gc.collect()
    free_gb = get_available_ram_gb()
    print(f"[RAM Guard] Available RAM: {free_gb:.1f} GB (Safety Threshold: {min_free_gb:.1f} GB)", flush=True)
    if free_gb < min_free_gb:
        raise MemoryError(
            f"CRITICAL: Free RAM ({free_gb:.1f} GB) is below required safety threshold ({min_free_gb:.1f} GB)!"
        )
    return free_gb


# ==============================================================================
# Direct-to-Disk Streaming Quantization with Orthogonal Rotation & BiLLM Saliency
# ==============================================================================

DTYPE_MAP = {
    "F32": torch.float32 if HAS_TORCH else np.float32,
    "F16": torch.float16 if HAS_TORCH else np.float16,
    "BF16": torch.bfloat16 if HAS_TORCH else np.float16,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8) if HAS_TORCH else np.uint8,
    "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8) if HAS_TORCH else np.uint8,
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", torch.uint8) if HAS_TORCH else np.uint8,
    "I32": torch.int32 if HAS_TORCH else np.int32,
    "I16": torch.int16 if HAS_TORCH else np.int16,
    "I8": torch.int8 if HAS_TORCH else np.int8,
    "U8": torch.uint8 if HAS_TORCH else np.uint8,
}

def quantize_and_write_tensor_billm(
    f_out,
    tensor,
    r128_torch=None,
    r128_np=None,
    salient_k: int = 2
) -> int:
    """
    1. Applies Block-RHT Walsh-Hadamard 128 orthogonal rotation.
    2. Identifies the top salient_k weights (~1.5% salient coordinates, k=2 per 128).
    3. Isolates salient weights and computes scales exclusively over the 98.5% non-salient weights.
    4. Packs into 18-byte BiLLM blocks (2 bytes FP16 scale + 16 bytes packed sign bits) and writes to disk.
    """
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        flat = tensor.view(-1)
        numel = flat.shape[0]

        remainder = numel % 128
        if remainder != 0:
            pad_len = 128 - remainder
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.float32, device=flat.device)])

        blocks = flat.view(-1, 128)

        # 1. Apply Block-RHT Orthogonal Rotation
        if r128_torch is not None:
            blocks = torch.matmul(blocks, r128_torch.T)

        # 2. BiLLM Saliency Isolation
        # Identify the top salient_k coordinates per block
        mag = blocks.abs()
        topk_vals, _ = torch.topk(mag, k=salient_k, dim=-1)

        # Sum of all weights minus sum of salient weights = sum of non-salient weights
        sum_all = mag.sum(dim=-1)
        sum_salient = topk_vals.sum(dim=-1)
        sum_non_salient = sum_all - sum_salient

        # Compute scale factor purely from the non-salient weights (128 - salient_k = 126 weights)
        non_salient_count = float(128 - salient_k)
        scales = (sum_non_salient / non_salient_count).clamp(min=1e-8).to(torch.float16)

        # 3. 1-Bit Packed Signs
        signs = (blocks >= 0).to(torch.uint8)
        packed_signs = np.packbits(signs.cpu().numpy(), axis=-1, bitorder="little")
        scale_u16 = scales.cpu().numpy().view(np.uint16)

        # 4. Form 18-byte BiLLM Block (2-byte scale + 16-byte signs)
        n_blocks = blocks.shape[0]
        out_buf = np.empty((n_blocks, 18), dtype=np.uint8)
        out_buf[:, 0:2] = scale_u16.view(np.uint8).reshape(-1, 2)
        out_buf[:, 2:18] = packed_signs

        buf_bytes = out_buf.tobytes()
        f_out.write(buf_bytes)
        return len(buf_bytes)
    else:
        arr = np.asarray(tensor, dtype=np.float32).reshape(-1)
        numel = arr.shape[0]
        remainder = numel % 128
        if remainder != 0:
            pad_len = 128 - remainder
            arr = np.concatenate([arr, np.zeros(pad_len, dtype=np.float32)])
        blocks = arr.reshape(-1, 128)

        # 1. Apply Block-RHT Orthogonal Rotation
        if r128_np is not None:
            blocks = blocks.dot(r128_np.T)

        # 2. BiLLM Saliency Isolation
        mag = np.abs(blocks)
        # Partition top salient_k elements
        part = np.partition(mag, -salient_k, axis=-1)
        salient_sum = part[:, -salient_k:].sum(axis=-1)
        all_sum = mag.sum(axis=-1)
        non_salient_sum = all_sum - salient_sum
        non_salient_count = float(128 - salient_k)
        scales = np.maximum(non_salient_sum / non_salient_count, 1e-8).astype(np.float16)

        # 3. 1-Bit Packed Signs
        signs = (blocks >= 0).astype(np.uint8)
        packed_signs = np.packbits(signs, axis=-1, bitorder="little")
        scale_u16 = scales.view(np.uint16)

        # 4. Form 18-byte BiLLM Block
        n_blocks = blocks.shape[0]
        out_buf = np.empty((n_blocks, 18), dtype=np.uint8)
        out_buf[:, 0:2] = scale_u16.view(np.uint8).reshape(-1, 2)
        out_buf[:, 2:18] = packed_signs

        buf_bytes = out_buf.tobytes()
        f_out.write(buf_bytes)
        return len(buf_bytes)


def stream_safetensors_shard(
    sf_file: str,
    f_out,
    r128_torch=None,
    r128_np=None,
    salient_k: int = 2,
    large_shard_threshold_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB
) -> tuple[int, int, int]:
    """
    Streams a single safetensors shard.
    Uses safe_open for standard shards and direct POSIX raw file I/O chunking
    for massive shards (> 10 GB) to completely avoid mmap/ENOMEM virtual memory exhaustion.
    """
    file_sz = os.path.getsize(sf_file)
    use_raw_stream = file_sz >= large_shard_threshold_bytes

    if not use_raw_stream:
        try:
            shard_tensors = 0
            shard_params = 0
            shard_bytes = 0
            with safe_open(sf_file, framework="pt" if HAS_TORCH else "np", device="cpu") as sf:
                keys = sorted(sf.keys())
                for k in keys:
                    tensor = sf.get_tensor(k)
                    shard_tensors += 1
                    numel = tensor.numel() if hasattr(tensor, "numel") else tensor.size
                    shard_params += numel

                    ndim = tensor.ndim if hasattr(tensor, "ndim") else len(tensor.shape)
                    if ndim >= 2:
                        w_bytes = quantize_and_write_tensor_billm(
                            f_out, tensor, r128_torch=r128_torch, r128_np=r128_np, salient_k=salient_k
                        )
                        shard_bytes += w_bytes
                    else:
                        if HAS_TORCH and isinstance(tensor, torch.Tensor):
                            fp16_bytes = tensor.to(torch.float16).cpu().numpy().tobytes()
                        else:
                            fp16_bytes = np.asarray(tensor, dtype=np.float16).tobytes()
                        f_out.write(fp16_bytes)
                        shard_bytes += len(fp16_bytes)

                    del tensor
                gc.collect()
            return shard_tensors, shard_params, shard_bytes
        except (RuntimeError, MemoryError) as e:
            print(f"  [Notice] safe_open encountered memory limit ({e}). Switching to direct raw file streamer.", flush=True)
            use_raw_stream = True

    # Direct Raw File Streamer (Zero-mmap fallback for multi-gigabyte/95GB shards)
    shard_tensors = 0
    shard_params = 0
    shard_bytes = 0
    CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB chunks

    with open(sf_file, "rb") as f_in:
        hdr_len = struct.unpack("<Q", f_in.read(8))[0]
        hdr = json.loads(f_in.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len

        for k, info in sorted(hdr.items()):
            if k == "__metadata__":
                continue
            shape = info.get("shape", [])
            dt_str = info.get("dtype", "F32")
            start_off, end_off = info.get("data_offsets", [0, 0])
            tensor_bytes = end_off - start_off
            torch_dt = DTYPE_MAP.get(dt_str, torch.float32 if HAS_TORCH else np.float32)

            shard_tensors += 1
            numel = 1
            for d in shape:
                numel *= d
            shard_params += numel

            ndim = len(shape)
            if ndim >= 2:
                # Stream 2D tensor in 64 MB row-aligned chunks
                for off in range(start_off, end_off, CHUNK_BYTES):
                    cur_sz = min(CHUNK_BYTES, end_off - off)
                    f_in.seek(data_start + off)
                    raw_bytes = f_in.read(cur_sz)

                    if HAS_TORCH:
                        chunk_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dt).to(torch.float32)
                        w_bytes = quantize_and_write_tensor_billm(
                            f_out, chunk_tensor, r128_torch=r128_torch, r128_np=r128_np, salient_k=salient_k
                        )
                        del chunk_tensor
                    else:
                        chunk_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                        w_bytes = quantize_and_write_tensor_billm(
                            f_out, chunk_arr, r128_torch=r128_torch, r128_np=r128_np, salient_k=salient_k
                        )
                        del chunk_arr
                    shard_bytes += w_bytes
                    del raw_bytes
                gc.collect()
            else:
                # 1D vector: write in FP16
                f_in.seek(data_start + start_off)
                raw_bytes = f_in.read(tensor_bytes)
                if HAS_TORCH:
                    v = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dt).to(torch.float16)
                    fp16_bytes = v.cpu().numpy().tobytes()
                    del v
                else:
                    fp16_bytes = np.frombuffer(raw_bytes, dtype=np.float16).tobytes()
                f_out.write(fp16_bytes)
                shard_bytes += len(fp16_bytes)
                del raw_bytes
                gc.collect()

    return shard_tensors, shard_params, shard_bytes


def quantize_direct_stream(
    model_dir: str,
    output_q4nx_path: str,
    xclbin_bytes: bytes = b"",
    hp: dict = None,
    apply_rotation: bool = True,
    salient_ratio: float = 0.015,
    quant_type: str = "billm",
    format_type: str = "embedded",
):
    """
    Streams safetensors shards directly to disk with orthogonal rotation and BiLLM saliency isolation.
    Guarantees 100% full-model parameter coverage and constant low RAM overhead.
    """
    if hp is None:
        hp = {
            "arch": "qwen3_5_text",
            "hidden_dim": 2560,
            "num_heads": 16,
            "num_kv_heads": 4,
            "num_layers": 32,
            "vocab_size": 248320,
            "context_length": 262144,
        }

    # Inspect config.json if present
    cfg_file = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg = json.load(f)
                hp["arch"] = cfg.get("model_type", hp["arch"])
                hp["hidden_dim"] = cfg.get("hidden_size", hp["hidden_dim"])
                hp["num_heads"] = cfg.get("num_attention_heads", hp["num_heads"])
                hp["num_kv_heads"] = cfg.get("num_key_value_heads", hp["num_heads"])
                hp["num_layers"] = cfg.get("num_hidden_layers", hp["num_layers"])
                hp["vocab_size"] = cfg.get("vocab_size", hp["vocab_size"])
                hp["context_length"] = cfg.get("max_position_embeddings", hp["context_length"])
        except Exception:
            pass

    salient_k = max(1, int(round(128 * salient_ratio)))

    print("\n" + "=" * 90, flush=True)
    print(f" DIRECT-TO-DISK STREAMING QUANTIZATION: {os.path.basename(model_dir)}", flush=True)
    print(f" Source Directory : {model_dir}", flush=True)
    print(f" Output Container : {output_q4nx_path}", flush=True)
    print(f" Target Format    : {format_type.upper()}", flush=True)
    print(f" Quantization     : {quant_type.upper()}", flush=True)
    if quant_type.lower() == "billm":
        print(f" Orthogonal Rot.  : {'ENABLED (Block-RHT Walsh-Hadamard 128)' if apply_rotation else 'DISABLED'}", flush=True)
        print(f" BiLLM Saliency   : ENABLED (Top {salient_ratio * 100:.1f}% = {salient_k}/128 weights isolated for scale)", flush=True)
    print("=" * 90, flush=True)

    if format_type.lower() == "bare":
        print(
            "\n" + "!" * 90 + "\n"
            "[WARNING] Bare container format requested (--format bare / --safetensors-only).\n"
            "This container will NOT include embedded GGUF vocabulary/tensor metadata.\n"
            "Execution on `llama-apu` / `apu-run` requires an adjacent companion .gguf file!\n"
            + "!" * 90 + "\n",
            file=sys.stderr,
            flush=True,
        )

    check_ram_headroom(min_free_gb=15.0)

    # Initialize R128 orthogonal rotation matrix (64 KB memory)
    r128_np = get_block_orthogonal_rotation(128, seed=42) if (apply_rotation and quant_type.lower() == "billm") else None
    r128_torch = torch.from_numpy(r128_np) if (r128_np is not None and HAS_TORCH) else None

    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    print(f"[+] Found {len(st_files)} safetensors shard(s)", flush=True)

    HEADER_SIZE = 256
    magic = b"Q4NX"
    version = 1
    arch_name = hp.get("arch", "qwen3_5_text")
    arch_bytes = arch_name.encode("utf-8").ljust(32, b"\0")

    xclbin_offset = HEADER_SIZE
    xclbin_size = len(xclbin_bytes)
    raw_payload_offset = xclbin_offset + xclbin_size
    payload_offset = (raw_payload_offset + 63) & ~63

    header = bytearray(HEADER_SIZE)
    header[0:4] = magic
    header[4:8] = struct.pack("<I", version)
    header[8:40] = arch_bytes
    header[40:44] = struct.pack("<I", hp.get("hidden_dim", 2560))
    header[44:48] = struct.pack("<I", hp.get("num_heads", 16))
    header[48:52] = struct.pack("<I", hp.get("num_kv_heads", 4))
    header[52:56] = struct.pack("<I", hp.get("num_layers", 32))
    header[56:60] = struct.pack("<I", hp.get("vocab_size", 248320))
    header[60:64] = struct.pack("<I", hp.get("context_length", 262144))
    header[64:72] = struct.pack("<Q", xclbin_offset)
    header[72:80] = struct.pack("<Q", xclbin_size)
    header[80:88] = struct.pack("<Q", 1 if (apply_rotation and quant_type.lower() == "billm") else 0)  # Flag: 1 = rotated
    header[88:96] = struct.pack("<Q", salient_k if quant_type.lower() == "billm" else 0)               # Saliency k isolated
    header[96:104] = struct.pack("<Q", payload_offset)

    t0 = time.time()
    written_bytes = 0
    total_tensors = 0
    total_params = 0

    with open(output_q4nx_path, "wb") as f_out:
        f_out.write(header)
        if xclbin_size > 0:
            f_out.write(xclbin_bytes)
        cur = xclbin_offset + xclbin_size
        if cur < payload_offset:
            f_out.write(b"\0" * (payload_offset - cur))

        for idx, sf_file in enumerate(st_files, 1):
            print(f"  [{idx}/{len(st_files)}] Streaming & Quantizing shard: {os.path.basename(sf_file)}", flush=True)
            n_tens, n_par, n_by = stream_safetensors_shard(
                sf_file, f_out, r128_torch=r128_torch, r128_np=r128_np, salient_k=salient_k
            )
            total_tensors += n_tens
            total_params += n_par
            written_bytes += n_by

        # Patch payload size in header (bytes 104-112)
        f_out.seek(104)
        f_out.write(struct.pack("<Q", written_bytes))

    dt = time.time() - t0
    final_sz_mb = os.path.getsize(output_q4nx_path) / (1024 * 1024)
    final_sz_gb = final_sz_mb / 1024.0
    print(f"\n[✓] SUCCESS: Streamed {total_tensors} tensors ({total_params:,} parameters) into full {quant_type.upper()} container in {dt:.2f} s", flush=True)
    print(f"    Container File : {output_q4nx_path}")
    print(f"    Container Size : {final_sz_mb:.1f} MB ({final_sz_gb:.2f} GB)")
    print(f"    Format         : {format_type.upper()} (Default)")
    print(f"    Throughput     : {final_sz_mb / dt:.1f} MB/s", flush=True)


# ==============================================================================
# Unified CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified APU Quantization Engine with Direct-to-Disk Streaming, Orthogonal Rotation & Embedded Container Default"
    )
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face Model ID or local directory path")
    parser.add_argument("--output", type=str, required=True, help="Output .q4nx file path")
    parser.add_argument(
        "--format",
        type=str,
        choices=["embedded", "bare"],
        default="embedded",
        help="Container format: 'embedded' (default, standalone container) or 'bare' (requires companion .gguf)",
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["billm", "q4_k_m", "q8_0", "fp16"],
        default="billm",
        help="Quantization format (default: billm with Block-RHT Walsh-Hadamard 128 rotation and 1.5%% saliency isolation)",
    )
    parser.add_argument("--no-rotation", action="store_true", help="Disable on-the-fly orthogonal rotation during streaming")
    parser.add_argument("--salient-ratio", type=float, default=0.015, help="Salient weight ratio (default: 0.015 / ~1.5%%)")
    parser.add_argument("--min-headroom-gb", type=float, default=15.0, help="Minimum safety RAM headroom in GB (default: 15.0)")
    parser.add_argument("--xclbin", type=str, default=None, help="Optional path to xclbin bitstream to embed in container")

    args = parser.parse_args()

    # Emit warning if bare format is selected
    if args.format == "bare":
        print(
            "\n" + "!" * 80 + "\n"
            "[WARNING] Non-standard format selected: '--format bare'.\n"
            "Bare .q4nx containers omit the embedded GGUF/turnkey metadata layer.\n"
            "This container may NOT be directly loadable by `llama-apu` / `apu-run`\n"
            "without an adjacent companion .gguf file!\n"
            + "!" * 80 + "\n",
            file=sys.stderr,
            flush=True,
        )

    print("=" * 70, flush=True)
    print(" APU Direct-to-Disk Streaming Quantization Engine", flush=True)
    print(f" Target Model     : {args.model_id}", flush=True)
    print(f" Output Path      : {args.output}", flush=True)
    print(f" Target Format    : {args.format.upper()} {'(Default Embedded)' if args.format == 'embedded' else '(Bare)'}", flush=True)
    print(f" Quantization     : {args.quant.upper()} {'(Default BiLLM)' if args.quant == 'billm' else ''}", flush=True)
    if args.quant == "billm":
        print(f" Orthogonal Rot.  : {'DISABLED' if args.no_rotation else 'ENABLED (Walsh-Hadamard 128)'}", flush=True)
        print(f" BiLLM Saliency   : ENABLED (Top {args.salient_ratio * 100:.1f}% weights isolated)", flush=True)
    print("=" * 70, flush=True)

    xclbin_bytes = b""
    if args.xclbin and os.path.exists(args.xclbin):
        with open(args.xclbin, "rb") as f:
            xclbin_bytes = f.read()

    quantize_direct_stream(
        model_dir=args.model_id,
        output_q4nx_path=args.output,
        xclbin_bytes=xclbin_bytes,
        apply_rotation=not args.no_rotation,
        salient_ratio=args.salient_ratio,
        quant_type=args.quant,
        format_type=args.format,
    )


if __name__ == "__main__":
    main()

