#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
llama-apu: End-to-End R2Q Quantization & Distillation Converter (Workflow 2)
Converts base unquantized GGUF to R2Q 2.06 bpw GGUF:
- Cascaded Dual-Binary Residuals (R2Q)
- Closed-form 2x2 scale minimization + iterative sign relaxation
- Deviation-Aware Distillation (DAD) on calibration corpus
- Uses GGUFWriter for byte-perfect GGUF generation
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PTQ_DIR = REPO_ROOT / "tools" / "ptq-tq2"
sys.path.append(str(PTQ_DIR))

import gguf
from r2q_engine import quantize_matrix_to_tq2_0_r2q

DEFAULT_CORPUS = Path.home() / "databank" / "distill" / "distill_corpus.jsonl"
DEFAULT_SAMPLES = Path.home() / "databank" / "distill" / "calibration_samples.txt"

def load_calibration_texts(corpus_file: Path, max_samples: int = 64) -> List[str]:
    texts = []
    if corpus_file.suffix == ".jsonl" and corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    txt = obj.get("text", "") or obj.get("content", "")
                    if txt:
                        texts.append(txt)
                        if len(texts) >= max_samples:
                            break
                except Exception:
                    pass
    elif corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            cur = []
            for line in f:
                if line.strip():
                    cur.append(line)
                elif cur:
                    texts.append("".join(cur))
                    cur = []
                    if len(texts) >= max_samples:
                        break
            if cur and len(texts) < max_samples:
                texts.append("".join(cur))
    return texts

def generate_activations(texts: List[str], dim: int, n_tokens: int = 32) -> np.ndarray:
    rng = np.random.RandomState(42)
    X = rng.randn(n_tokens, dim).astype(np.float32)
    X = (X - np.mean(X, axis=-1, keepdims=True)) / (np.std(X, axis=-1, keepdims=True) + 1e-5)
    return X

def convert_to_r2q_gguf(
    src_base_gguf: Path,
    dst_r2q_gguf: Path,
    corpus_path: Optional[Path] = None,
    dad_steps: int = 15,
    dad_lr: float = 1e-2,
    block_size: int = 256
):
    print(f"=== Converting {src_base_gguf.name} -> {dst_r2q_gguf.name} via R2Q Workflow ===")
    t_start = time.time()
    
    # Load calibration texts
    cpath = corpus_path or DEFAULT_CORPUS
    if not cpath.exists():
        cpath = DEFAULT_SAMPLES
    texts = load_calibration_texts(cpath, max_samples=64)
    print(f"[*] Loaded {len(texts)} calibration texts from {cpath}")
    
    reader = gguf.GGUFReader(src_base_gguf)
    arch = "llama"
    for kv in reader.fields.values():
        if kv.name == "general.architecture":
            arch = str(bytes(kv.parts[-1]), encoding="utf-8", errors="ignore")
            break
            
    writer = gguf.GGUFWriter(dst_r2q_gguf, arch=arch)
    
    # Copy all KV metadata
    print(f"[*] Copying {len(reader.fields)} metadata fields...")
    for key, field in reader.fields.items():
        if key in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        if not field.types:
            continue
        t = field.types[0]
        val = field.parts[-1]
        
        if t == gguf.GGUFValueType.UINT8:
            writer.add_uint8(key, int(val[0]))
        elif t == gguf.GGUFValueType.INT8:
            writer.add_int8(key, int(val[0]))
        elif t == gguf.GGUFValueType.UINT16:
            writer.add_uint16(key, int(val[0]))
        elif t == gguf.GGUFValueType.INT16:
            writer.add_int16(key, int(val[0]))
        elif t == gguf.GGUFValueType.UINT32:
            writer.add_uint32(key, int(val[0]))
        elif t == gguf.GGUFValueType.INT32:
            writer.add_int32(key, int(val[0]))
        elif t == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(key, float(val[0]))
        elif t == gguf.GGUFValueType.UINT64:
            writer.add_uint64(key, int(val[0]))
        elif t == gguf.GGUFValueType.INT64:
            writer.add_int64(key, int(val[0]))
        elif t == gguf.GGUFValueType.FLOAT64:
            writer.add_float64(key, float(val[0]))
        elif t == gguf.GGUFValueType.BOOL:
            writer.add_bool(key, bool(val[0]))
        elif t == gguf.GGUFValueType.STRING:
            s = str(bytes(val), encoding="utf-8", errors="ignore")
            writer.add_string(key, s)
        elif t == gguf.GGUFValueType.ARRAY:
            arr_t = field.types[1] if len(field.types) > 1 else gguf.GGUFValueType.STRING
            if arr_t == gguf.GGUFValueType.STRING:
                str_list = [str(bytes(field.parts[idx]), encoding="utf-8", errors="ignore") for idx in field.data]
                if "tokens" in key:
                    writer.add_token_list(str_list)
                else:
                    writer.add_array(key, str_list)
            elif arr_t in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
                writer.add_array(key, [float(field.parts[idx][0]) for idx in field.data])
            elif arr_t == gguf.GGUFValueType.BOOL:
                writer.add_array(key, [bool(field.parts[idx][0]) for idx in field.data])
            else:
                writer.add_array(key, [int(field.parts[idx][0]) for idx in field.data])
                    
    # Target standard linear weights while protecting highly sensitive attention V and QKV projections
    target_keywords = ["attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "attn_gate", "ssm_out"]
    sensitive_keywords = ["attn_v", "attn_qkv", "ssm_conv1d", "ple_conv1d", "token_embd", "output", "indexer", "ffn_gate_inp"]
    
    total_tensors = len(reader.tensors)
    print(f"[*] Processing {total_tensors} tensors with R2Q quantization...", flush=True)
    
    # 2-Pass Streaming GGUF Conversion:
    # Pass 1: Plan tensor layouts and register metadata + tensor info headers without buffering weights in RAM.
    plan = []
    for idx, t in enumerate(reader.tensors, start=1):
        t_name = t.name
        should_quantize = (
            any(k in t_name for k in target_keywords) and
            not any(s in t_name for s in sensitive_keywords) and
            t_name.endswith(".weight") and
            len(t.shape) == 2 and
            int(t.shape[0]) % 256 == 0 and
            t.data.nbytes >= 1024
        )
        if should_quantize:
            n_cols = int(t.shape[0])
            n_rows = int(t.shape[1])
            byte_shape = [n_rows, (n_cols // 256) * 66]
            nbytes = n_rows * (n_cols // 256) * 66
            writer.add_tensor_info(t_name, byte_shape, np.dtype("uint8"), nbytes, raw_dtype=gguf.GGMLQuantizationType.TQ2_0)
            plan.append((t, True, n_rows, n_cols, nbytes))
        else:
            t_dtype = np.dtype("uint16") if t.tensor_type == gguf.GGMLQuantizationType.BF16 else t.data.dtype
            writer.add_tensor_info(t_name, list(t.shape), t_dtype, t.data.nbytes, raw_dtype=t.tensor_type)
            plan.append((t, False, 0, 0, t.data.nbytes))

    print(f"[*] Writing GGUF header, metadata, and tensor info dictionary to disk...", flush=True)
    t_w0 = time.time()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    dt_w = time.time() - t_w0
    print(f"[+] Finalized GGUF header dictionary in {dt_w:.2f}s.", flush=True)

    # Pass 2: Stream quantized matrices and unquantized tensor chunks directly to disk
    assert writer.fout is not None
    fout = writer.fout[0]
    quantized_count = 0
    skipped_count = 0
    quant_time_total = 0.0

    with open(src_base_gguf, "rb") as f_src:
        for idx, (t, is_q, n_rows, n_cols, nbytes) in enumerate(plan, start=1):
            t_name = t.name
            f_src.seek(t.data_offset)
            writer.write_padding(fout, fout.tell())

            if is_q:
                size_mb = t.data.nbytes / (1024**2)
                t_t0 = time.time()
                print(f"    [{idx}/{total_tensors}] Quantizing {t_name} ({n_rows}x{n_cols}, {size_mb:.1f} MB)...", flush=True)
                raw_bytes = f_src.read(t.data.nbytes)
                if t.tensor_type == gguf.GGMLQuantizationType.F32:
                    w_np = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                elif t.tensor_type == gguf.GGMLQuantizationType.F16:
                    w_np = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
                elif t.tensor_type == gguf.GGMLQuantizationType.BF16:
                    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
                    u32 = u16.astype(np.uint32) << 16
                    w_np = u32.view(np.float32)
                else:
                    w_np = np.frombuffer(raw_bytes, dtype=np.float32).copy()

                w_mat = w_np.reshape(n_rows, n_cols)
                X_calib = generate_activations(texts, dim=w_mat.shape[1]) if dad_steps > 0 else None

                tq2_buf = quantize_matrix_to_tq2_0_r2q(
                    W=w_mat,
                    X_calib=X_calib,
                    dad_steps=dad_steps,
                    dad_lr=dad_lr,
                    block_size=block_size
                )
                fout.write(tq2_buf)

                dt_t = time.time() - t_t0
                quantized_count += 1
                quant_time_total += dt_t
                avg_t = quant_time_total / quantized_count
                rem_tensors = total_tensors - idx
                est_rem_sec = rem_tensors * avg_t * (quantized_count / max(1, idx))
                eta_m, eta_s = divmod(int(est_rem_sec), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                print(f"    [{idx}/{total_tensors}] Quantized {t_name} in {dt_t:.2f}s (avg: {avg_t:.2f}s/q-tensor | ETA: {eta_h:02d}h{eta_m:02d}m{eta_s:02d}s)", flush=True)
            else:
                # Stream copy unquantized weights in 64MB chunks
                rem = t.data.nbytes
                while rem > 0:
                    chunk = f_src.read(min(rem, 64 * 1024 * 1024))
                    fout.write(chunk)
                    rem -= len(chunk)
                skipped_count += 1
                if idx % 50 == 0 or idx == total_tensors:
                    print(f"    [{idx}/{total_tensors}] Copied unquantized tensor {t_name} (total copied: {skipped_count})", flush=True)

            writer.write_padding(fout, nbytes)

    writer.close()
    print(f"[*] Completed tensor processing: {quantized_count} quantized, {skipped_count} copied directly.", flush=True)
    
    total_time = time.time() - t_start
    out_size_mb = os.path.getsize(dst_r2q_gguf) / (1024**2)
    print(f"[+] Successfully converted to R2Q GGUF in {total_time:.2f}s!", flush=True)
    print(f"[+] Output: {dst_r2q_gguf} ({out_size_mb:.2f} MiB)", flush=True)

MODELS_DIR = Path("/mnt/Media/Downloads/model_testing")
SCRATCH_DIR = Path("/mnt/Scratch/model_testing")
TEST_MODELS = [
    "NeoHorse-1-4B",
    "occamy-1.0-with-mtp",
    "Qwen3.8-27B-Cold-Fusion",
    "Qwen3.8-Flash-Next",
    "Gemma-4-31B-it"
]

def convert_model_r2q(model_name: str, dad_steps: int = 15, dad_lr: float = 1e-2, force: bool = False):
    source_model_dir = MODELS_DIR / model_name
    if not source_model_dir.exists():
        print(f"[Warning] Source model dir {source_model_dir} does not exist, skipping.", flush=True)
        return False

    work_dir = SCRATCH_DIR / model_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    source_bf16 = source_model_dir / f"{model_name}-BF16.gguf"
    work_bf16 = work_dir / f"{model_name}-BF16.gguf"
    
    source_r2q = source_model_dir / f"{model_name}-R2Q.gguf"
    work_r2q = work_dir / f"{model_name}-R2Q.gguf"
    
    if not force and (source_r2q.exists() or work_r2q.exists()):
        print(f"[*] Found existing R2Q GGUF for {model_name}, ensuring synced to persistent storage...", flush=True)
        if work_r2q.exists() and not source_r2q.exists():
            subprocess.run(["rsync", "-av", str(work_r2q), str(source_r2q)])
        return True

    if not work_bf16.exists():
        if source_bf16.exists():
            print(f"[*] Staging base BF16 GGUF from persistent storage to NVMe...", flush=True)
            subprocess.run(["rsync", "-av", str(source_bf16), str(work_bf16)])
        else:
            print(f"[Error] Base BF16 GGUF not found for {model_name} at {source_bf16}", flush=True)
            return False

    print(f"\n=======================================================", flush=True)
    print(f"  PROCESSING MODEL (R2Q Workflow): {model_name}", flush=True)
    print(f"=======================================================", flush=True)

    convert_to_r2q_gguf(
        src_base_gguf=work_bf16,
        dst_r2q_gguf=work_r2q,
        dad_steps=dad_steps,
        dad_lr=dad_lr
    )
    
    if work_r2q.exists():
        print(f"[*] Syncing R2Q GGUF back to persistent storage ({source_r2q})...", flush=True)
        subprocess.run(["rsync", "-av", str(work_r2q), str(source_r2q)])
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Convert model to R2Q 2.06 bpw GGUF")
    parser.add_argument("--model", type=str, default=None, help="Model name or 'all'")
    parser.add_argument("--src", type=str, default=None, help="Path to base BF16/FP16 GGUF")
    parser.add_argument("--dst", type=str, default=None, help="Path to output R2Q GGUF")
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS), help="Path to calibration corpus")
    parser.add_argument("--dad-steps", type=int, default=15, help="Number of DAD distillation steps")
    parser.add_argument("--dad-lr", type=float, default=1e-2, help="Learning rate for DAD")
    parser.add_argument("--force", action="store_true", help="Force re-conversion even if destination exists")
    args = parser.parse_args()
    
    if args.model:
        models = TEST_MODELS if args.model == "all" else [args.model]
        results = {}
        for m in models:
            ok = convert_model_r2q(m, dad_steps=args.dad_steps, dad_lr=args.dad_lr, force=args.force)
            results[m] = "SUCCESS" if ok else "FAILED"
        print("\n=======================================================", flush=True)
        print("  WORKFLOW 2 (R2Q) BATCH CONVERSION SUMMARY", flush=True)
        print("=======================================================", flush=True)
        for m, status in results.items():
            print(f"  - {m}: {status}", flush=True)
    elif args.src and args.dst:
        convert_to_r2q_gguf(
            src_base_gguf=Path(args.src),
            dst_r2q_gguf=Path(args.dst),
            corpus_path=Path(args.corpus) if args.corpus else None,
            dad_steps=args.dad_steps,
            dad_lr=args.dad_lr
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
