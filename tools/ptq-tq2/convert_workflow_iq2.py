#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
llama-apu: Workflow 1 (Mixed-Precision IQ2_S / IQ2_XXS with Native IMatrix)
Converts all 5 test models to IQ2_S GGUF format:
1. NeoHorse-1-4B
2. occamy-1.0-with-mtp
3. Qwen3.8-27B-Cold-Fusion
4. Qwen3.8-Flash-Next
5. Gemma-4-31B-it
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_BIN = REPO_ROOT / "build" / "bin"
LLAMA_IMATRIX = BUILD_BIN / "llama-imatrix"
LLAMA_QUANTIZE = BUILD_BIN / "llama-quantize"
LLAMA_CLI = BUILD_BIN / "llama-cli"
CONVERT_HF = REPO_ROOT / "convert_hf_to_gguf.py"
PYTHON_BIN = Path.home() / "databank" / "distill" / ".venv" / "bin" / "python3"
CALIB_SAMPLES = Path.home() / "databank" / "distill" / "calibration_samples.txt"
MODELS_DIR = Path("/mnt/Media/Downloads/model_testing")

TEST_MODELS = [
    "NeoHorse-1-4B",
    "occamy-1.0-with-mtp",
    "Qwen3.8-27B-Cold-Fusion",
    "Qwen3.8-Flash-Next",
    "Gemma-4-31B-it"
]

def run_cmd(cmd_list, description: str, env: Optional[dict] = None):
    print(f"\n[Command] {description}")
    print(" ".join(str(x) for x in cmd_list), flush=True)
    t0 = time.time()
    res = subprocess.run([str(x) for x in cmd_list], env=env)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[Error] Command failed with exit code {res.returncode} after {dt:.1f}s", flush=True)
        return False
    print(f"[Success] Completed {description} in {dt:.1f}s", flush=True)
    return True

SCRATCH_DIR = Path("/mnt/Scratch/model_testing")

def get_dir_size(path: Path) -> int:
    """Returns total size of files in bytes in path."""
    total = 0
    if path.is_file():
        return path.stat().st_size
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

def sync_to_scratch(source_dir: Path, target_dir: Path):
    """Syncs source model directory to NVMe scratch directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Staging {source_dir.name} to local NVMe scratch ({target_dir})...", flush=True)
    t0 = time.time()
    # Use rsync --update to copy new/updated files quickly
    res = subprocess.run(["rsync", "-av", "--update", f"{source_dir}/", f"{target_dir}/"])
    dt = time.time() - t0
    if res.returncode == 0:
        print(f"[+] Staged {source_dir.name} to {target_dir} in {dt:.1f}s.", flush=True)
        return True
    else:
        print(f"[Warning] rsync failed with exit code {res.returncode}, falling back...", flush=True)
        return False

def convert_model_wf1(model_name: str, quant_type: str = "IQ2_S"):
    source_model_dir = MODELS_DIR / model_name
    if not source_model_dir.exists():
        print(f"[Warning] Source model dir {source_model_dir} does not exist, skipping.", flush=True)
        return False

    size_bytes = get_dir_size(source_model_dir)
    size_gb = size_bytes / (1024**3)
    print(f"\n=======================================================", flush=True)
    print(f"  PROCESSING MODEL: {model_name} ({size_gb:.1f} GB) -> {quant_type}", flush=True)
    print(f"=======================================================", flush=True)

    # Check 1500GB threshold for NVMe staging (uses /mnt/Scratch NVMe drive)
    if size_gb < 1500.0:
        work_dir = SCRATCH_DIR / model_name
        sync_to_scratch(source_model_dir, work_dir)
    else:
        print(f"[*] Model size ({size_gb:.1f} GB) >= 1500 GB. Processing directly from {source_model_dir}...", flush=True)
        work_dir = source_model_dir

    bf16_gguf = work_dir / f"{model_name}-BF16.gguf"
    imatrix_dat = work_dir / "imatrix.dat"
    quant_gguf = work_dir / f"{model_name}-{quant_type}.gguf"

    # Also check if existing files are in source_model_dir
    source_bf16 = source_model_dir / f"{model_name}-BF16.gguf"
    source_imatrix = source_model_dir / "imatrix.dat"

    # 1. Base BF16 GGUF conversion if needed
    if not bf16_gguf.exists():
        if source_bf16.exists():
            print(f"[*] Copying existing BF16 GGUF from persistent storage to NVMe...", flush=True)
            subprocess.run(["rsync", "-av", str(source_bf16), str(bf16_gguf)])
        else:
            print(f"[*] Base BF16 GGUF not found. Converting from HF SafeTensors...", flush=True)
            ok = run_cmd([
                PYTHON_BIN, CONVERT_HF,
                work_dir,
                "--outtype", "bf16",
                "--outfile", bf16_gguf
            ], f"Convert {model_name} HF to BF16 GGUF")
            if not ok or not bf16_gguf.exists():
                print(f"[Error] Failed to create BF16 GGUF for {model_name}", flush=True)
                return False
            # Sync created BF16 back to persistent storage
            if work_dir != source_model_dir:
                subprocess.run(["rsync", "-av", str(bf16_gguf), str(source_bf16)])
    else:
        print(f"[*] Found existing BF16 GGUF in work dir: {bf16_gguf} ({os.path.getsize(bf16_gguf)/(1024**2):.1f} MiB)", flush=True)

    # 2. Compute imatrix if needed
    if not imatrix_dat.exists():
        if source_imatrix.exists():
            print(f"[*] Copying existing imatrix.dat from persistent storage to NVMe...", flush=True)
            subprocess.run(["rsync", "-av", str(source_imatrix), str(imatrix_dat)])
        else:
            print(f"[*] imatrix.dat not found. Computing importance matrix on calibration data...", flush=True)
            cpu_env = {**os.environ, "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
            ngl_val = "0" if "occamy" in model_name.lower() else "99"
            cmd_env = cpu_env if ngl_val == "0" else None
            ok = run_cmd([
                LLAMA_IMATRIX,
                "-m", bf16_gguf,
                "-f", CALIB_SAMPLES,
                "-o", imatrix_dat,
                "--chunks", "64",
                "-ngl", ngl_val
            ], f"Compute imatrix for {model_name} (ngl={ngl_val})", env=cmd_env)

            # Fallback to -ngl 0 if GPU offload failed
            if not ok and ngl_val != "0":
                print(f"[!] GPU imatrix failed. Retrying in CPU mode (-ngl 0)...", flush=True)
                ok = run_cmd([
                    LLAMA_IMATRIX,
                    "-m", bf16_gguf,
                    "-f", CALIB_SAMPLES,
                    "-o", imatrix_dat,
                    "--chunks", "64",
                    "-ngl", "0"
                ], f"Compute imatrix for {model_name} (CPU mode)", env=cpu_env)

            if not ok or not imatrix_dat.exists():
                print(f"[Error] Failed to compute imatrix for {model_name}", flush=True)
                return False
            # Sync imatrix.dat back to persistent storage
            if work_dir != source_model_dir:
                subprocess.run(["rsync", "-av", str(imatrix_dat), str(source_imatrix)])
    else:
        print(f"[*] Found existing imatrix.dat in work dir: {imatrix_dat}", flush=True)

    # 3. Quantize with mixed precision
    source_quant_gguf = source_model_dir / f"{model_name}-{quant_type}.gguf"
    if not quant_gguf.exists():
        if source_quant_gguf.exists():
            print(f"[*] Copying existing {quant_type} GGUF from persistent storage to NVMe...", flush=True)
            subprocess.run(["rsync", "-av", str(source_quant_gguf), str(quant_gguf)])
        else:
            print(f"[*] Quantizing to {quant_type} with importance matrix...", flush=True)
            ok = run_cmd([
                LLAMA_QUANTIZE,
                "--imatrix", imatrix_dat,
                bf16_gguf,
                quant_gguf,
                quant_type,
                "16"
            ], f"Quantize {model_name} to {quant_type}")
            if not ok or not quant_gguf.exists():
                print(f"[Error] Failed to quantize {model_name} to {quant_type}", flush=True)
                return False
    else:
        print(f"[*] Found existing {quant_type} GGUF in work dir: {quant_gguf}", flush=True)

    # Sync quantized output GGUF back to persistent storage
    source_quant_gguf = source_model_dir / f"{model_name}-{quant_type}.gguf"
    if work_dir != source_model_dir:
        print(f"[*] Syncing quantized model back to persistent storage ({source_quant_gguf})...", flush=True)
        subprocess.run(["rsync", "-av", str(quant_gguf), str(source_quant_gguf)])

    print(f"[+] Output: {quant_gguf} ({os.path.getsize(quant_gguf)/(1024**2):.1f} MiB)", flush=True)
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert all test models for Workflow 1")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    parser.add_argument("--type", type=str, default="IQ2_S", help="Quantization type (IQ2_S, IQ2_XXS, Q2_K)")
    args = parser.parse_args()
    
    models = TEST_MODELS if args.model == "all" else [args.model]
    
    results = {}
    for m in models:
        ok = convert_model_wf1(m, quant_type=args.type)
        results[m] = "SUCCESS" if ok else "FAILED"
        
    print("\n=======================================================")
    print("  WORKFLOW 1 BATCH CONVERSION SUMMARY")
    print("=======================================================")
    for m, status in results.items():
        print(f"  - {m}: {status}")

if __name__ == "__main__":
    main()
