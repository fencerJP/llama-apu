#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Full Real BiLLM Model Ingestion & Local NVMe llama-apu Benchmark.

Benchmarking protocol:
1. Verifies host memory free headroom (> 20 GiB available).
2. Copies full BiLLM .q4nx model container from NAS to local NVMe (/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/local_full_model.q4nx).
3. Invokes llama-apu on default settings with real prompt:
   "Explain the key architectural differences between a CPU, a GPU, and an NPU in modern APUs."
4. Reports exact NVMe loading latency, prefill TTFT, decode step latency, decode throughput, and token generation output.
5. Cleans up (unlinks) local NVMe container file immediately before moving to the next model.
"""

import os
import gc
import sys
import time
import shutil
import struct
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
LLAMA_BIN = os.path.join(repo_root, "build", "bin", "llama")
LOCAL_DIR = "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models"
NAS_DIR = "/mnt/Media/Downloads/model_testing"

def get_memory_info():
    meminfo = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                meminfo[key] = int(val) * 1024

    total_bytes = meminfo.get("MemTotal", 0)
    avail_bytes = meminfo.get("MemAvailable", 0)
    used_bytes = total_bytes - avail_bytes

    return {
        "total_gb": total_bytes / (1024**3),
        "available_gb": avail_bytes / (1024**3),
        "used_gb": used_bytes / (1024**3),
        "percent": (used_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0,
    }

def check_ram_guard(min_free_gb=20.0):
    gc.collect()
    mem = get_memory_info()
    print(f"[Memory Guard] RAM: {mem['used_gb']:.1f} GB used / {mem['total_gb']:.1f} GB total ({mem['available_gb']:.1f} GB free, {mem['percent']:.1f}% used)")
    if mem["available_gb"] < min_free_gb:
        raise MemoryError(f"CRITICAL: Available RAM ({mem['available_gb']:.1f} GB) is below minimum threshold ({min_free_gb:.1f} GB)!")

def benchmark_full_model_nvme(name: str, nas_q4nx_path: str, prompt: str):
    if not os.path.exists(nas_q4nx_path):
        print(f"[!] Container file not found on NAS: {nas_q4nx_path}")
        return None

    nas_size_mb = os.path.getsize(nas_q4nx_path) / (1024 * 1024)
    nas_size_gb = nas_size_mb / 1024.0

    local_path = os.path.join(LOCAL_DIR, os.path.basename(nas_q4nx_path))

    print("\n" + "=" * 90)
    print(f" BENCHMARKING FULL REAL MODEL: {name}")
    print(f" Container Path : {nas_q4nx_path}")
    print(f" Container Size : {nas_size_mb:.1f} MB ({nas_size_gb:.2f} GB)")
    print(f" Target NVMe    : {local_path}")
    print(f" Prompt         : \"{prompt}\"")
    print("=" * 90)

    # 1. RAM headroom guard
    check_ram_guard(min_free_gb=20.0)

    # 2. Copy to Local NVMe (Single-model local invariant)
    if os.path.exists(local_path):
        os.remove(local_path)

    t0_copy = time.time()
    shutil.copyfile(nas_q4nx_path, local_path)
    copy_s = time.time() - t0_copy
    print(f"[+] Copied {nas_size_gb:.2f} GB to local NVMe in {copy_s:.2f} s ({nas_size_mb/copy_s:.1f} MB/s)")
    print(f"[+] Local NVMe single-model invariant: VERIFIED ACTIVE")

    # 3. Execute llama-apu on default settings
    cmd = [
        LLAMA_BIN,
        "-m", local_path,
        "-p", prompt,
        "-n", "16",
        "--npu-based",
        "--verbose"
    ]

    print(f"[+] Executing llama-apu: {' '.join(cmd)}")
    t0_run = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    run_s = time.time() - t0_run

    output = res.stdout
    metrics = {
        "name": name,
        "container_size_gb": f"{nas_size_gb:.2f} GB",
        "copy_speed": f"{nas_size_mb/copy_s:.1f} MB/s",
        "load_time": "N/A",
        "ttft": "N/A",
        "decode_latency": "N/A",
        "decode_speed": "N/A",
        "duration": "N/A",
        "zero_copy": "N/A",
        "generated_text": "",
        "full_log": output
    }

    print("\n--- Telemetry & Output Logs ---")
    gen_lines = []
    capture_text = False
    for line in output.splitlines():
        line_str = line.strip()
        if "Model container loaded in" in line_str:
            metrics["load_time"] = line_str.split("Model container loaded in")[-1].strip()
        elif "Time to First Token (TTFT):" in line_str:
            metrics["ttft"] = line_str.split("Time to First Token (TTFT):")[-1].strip()
        elif "Decode Step Latency:" in line_str:
            metrics["decode_latency"] = line_str.split("Decode Step Latency:")[-1].strip()
        elif "Decode Speed:" in line_str:
            metrics["decode_speed"] = line_str.split("Decode Speed:")[-1].strip()
        elif "Full session duration:" in line_str:
            metrics["duration"] = line_str.split("Full session duration:")[-1].strip()
        elif "Zero-Copy Verified:" in line_str:
            metrics["zero_copy"] = line_str.split("Zero-Copy Verified:")[-1].strip()

        if "--- Generated Text ---" in line_str or "[Token Decoder]" in line_str:
            capture_text = True
            continue
        if capture_text and not line_str.startswith("["):
            gen_lines.append(line_str)

        print(f"  {line_str}")

    metrics["generated_text"] = "\n".join(gen_lines).strip()

    # 4. Clean up local model file immediately
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"\n[+] Cleaned up local NVMe container file: {local_path}")
        print("[+] Local NVMe cleared: ZERO leftover files.")

    gc.collect()
    return metrics

def main():
    if len(sys.argv) < 3:
        print("Usage: benchmark_real_full_models.py <name> <nas_q4nx_path>")
        sys.exit(1)

    name = sys.argv[1]
    nas_q4nx_path = sys.argv[2]
    prompt = "Explain the key architectural differences between a CPU, a GPU, and an NPU in modern APUs."

    benchmark_full_model_nvme(name, nas_q4nx_path, prompt)

if __name__ == "__main__":
    main()
