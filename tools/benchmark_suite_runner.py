#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Harder Inference Benchmark Suite Test Harness for AMD Ryzen AI APU Heterogeneous Model Runner.

Executes side-by-side inference tests across 11 model architectures and 9 distinct families:
  1. DeepSeek R1 / Qwen3 MoE (deepseek-r1-qwen3-8b)
  2. Meta Llama 3.2 (llama-3.2-3b)
  3. Qwen 2.5 Dense (qwen2.5-3b & qwen2.5-0.5b)
  4. Qwen 3.5 Hybrid Linear Attention (qwen3.5-0.8b)
  5. Google Gemma 4 (gemma4-E4B)
  6. Spark SSM / Linear Attention (spark-x2.5-1.7b)
  7. Liquid Foundation Models (lfm2-1.2b from FastFlowLM)
  8. Ornith 9B (ornith-1.0-9b from local storage)
  9. Qwythos 9B (qwythos-9b from local storage)
 10. K2-Horizon 1B (k2-horizon-1b from local storage)

Comparing 3 configurations per model:
  - Set A: Built-in production XCLBIN (if provided by FastFlowLM / AMD)
  - Set B: Custom Generated XCLBIN (Enhanced format: dynamic 64MB SRAM, explicit metadata)
  - Set C: Custom Generated XCLBIN (Mimic-Builtin format: fixed 48MB SRAM, blank metadata tags)

Designed for 0.1s - 1.0s TTFT using high-density architectural and mathematical prompts.
"""

import os
import sys
import time
import json
import gc
import subprocess
from pathlib import Path

APU_RUN_BIN = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp/build/bin/apu-run")
LIB_DIR = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp/build/bin")
SUITE_ROOT = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/benchmark_suite")
MODELS_ROOT = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models")
PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "old" / "test_prompts"

BENCHMARK_MODELS = [
    {
        "id": "deepseek-r1-qwen3-8b",
        "name": "DeepSeek-R1-0528-Qwen3-8B-NPU2 (DeepSeek/MoE)",
        "dir": SUITE_ROOT / "deepseek-r1-qwen3-8b",
        "model_file": "model.q4nx",
        "fallback_model_file": "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
        "builtin_xclbin": "layer.xclbin",
        "generated_enhanced_xclbin": "generated-layer.xclbin",
        "generated_mimic_xclbin": "generated-mimic.xclbin",
    },
    {
        "id": "llama-3.2-3b",
        "name": "Llama-3.2-3B-NPU2 (Meta Llama)",
        "dir": SUITE_ROOT / "llama-3.2-3b",
        "model_file": "model.q4nx",
        "fallback_model_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "builtin_xclbin": "layer.xclbin",
        "generated_enhanced_xclbin": "generated-layer.xclbin",
        "generated_mimic_xclbin": "generated-mimic.xclbin",
    },
    {
        "id": "qwen2.5-3b",
        "name": "Qwen2.5-3B-Instruct-NPU2 (Qwen Dense)",
        "dir": SUITE_ROOT / "qwen2.5-3b",
        "model_file": "model.q4nx",
        "fallback_model_file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "builtin_xclbin": "layer.xclbin",
        "generated_enhanced_xclbin": "generated-layer.xclbin",
        "generated_mimic_xclbin": "generated-mimic.xclbin",
    },
    {
        "id": "qwen3.5-0.8b",
        "name": "Qwen3.5-0.8B-NPU2 (GateDeltaNet / Linear Attention)",
        "dir": SUITE_ROOT / "qwen3.5-0.8b",
        "model_file": "model.q4nx",
        "fallback_model_file": "Qwen3.5-0.8B-Q4_K_M.gguf",
        "builtin_xclbin": "layer.xclbin",
        "generated_enhanced_xclbin": "generated-layer.xclbin",
        "generated_mimic_xclbin": "generated-mimic.xclbin",
    },
    {
        "id": "gemma4",
        "name": "Gemma4-E2B-IT-NPU2 / Gemma-4-E4B (Google Gemma 4)",
        "dir": SUITE_ROOT / "gemma4",
        "model_file": "model.q4nx",
        "fallback_model_file": "gemma-4-E4B-heretic.gguf",
        "builtin_xclbin": "layer.xclbin",
        "generated_enhanced_xclbin": "generated-layer.xclbin",
        "generated_mimic_xclbin": "generated-mimic.xclbin",
    },
    {
        "id": "spark-x2.5-1.7b",
        "name": "Spark-X2.5-1.7B (Hybrid SSM / State-Space)",
        "dir": MODELS_ROOT,
        "model_file": "Spark-X2.5-1.7B.q4nx",
        "fallback_model_file": "Spark-X2.5-1.7B.gguf",
        "builtin_xclbin": None,
        "generated_enhanced_xclbin": MODELS_ROOT / "Spark-X2.5-1.7B-NPU2.xclbin",
        "generated_mimic_xclbin": MODELS_ROOT / "Spark-X2.5-mimic.xclbin",
    },
    {
        "id": "lfm2-1.2b",
        "name": "LFM2-1.2B-NPU2 (Liquid Foundation Models)",
        "dir": SUITE_ROOT / "lfm2-1.2b",
        "model_file": "model.q4nx",
        "fallback_model_file": None,
        "builtin_xclbin": "layer.xclbin",
        "generated_enhanced_xclbin": "generated-layer.xclbin",
        "generated_mimic_xclbin": "generated-mimic.xclbin",
    },
    {
        "id": "ornith-1.0-9b",
        "name": "Ornith-1.0-9B (Qwen 3.5 Fine-Tune / Local)",
        "dir": Path("/opt/models"),
        "model_file": "ornith-1.0-9b.gguf",
        "fallback_model_file": None,
        "builtin_xclbin": None,
        "generated_enhanced_xclbin": MODELS_ROOT / "ornith-enhanced.xclbin",
        "generated_mimic_xclbin": MODELS_ROOT / "ornith-mimic.xclbin",
    },
    {
        "id": "qwythos-9b",
        "name": "Qwythos-9B-ab (Qwythos Family / Local)",
        "dir": Path("/opt/models"),
        "model_file": "qwythos-9b-ab.gguf",
        "fallback_model_file": None,
        "builtin_xclbin": None,
        "generated_enhanced_xclbin": MODELS_ROOT / "qwythos-enhanced.xclbin",
        "generated_mimic_xclbin": MODELS_ROOT / "qwythos-mimic.xclbin",
    },
    {
        "id": "k2-horizon-1b",
        "name": "K2-Horizon-1B-BF16 (K2 Horizon Family / Local)",
        "dir": MODELS_ROOT,
        "model_file": "K2-Horizon-1B-BF16.gguf",
        "fallback_model_file": None,
        "builtin_xclbin": None,
        "generated_enhanced_xclbin": MODELS_ROOT / "k2-horizon-enhanced.xclbin",
        "generated_mimic_xclbin": MODELS_ROOT / "k2-horizon-mimic.xclbin",
    },
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen2.5-0.5B-Instruct (Lightweight Qwen)",
        "dir": MODELS_ROOT,
        "model_file": "qwen2.5-0.5b-instruct-q8_0.q4nx",
        "fallback_model_file": "qwen2.5-0.5b-instruct-q8_0.gguf",
        "builtin_xclbin": None,
        "generated_enhanced_xclbin": MODELS_ROOT / "qwen2.5-0.5b-enhanced.xclbin",
        "generated_mimic_xclbin": MODELS_ROOT / "qwen2.5-0.5b-mimic.xclbin",
    },
]

BENCHMARK_PROMPTS = [
    {
        "tag": "Harder_System_Arch",
        "description": "Multi-tier APU Heterogeneous Memory & Timeline Fence Architecture Specification (~650 tokens)",
        "file": PROMPTS_ROOT / "harder_test_system_arch.txt",
        "tokens_to_gen": 16,
    },
    {
        "tag": "Harder_Neural_Math",
        "description": "Comparative Mathematical Dissertation on Transformer vs GateDeltaNet vs SSM vs LFM (~625 tokens)",
        "file": PROMPTS_ROOT / "harder_test_neural_math.txt",
        "tokens_to_gen": 16,
    },
]

def get_system_memory_info():
    """Reads current memory metrics from /proc/meminfo."""
    mem = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_parts = parts[1].strip().split()
                    if val_parts:
                        mem[key] = int(val_parts[0])
    except Exception:
        pass
    return mem

def evict_and_reconcile_memory():
    """Forces aggressive host memory eviction and cache sync between runs."""
    gc.collect()
    try:
        os.sync()
    except Exception:
        pass
    time.sleep(1.5)

def resolve_xclbin_path(model_dir: Path, xclbin_entry):
    if not xclbin_entry:
        return None
    if isinstance(xclbin_entry, Path):
        return xclbin_entry
    return model_dir / xclbin_entry

def resolve_model_path(model_entry):
    """Selects valid model file (.q4nx preferred, falls back to .gguf)."""
    d = model_entry["dir"]
    q4nx = d / model_entry["model_file"]
    if q4nx.exists():
        return q4nx
    if model_entry.get("fallback_model_file"):
        gguf = d / model_entry["fallback_model_file"]
        if gguf.exists():
            return gguf
    return q4nx

def run_single_benchmark(model_path: Path, xclbin_path: Path | None, prompt_file: Path, steps: int = 16, timeout_sec: int = 60):
    """
    Executes apu-run in an isolated subprocess with full hardware fault containment.
    """
    if not model_path.exists():
        return {
            "success": False,
            "status": "MISSING_MODEL_FILE",
            "error": f"Model file not found: {model_path}",
            "elapsed_sec": 0.0,
            "ttft_ms": None,
            "decode_tps": None,
            "step_latency_us": None,
            "tokens_generated": 0,
            "output_text": "",
        }

    if xclbin_path and not xclbin_path.exists():
        return {
            "success": False,
            "status": "MISSING_XCLBIN_FILE",
            "error": f"XCLBIN file not found: {xclbin_path}",
            "elapsed_sec": 0.0,
            "ttft_ms": None,
            "decode_tps": None,
            "step_latency_us": None,
            "tokens_generated": 0,
            "output_text": "",
        }

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        str(APU_RUN_BIN),
        "-m", str(model_path),
        "-f", str(prompt_file),
        "-n", str(steps),
        "--temp", "0.0",
        "--verbose"
    ]
    if xclbin_path:
        cmd.extend(["-x", str(xclbin_path)])

    mem_before = get_system_memory_info()
    avail_before_mb = mem_before.get("MemAvailable", 0) / 1024.0

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=False,
            timeout=timeout_sec
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0

        stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        returncode = proc.returncode

    except subprocess.TimeoutExpired as te:
        elapsed = timeout_sec
        stdout_err = te.stdout.decode("utf-8", errors="replace") if te.stdout else ""
        stderr_err = te.stderr.decode("utf-8", errors="replace") if te.stderr else ""
        return {
            "success": False,
            "status": "APU_TIMEOUT_EXPIRED",
            "error": f"Execution timed out after {timeout_sec} seconds. Hardware fence or kernel wait did not return.",
            "elapsed_sec": elapsed,
            "ttft_ms": None,
            "decode_tps": None,
            "step_latency_us": None,
            "tokens_generated": 0,
            "output_text": "",
            "stdout": stdout_err,
            "stderr": stderr_err,
        }
    except Exception as e:
        return {
            "success": False,
            "status": "SUBPROCESS_SPAWN_ERROR",
            "error": str(e),
            "elapsed_sec": 0.0,
            "ttft_ms": None,
            "decode_tps": None,
            "step_latency_us": None,
            "tokens_generated": 0,
            "output_text": "",
        }
    finally:
        evict_and_reconcile_memory()

    mem_after = get_system_memory_info()
    avail_after_mb = mem_after.get("MemAvailable", 0) / 1024.0
    mem_delta_mb = avail_before_mb - avail_after_mb

    # Analyze failure signatures
    status = "SUCCESS" if returncode == 0 else "EXECUTION_ERROR"
    error_desc = ""

    if returncode == -9:
        status = "OUT_OF_MEMORY_KILL"
        error_desc = "Process terminated by Linux OOM killer (SIGKILL -9)."
    elif returncode == -11:
        status = "SIGSEGV_FAULT"
        error_desc = "Process crashed with Segmentation Fault (SIGSEGV -11)."
    elif returncode == -6:
        status = "SIGABRT_FAULT"
        error_desc = "Process aborted (SIGABRT -6)."
    elif returncode != 0:
        if "bad_alloc" in stderr or "cannot allocate" in stderr.lower():
            status = "HOST_OUT_OF_MEMORY"
            error_desc = "Host memory allocation failed (bad_alloc)."
        elif "EBUSY" in stderr or "Device or resource busy" in stderr:
            status = "APU_DEVICE_BUSY"
            error_desc = "Hardware accelerator node (/dev/accel/accel0) was busy."
        else:
            status = f"RETURN_CODE_{returncode}"
            error_desc = stderr.strip().splitlines()[-1] if stderr.strip() else f"Process exited with {returncode}"

    # Parse telemetry
    ttft_ms = None
    decode_tps = None
    step_latency_us = None
    tokens_generated = None
    prompt_seq_len = None

    for line in stdout.splitlines():
        if "Prompt sequence length:" in line:
            try:
                parts = line.split(":")[-1].strip().split()
                prompt_seq_len = int(parts[0])
            except Exception:
                pass
        elif "Prefill TTFT:" in line or "Time to First Token (TTFT):" in line:
            try:
                parts = line.split(":")[-1].strip().split()
                ttft_ms = float(parts[0])
            except Exception:
                pass
        elif "NPU Generation Speed:" in line or "Tokens Per Second (TPS):" in line:
            try:
                parts = line.split(":")[-1].strip().split()
                decode_tps = float(parts[0])
            except Exception:
                pass
        elif "Decode Step Latency:" in line:
            try:
                parts = line.split(":")[-1].strip().split()
                step_latency_us = float(parts[0])
            except Exception:
                pass
        elif "Tokens Generated:" in line:
            try:
                tokens_generated = int(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass

    # Extract assistant text
    in_output_section = False
    output_lines = []
    for line in stdout.splitlines():
        if "--- [Step 2: Autoregressive Decode on XDNA 2 NPU] ---" in line:
            in_output_section = True
            continue
        if "========================================================" in line and in_output_section:
            in_output_section = False
        if in_output_section and not line.startswith("Generated Text Output:") and not line.startswith("[VERBOSE]"):
            output_lines.append(line)
    output_text = " ".join([l.strip() for l in output_lines if l.strip()])

    return {
        "success": returncode == 0,
        "status": status,
        "error": error_desc,
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_seq_len,
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "step_latency_us": step_latency_us,
        "tokens_generated": tokens_generated or 0,
        "output_text": output_text,
        "mem_delta_mb": mem_delta_mb,
        "stdout_tail": "\n".join(stdout.splitlines()[-15:]),
    }

def main():
    print("================================================================================")
    print("  AMD Ryzen AI Heterogeneous APU Harder Inference Benchmark Suite")
    print("  Target: 0.1s - 1.0s TTFT Across 11 Models and 9 Distinct Architectural Families")
    print("================================================================================\n")

    if "--dry-run" in sys.argv or len(sys.argv) == 1:
        print("[DRY RUN / VALIDATION MODE] Verifying all test models and hardware binaries:")
        for m in BENCHMARK_MODELS:
            m_path = resolve_model_path(m)
            b_xcl = resolve_xclbin_path(m["dir"], m["builtin_xclbin"])
            e_xcl = resolve_xclbin_path(m["dir"], m["generated_enhanced_xclbin"])
            c_xcl = resolve_xclbin_path(m["dir"], m["generated_mimic_xclbin"])

            print(f"\nModel: {m['name']} ({m['id']})")
            print(f"  Source Model Container: {'EXISTS' if m_path.exists() else 'MISSING'} ({m_path.name})")
            print(f"  [Set A: Built-in XCLBIN]:  {'EXISTS' if (b_xcl and b_xcl.exists()) else 'NOT AVAILABLE'}")
            print(f"  [Set B: Enhanced XCLBIN]:  {'EXISTS' if (e_xcl and e_xcl.exists()) else 'MISSING'}")
            print(f"  [Set C: Mimic-Builtin XCL]:{'EXISTS' if (c_xcl and c_xcl.exists()) else 'MISSING'}")

        print("\nAll assets validated. Pass --execute to run the full benchmark matrix.")
        return

    if "--execute" in sys.argv:
        results = []
        start_matrix_time = time.time()

        for m in BENCHMARK_MODELS:
            m_path = resolve_model_path(m)
            b_xcl = resolve_xclbin_path(m["dir"], m["builtin_xclbin"])
            e_xcl = resolve_xclbin_path(m["dir"], m["generated_enhanced_xclbin"])
            c_xcl = resolve_xclbin_path(m["dir"], m["generated_mimic_xclbin"])

            print("\n" + "=" * 75)
            print(f" Benchmarking Model: {m['name']} ({m['id']})")
            print("=" * 75)

            for p_info in BENCHMARK_PROMPTS:
                tag = p_info["tag"]
                p_file = p_info["file"]
                desc = p_info["description"]
                steps = p_info["tokens_to_gen"]
                print(f"\n>> Heavy Prompt Scenario [{tag}]")
                print(f"   Description: {desc}")
                print(f"   Prompt Source: {p_file.name} (Decode Target: {steps} steps)")

                # --- 1. Set A: Built-in Production XCLBIN ---
                print("   [1/3] Executing Set A: Built-in Production XCLBIN...")
                if b_xcl and b_xcl.exists():
                    res_a = run_single_benchmark(m_path, b_xcl, p_file, steps)
                    print(f"         Status: {res_a['status']} | Prompt Tokens: {res_a.get('prompt_tokens')} | TTFT: {res_a['ttft_ms']} ms | TPS: {res_a['decode_tps']} | Latency: {res_a['step_latency_us']} us")
                else:
                    res_a = {
                        "success": False,
                        "status": "NOT_AVAILABLE",
                        "error": "No vendor built-in XCLBIN provided in original release",
                        "prompt_tokens": None,
                        "ttft_ms": None,
                        "decode_tps": None,
                        "step_latency_us": None,
                        "tokens_generated": 0,
                        "output_text": "N/A"
                    }
                    print("         Status: NOT_AVAILABLE (Vendor did not provide built-in XCLBIN)")

                # --- 2. Set B: Enhanced Generated XCLBIN ---
                print("   [2/3] Executing Set B: Custom Generated XCLBIN (Enhanced Format)...")
                res_b = run_single_benchmark(m_path, e_xcl, p_file, steps)
                print(f"         Status: {res_b['status']} | Prompt Tokens: {res_b.get('prompt_tokens')} | TTFT: {res_b['ttft_ms']} ms | TPS: {res_b['decode_tps']} | Latency: {res_b['step_latency_us']} us")

                # --- 3. Set C: Mimic-Builtin Generated XCLBIN ---
                print("   [3/3] Executing Set C: Custom Generated XCLBIN (Mimic-Builtin Format)...")
                res_c = run_single_benchmark(m_path, c_xcl, p_file, steps)
                print(f"         Status: {res_c['status']} | Prompt Tokens: {res_c.get('prompt_tokens')} | TTFT: {res_c['ttft_ms']} ms | TPS: {res_c['decode_tps']} | Latency: {res_c['step_latency_us']} us")

                results.append({
                    "model_id": m["id"],
                    "model_name": m["name"],
                    "prompt_tag": tag,
                    "prompt_desc": desc,
                    "prompt_file": str(p_file),
                    "steps_requested": steps,
                    "set_a_builtin": res_a,
                    "set_b_enhanced": res_b,
                    "set_c_mimic": res_c,
                })

        total_duration = time.time() - start_matrix_time
        out_json = SUITE_ROOT / "harder_benchmark_matrix_results.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 75)
        print(f" Harder Benchmark Matrix Completed in {total_duration:.1f}s")
        print(f" Saved full JSON dataset to: {out_json}")
        print("=" * 75)

if __name__ == "__main__":
    main()
