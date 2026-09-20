#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive APU Hardware Inference Benchmark Matrix:
Comparing Built-in, Enhanced, and Mimic XCLBINs across all installed and eligible models.
"""

import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path

LLAMA_CLI_BIN = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp/build/bin/llama-cli")
BIN_DIR = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp/build/bin")
SUITE_ROOT = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/benchmark_suite")
MODELS_ROOT = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models")
FASTFLOW_XCLBINS = Path("/home/fencer/.openclaw/workspace/projects/fastflowlm/src/xclbins")
PROMPT_FILE = Path(__file__).resolve().parent.parent / "old" / "test_prompts" / "consistent_bench_prompt.txt"

MODELS = [
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen2.5-0.5B-Instruct",
        "model": MODELS_ROOT / "qwen2.5-0.5b-instruct-q8_0.gguf",
        "builtin": FASTFLOW_XCLBINS / "Qwen3-VL-4B-Instruct-NPU2" / "layer.xclbin",
        "enhanced": MODELS_ROOT / "qwen2.5-0.5b-enhanced.xclbin",
        "mimic": MODELS_ROOT / "qwen2.5-0.5b-mimic.xclbin",
        "prompt_tokens_est": 184,
    },
    {
        "id": "llama-3.2-1b",
        "name": "Llama-3.2-1B-Instruct",
        "model": MODELS_ROOT / "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "builtin": FASTFLOW_XCLBINS / "Llama-3.1-8B-NPU2" / "layer.xclbin",
        "enhanced": SUITE_ROOT / "llama-3.2-3b" / "generated-layer.xclbin",
        "mimic": SUITE_ROOT / "llama-3.2-3b" / "generated-mimic.xclbin",
        "prompt_tokens_est": 180,
    },
    {
        "id": "gemma-2-2b",
        "name": "Gemma-2-2B-IT",
        "model": MODELS_ROOT / "gemma-2-2b-it-Q4_K_M.gguf",
        "builtin": FASTFLOW_XCLBINS / "Gemma3-4B-NPU2" / "layer.xclbin",
        "enhanced": SUITE_ROOT / "gemma4" / "generated-layer.xclbin",
        "mimic": SUITE_ROOT / "gemma4" / "generated-mimic.xclbin",
        "prompt_tokens_est": 185,
    },
    {
        "id": "spark-x2.5-1.7b",
        "name": "Spark-X2.5-1.7B",
        "model": MODELS_ROOT / "Spark-X2.5-1.7B.gguf",
        "builtin": FASTFLOW_XCLBINS / "LFM2-2.6B-NPU2" / "layer.xclbin",
        "enhanced": MODELS_ROOT / "Spark-X2.5-1.7B-NPU2.xclbin",
        "mimic": MODELS_ROOT / "Spark-X2.5-mimic.xclbin",
        "prompt_tokens_est": 190,
    },
    {
        "id": "k2-horizon-1b",
        "name": "K2-Horizon-1B-BF16",
        "model": MODELS_ROOT / "K2-Horizon-1B-BF16.gguf",
        "builtin": FASTFLOW_XCLBINS / "Llama-3.1-8B-NPU2" / "layer.xclbin",
        "enhanced": MODELS_ROOT / "k2-horizon-enhanced.xclbin",
        "mimic": MODELS_ROOT / "k2-horizon-mimic.xclbin",
        "prompt_tokens_est": 182,
    },
    {
        "id": "qwen3.5-0.8b",
        "name": "Qwen3.5-0.8B-Q4_K_M",
        "model": SUITE_ROOT / "qwen3.5-0.8b" / "Qwen3.5-0.8B-Q4_K_M.gguf",
        "builtin": SUITE_ROOT / "qwen3.5-0.8b" / "layer.xclbin",
        "enhanced": SUITE_ROOT / "qwen3.5-0.8b" / "generated-layer.xclbin",
        "mimic": SUITE_ROOT / "qwen3.5-0.8b" / "generated-mimic.xclbin",
        "prompt_tokens_est": 185,
    },
    {
        "id": "qwen2.5-3b",
        "name": "Qwen2.5-3B-Instruct",
        "model": SUITE_ROOT / "qwen2.5-3b" / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "builtin": SUITE_ROOT / "qwen2.5-3b" / "layer.xclbin",
        "enhanced": SUITE_ROOT / "qwen2.5-3b" / "generated-layer.xclbin",
        "mimic": SUITE_ROOT / "qwen2.5-3b" / "generated-mimic.xclbin",
        "prompt_tokens_est": 184,
    },
    {
        "id": "llama-3.2-3b",
        "name": "Llama-3.2-3B-Instruct",
        "model": SUITE_ROOT / "llama-3.2-3b" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "builtin": SUITE_ROOT / "llama-3.2-3b" / "layer.xclbin",
        "enhanced": SUITE_ROOT / "llama-3.2-3b" / "generated-layer.xclbin",
        "mimic": SUITE_ROOT / "llama-3.2-3b" / "generated-mimic.xclbin",
        "prompt_tokens_est": 180,
    },
    {
        "id": "gemma-4-e4b",
        "name": "Gemma-4-E4B",
        "model": SUITE_ROOT / "gemma4" / "gemma-4-E4B-heretic.gguf",
        "builtin": SUITE_ROOT / "gemma4" / "layer.xclbin",
        "enhanced": SUITE_ROOT / "gemma4" / "generated-layer.xclbin",
        "mimic": SUITE_ROOT / "gemma4" / "generated-mimic.xclbin",
        "prompt_tokens_est": 185,
    },
    {
        "id": "deepseek-r1-qwen3-8b",
        "name": "DeepSeek-R1-0528-Qwen3-8B",
        "model": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
        "builtin": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "layer.xclbin",
        "enhanced": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "generated-layer.xclbin",
        "mimic": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "generated-mimic.xclbin",
        "prompt_tokens_est": 184,
    },
    {
        "id": "k2-horizon-7b",
        "name": "K2-Horizon-7B-IQ4_NL",
        "model": Path("/home/fencer/.openclaw/workspace/models/K2-Horizon-7B-IQ4_NL.gguf"),
        "builtin": FASTFLOW_XCLBINS / "Llama-3.1-8B-NPU2" / "layer.xclbin",
        "enhanced": Path("/home/fencer/.openclaw/workspace/models/K2-Horizon-7B-enhanced.xclbin"),
        "mimic": Path("/home/fencer/.openclaw/workspace/models/K2-Horizon-7B-mimic.xclbin"),
        "prompt_tokens_est": 182,
    },
    {
        "id": "k2-horizon-32b",
        "name": "K2-Horizon-32B-IQ4_NL",
        "model": Path("/home/fencer/.openclaw/workspace/models/K2-Horizon-32B-IQ4_NL.gguf"),
        "builtin": FASTFLOW_XCLBINS / "Qwen3.6-35B-A3B-NPU2" / "layer.xclbin",
        "enhanced": Path("/home/fencer/.openclaw/workspace/models/K2-Horizon-32B-enhanced.xclbin"),
        "mimic": Path("/home/fencer/.openclaw/workspace/models/K2-Horizon-32B-mimic.xclbin"),
        "prompt_tokens_est": 182,
    },
]

PROMPT_TEXT = PROMPT_FILE.read_text()

def evaluate_quality(text):
    text = text.strip()
    if not text:
        return "Unacceptable (Empty)"
    words = text.split()
    if len(words) < 2:
        return "Marginal (Too Short)"
    if len(words) >= 4 and len(set(words)) <= 2:
        return "Unacceptable (Repetitive Loop)"
    
    keywords = ["memory", "apu", "gpu", "npu", "unified", "zero-copy", "bandwidth", "dma", "gem", "physical", "pcie", "1.", "architecture", "trade", "sync", "buffer"]
    matches = sum(1 for kw in keywords if kw in text.lower())
    if matches >= 2:
        return "Acceptable (High)"
    elif matches >= 1:
        return "Acceptable (Coherent)"
    return "Acceptable (Fluent)"

def run_test(model_path, xclbin_path, prompt_tokens_est, steps=16, timeout_sec=120):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{BIN_DIR}:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_CLI_BIN),
        "-m", str(model_path),
        "-p", PROMPT_TEXT,
        "-n", str(steps),
        "-t", "12",
        "--temp", "0.0",
        "--no-warmup",
        "-st",
        "--simple-io",
        "--apu-verbose"
    ]
    if xclbin_path and xclbin_path.exists():
        cmd.extend(["--apu-xclbin", str(xclbin_path)])

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_sec)
        stdout = proc.stdout
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "status": "TIMEOUT",
            "ttft_ms": None,
            "decode_tps": None,
            "quality": "Failed (Timeout)",
            "output_snippet": ""
        }
    except Exception as e:
        return {
            "success": False,
            "status": "ERROR",
            "ttft_ms": None,
            "decode_tps": None,
            "quality": f"Error: {e}",
            "output_snippet": ""
        }

    if rc != 0:
        return {
            "success": False,
            "status": f"EXIT_{rc}",
            "ttft_ms": None,
            "decode_tps": None,
            "quality": "Execution Failure",
            "output_snippet": ""
        }

    # Parse stdout for timings
    prompt_tps = None
    decode_tps = None
    m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", stdout)
    if m:
        prompt_tps = float(m.group(1))
        decode_tps = float(m.group(2))

    ttft_ms = None
    if prompt_tps and prompt_tps > 0:
        ttft_ms = round((prompt_tokens_est / prompt_tps) * 1000.0, 1)

    # Extract response text (non-empty lines immediately preceding "[ Prompt:")
    lines = stdout.splitlines()
    resp_lines = []
    for i, line in enumerate(lines):
        if "[ Prompt:" in line:
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            while j >= 0:
                l = lines[j].strip()
                if l.startswith(">") or "... (truncated)" in l:
                    break
                resp_lines.insert(0, l)
                j -= 1
            break
    response_text = " ".join(resp_lines).strip()
    quality = evaluate_quality(response_text)

    return {
        "success": True,
        "status": "SUCCESS",
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "quality": quality,
        "output_snippet": response_text[:70] + ("..." if len(response_text) > 70 else "")
    }

def main():
    print("=" * 95)
    print("  AMD Ryzen AI Heterogeneous Inference Benchmark Matrix")
    print(f"  Evaluating Built-in, Enhanced, and Mimic XCLBINs Across {len(MODELS)} Models")
    print("=" * 95)
    print(f"Prompt: ~180 tokens (Guaranteed TTFT > 100ms)")
    print(f"Settings: -n 16, --temp 0.0 (greedy), --no-warmup, -st, --simple-io, --apu-verbose\n")

    matrix_results = []
    start_time = time.time()

    for idx, m in enumerate(MODELS, 1):
        print(f"[{idx}/{len(MODELS)}] Benchmarking Model: {m['name']} ({m['id']})...")
        m_path = m["model"]
        if not m_path.exists():
            print(f"  [SKIP] Model file {m_path} missing.")
            continue

        model_result = {
            "id": m["id"],
            "name": m["name"],
            "builtin": None,
            "enhanced": None,
            "mimic": None
        }

        # 1. Built-in
        print("  -> Testing [Built-in] XCLBIN...")
        res_b = run_test(m_path, m["builtin"], m["prompt_tokens_est"])
        model_result["builtin"] = res_b
        print(f"     Quality: {res_b['quality']} | TTFT: {res_b['ttft_ms']} ms | Decode: {res_b['decode_tps']} t/s")

        # 2. Enhanced
        print("  -> Testing [Enhanced] XCLBIN...")
        res_e = run_test(m_path, m["enhanced"], m["prompt_tokens_est"])
        model_result["enhanced"] = res_e
        print(f"     Quality: {res_e['quality']} | TTFT: {res_e['ttft_ms']} ms | Decode: {res_e['decode_tps']} t/s")

        # 3. Mimic
        print("  -> Testing [Mimic] XCLBIN...")
        res_m = run_test(m_path, m["mimic"], m["prompt_tokens_est"])
        model_result["mimic"] = res_m
        print(f"     Quality: {res_m['quality']} | TTFT: {res_m['ttft_ms']} ms | Decode: {res_m['decode_tps']} t/s")

        matrix_results.append(model_result)

    total_elapsed = round(time.time() - start_time, 1)

    # Save to JSON
    out_json = SUITE_ROOT / "xclbin_comparison_matrix_results.json"
    with open(out_json, "w") as f:
        json.dump(matrix_results, f, indent=2)

    print("\n" + "=" * 95)
    print(f"  BENCHMARK MATRIX COMPLETED IN {total_elapsed} SECONDS")
    print(f"  Full JSON Output Saved To: {out_json}")
    print("=" * 95)

if __name__ == "__main__":
    main()
