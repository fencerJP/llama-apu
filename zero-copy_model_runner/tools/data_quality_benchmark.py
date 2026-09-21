#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Data Quality & Functional Evaluation Benchmark: Built-in vs Enhanced vs Mimic XCLBINs.

Measures:
1. Intelligible Response Generation (Grammatical, coherent semantic text)
2. Tool Calling Ability (Valid structured JSON function/tool invocation)
3. Strict Pass/Fail Data Quality Determination
"""

import os
import sys
import json
import time
import gc
import subprocess
from pathlib import Path

APU_RUN_BIN = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp/build/bin/apu-run")
LIB_DIR = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp/build/bin")
SUITE_ROOT = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/benchmark_suite")

MODELS = [
    {
        "id": "deepseek-r1-qwen3-8b",
        "name": "DeepSeek-R1-0528-Qwen3-8B-NPU2 (DeepSeek MoE)",
        "model_path": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "model.q4nx",
        "builtin_xclbin": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "layer.xclbin",
        "enhanced_xclbin": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "generated-layer.xclbin",
        "mimic_xclbin": SUITE_ROOT / "deepseek-r1-qwen3-8b" / "generated-mimic.xclbin",
    },
    {
        "id": "llama-3.2-3b",
        "name": "Llama-3.2-3B-NPU2 (Meta Llama 3.2)",
        "model_path": SUITE_ROOT / "llama-3.2-3b" / "model.q4nx",
        "builtin_xclbin": SUITE_ROOT / "llama-3.2-3b" / "layer.xclbin",
        "enhanced_xclbin": SUITE_ROOT / "llama-3.2-3b" / "generated-layer.xclbin",
        "mimic_xclbin": SUITE_ROOT / "llama-3.2-3b" / "generated-mimic.xclbin",
    },
    {
        "id": "qwen2.5-3b",
        "name": "Qwen2.5-3B-Instruct-NPU2 (Qwen 2.5 Dense)",
        "model_path": SUITE_ROOT / "qwen2.5-3b" / "model.q4nx",
        "builtin_xclbin": SUITE_ROOT / "qwen2.5-3b" / "layer.xclbin",
        "enhanced_xclbin": SUITE_ROOT / "qwen2.5-3b" / "generated-layer.xclbin",
        "mimic_xclbin": SUITE_ROOT / "qwen2.5-3b" / "generated-mimic.xclbin",
    },
    {
        "id": "qwen3.5-0.8b",
        "name": "Qwen3.5-0.8B-NPU2 (GateDeltaNet Hybrid)",
        "model_path": SUITE_ROOT / "qwen3.5-0.8b" / "model.q4nx",
        "builtin_xclbin": SUITE_ROOT / "qwen3.5-0.8b" / "layer.xclbin",
        "enhanced_xclbin": SUITE_ROOT / "qwen3.5-0.8b" / "generated-layer.xclbin",
        "mimic_xclbin": SUITE_ROOT / "qwen3.5-0.8b" / "generated-mimic.xclbin",
    },
    {
        "id": "gemma4",
        "name": "Gemma4-E2B-IT-NPU2 (Google Gemma 4)",
        "model_path": SUITE_ROOT / "gemma4" / "model.q4nx",
        "builtin_xclbin": SUITE_ROOT / "gemma4" / "layer.xclbin",
        "enhanced_xclbin": SUITE_ROOT / "gemma4" / "generated-layer.xclbin",
        "mimic_xclbin": SUITE_ROOT / "gemma4" / "generated-mimic.xclbin",
    },
    {
        "id": "lfm2-1.2b",
        "name": "LFM2-1.2B-NPU2 (Liquid Foundation Model)",
        "model_path": SUITE_ROOT / "lfm2-1.2b" / "model.q4nx",
        "builtin_xclbin": SUITE_ROOT / "lfm2-1.2b" / "layer.xclbin",
        "enhanced_xclbin": SUITE_ROOT / "lfm2-1.2b" / "generated-layer.xclbin",
        "mimic_xclbin": SUITE_ROOT / "lfm2-1.2b" / "generated-mimic.xclbin",
    },
]

PROMPTS = [
    {
        "id": "intelligible_response",
        "name": "Intelligible Response Test",
        "text": "Explain what an operating system kernel does in two clear sentences.",
        "tokens_to_gen": 32,
    },
    {
        "id": "tool_calling",
        "name": "Tool Calling Ability Test",
        "text": 'You are a tool-calling assistant. Available tools: [{"name": "get_current_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]. The user asks: "What is the weather in Tokyo?" Respond ONLY with a valid JSON tool call.',
        "tokens_to_gen": 32,
    },
]

def evict_memory():
    gc.collect()
    try:
        os.sync()
    except Exception:
        pass
    time.sleep(1.0)

def evaluate_data_quality(prompt_id, output_text, tokens_generated):
    """
    Evaluates whether the output text meets acceptable data quality standards.
    """
    text = output_text.strip()
    
    # Baseline check: did it generate more than 3 tokens?
    if tokens_generated < 4 or len(text) < 10:
        return {
            "verdict": "FAIL",
            "reason": f"Premature generation cutoff ({tokens_generated} tokens, length {len(text)} chars). Immediate EOS or truncated stub.",
            "is_intelligible": False,
            "tool_call_valid": False,
        }

    if prompt_id == "intelligible_response":
        # Needs to have real English words and coherent structure
        words = [w for w in text.split() if len(w) > 1 and w.isalpha()]
        has_min_words = len(words) >= 5
        
        # Check for gibberish or non-ASCII noise
        ascii_ratio = sum(1 for c in text if c.isascii()) / max(1, len(text))
        is_clean = ascii_ratio > 0.85
        
        is_intelligible = has_min_words and is_clean
        return {
            "verdict": "PASS" if is_intelligible else "FAIL",
            "reason": "Coherent natural language response" if is_intelligible else "Incoherent, disjointed, or foreign-token gibberish",
            "is_intelligible": is_intelligible,
            "tool_call_valid": False,
        }

    elif prompt_id == "tool_calling":
        # Must contain valid JSON and mention get_current_weather and Tokyo
        tool_call_valid = False
        reason = "Output is not valid JSON"
        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                data = json.loads(json_str)
                # Check for tool name and argument
                has_name = "get_current_weather" in json_str
                has_tokyo = "Tokyo" in json_str or "tokyo" in json_str
                if has_name and has_tokyo:
                    tool_call_valid = True
                    reason = "Valid structured JSON tool call emitted"
                else:
                    reason = f"JSON parsed but missing tool name ('get_current_weather') or city ('Tokyo'): {data}"
            else:
                reason = "No JSON object brackets found in response"
        except Exception as e:
            reason = f"JSON parse error: {str(e)}"

        return {
            "verdict": "PASS" if tool_call_valid else "FAIL",
            "reason": reason,
            "is_intelligible": len(text) > 10,
            "tool_call_valid": tool_call_valid,
        }

    return {"verdict": "FAIL", "reason": "Unknown prompt type", "is_intelligible": False, "tool_call_valid": False}

def run_inference(model_path, xclbin_path, prompt_text, steps=32):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        str(APU_RUN_BIN),
        "-m", str(model_path),
        "-p", prompt_text,
        "-n", str(steps),
        "--temp", "0.0",
        "--seed", "42",
        "--verbose"
    ]
    if xclbin_path and xclbin_path.exists():
        cmd.extend(["-x", str(xclbin_path)])

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=40
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = proc.returncode
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output_text": "",
            "tokens_generated": 0,
        }
    finally:
        evict_memory()

    # Extract assistant text
    in_output = False
    output_lines = []
    tokens_gen = 0
    ttft_ms = None
    tps = None

    for line in stdout.splitlines():
        if "Tokens Generated:" in line:
            try:
                tokens_gen = int(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        elif "Prefill TTFT:" in line or "Time to First Token (TTFT):" in line:
            try:
                ttft_ms = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        elif "NPU Generation Speed:" in line:
            try:
                tps = float(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        elif "--- [Step 2: Autoregressive Decode on XDNA 2 NPU] ---" in line:
            in_output = True
            continue
        elif "========================================================" in line and in_output:
            in_output = False
        elif in_output and not line.startswith("Generated Text Output:") and not line.startswith("[VERBOSE]"):
            output_lines.append(line)

    raw_text = " ".join([l.strip() for l in output_lines if l.strip()])
    # Strip ANSI escape sequences
    clean_text = ""
    in_escape = False
    for c in raw_text:
        if c == '\x1b':
            in_escape = True
        elif in_escape:
            if c == 'm':
                in_escape = False
        else:
            clean_text += c

    return {
        "success": rc == 0,
        "returncode": rc,
        "tokens_generated": tokens_gen,
        "ttft_ms": ttft_ms,
        "decode_tps": tps,
        "output_text": clean_text.strip(),
        "stderr": stderr,
    }

def main():
    print("=" * 80)
    print("  XCLBIN Data Quality & Functional Evaluation Benchmark")
    print("  Comparing 3 XCLBIN Types: Built-in vs Enhanced vs Mimic-Builtin")
    print("  Settings: Temp=0.0, Seed=42, Steps=32 (Consistent per model)")
    print("  Evaluation: Pass/Fail based purely on Data Quality & Tool Calling")
    print("=" * 80 + "\n")

    results = []

    for m in MODELS:
        print("\n" + "=" * 75)
        print(f" Model: {m['name']} ({m['id']})")
        print("=" * 75)

        for p in PROMPTS:
            print(f"\n>> Prompt Test: {p['name']} [{p['id']}]")
            print(f"   Prompt Text: \"{p['text'][:60]}...\"")

            # 1. Set A: Built-in
            print("   Evaluating Set A: Built-in XCLBIN...")
            res_a = run_inference(m["model_path"], m["builtin_xclbin"], p["text"], p["tokens_to_gen"])
            eval_a = evaluate_data_quality(p["id"], res_a["output_text"], res_a["tokens_generated"])
            print(f"      Text: \"{res_a['output_text'][:60]}\"")
            print(f"      Verdict: {eval_a['verdict']} | Reason: {eval_a['reason']}")

            # 2. Set B: Enhanced
            print("   Evaluating Set B: Enhanced XCLBIN...")
            res_b = run_inference(m["model_path"], m["enhanced_xclbin"], p["text"], p["tokens_to_gen"])
            eval_b = evaluate_data_quality(p["id"], res_b["output_text"], res_b["tokens_generated"])
            print(f"      Text: \"{res_b['output_text'][:60]}\"")
            print(f"      Verdict: {eval_b['verdict']} | Reason: {eval_b['reason']}")

            # 3. Set C: Mimic
            print("   Evaluating Set C: Mimic-Builtin XCLBIN...")
            res_c = run_inference(m["model_path"], m["mimic_xclbin"], p["text"], p["tokens_to_gen"])
            eval_c = evaluate_data_quality(p["id"], res_c["output_text"], res_c["tokens_generated"])
            print(f"      Text: \"{res_c['output_text'][:60]}\"")
            print(f"      Verdict: {eval_c['verdict']} | Reason: {eval_c['reason']}")

            results.append({
                "model_id": m["id"],
                "model_name": m["name"],
                "prompt_id": p["id"],
                "prompt_name": p["name"],
                "prompt_text": p["text"],
                "set_a_builtin": {**res_a, **eval_a},
                "set_b_enhanced": {**res_b, **eval_b},
                "set_c_mimic": {**res_c, **eval_c},
            })

    out_file = Path("/home/fencer/.openclaw/workspace/projects/zero-copy_model_runner/xclbin_data_quality_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f" Data Quality Benchmark Complete. Results saved to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
