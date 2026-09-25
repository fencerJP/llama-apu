#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Model Question Benchmark Suite
Evaluates distilled TQ2_0 models across 6 difficulty tiers:
1. Easy (Common Knowledge)
2. Easy-Medium (Arithmetic)
3. Medium (Coding)
4. Medium-Hard (Logical Deduction)
5. Hard (Calculus / Math)
6. Frontier (Distributed Systems)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LLAMA_CLI = Path("/home/fencer/.openclaw/workspace/projects/llama-apu/llama.cpp/build/bin/llama-cli")
DISTILL_BASE = Path("/mnt/Media/Downloads/model_testing/distill_test")
OUTPUT_JSON = DISTILL_BASE / "model_benchmark_results.json"

QUESTIONS = [
    {
        "id": "Q1_EASY",
        "difficulty": "Easy",
        "category": "General Knowledge",
        "prompt": "What is the capital of France, and what river runs through it?"
    },
    {
        "id": "Q2_EASY_MED",
        "difficulty": "Easy-Medium",
        "category": "Arithmetic",
        "prompt": "If you have 15 pencils and give 4 to your friend and 3 to your sister, how many pencils do you have left?"
    },
    {
        "id": "Q3_MEDIUM",
        "difficulty": "Medium",
        "category": "Coding (Python)",
        "prompt": "Write a Python function is_even(n) that returns True if n is an even integer, and False otherwise."
    },
    {
        "id": "Q4_MED_HARD",
        "difficulty": "Medium-Hard",
        "category": "Logical Deduction",
        "prompt": "No reptiles have fur. All snakes are reptiles. Can a snake have fur? Answer with YES or NO and a brief explanation."
    },
    {
        "id": "Q5_HARD",
        "difficulty": "Hard",
        "category": "Mathematics",
        "prompt": "What is the derivative of f(x) = 3*x^2 + 5*x - 7 with respect to x?"
    },
    {
        "id": "Q6_FRONTIER",
        "difficulty": "Frontier",
        "category": "Distributed Systems",
        "prompt": "Explain what the Byzantine Generals Problem is in distributed consensus and how many faulty nodes a system of N nodes can tolerate."
    }
]

MODELS = [
    "K2-Horizon-0.9B",
    "NeoHorse-1-4B",
    "NeoHorse-1-9B",
    "K2-Horizon-3.7B",
    "Qwen3.8-27B-Cold-Fusion",
    "Gemma-4-31B-it",
    "occamy-1.0-with-mtp"
]

def clean_output(raw_text: str) -> str:
    """Extracts only the generated model response, stripping banner and stats."""
    lines = raw_text.splitlines()
    in_response = False
    response_lines = []
    
    for line in lines:
        if line.strip().startswith("> "):
            in_response = True
            continue
        if in_response:
            if "[ Prompt:" in line or "Exiting..." in line:
                break
            response_lines.append(line)
            
    text = "\n".join(response_lines).strip()
    # Remove thinking tags if present
    text = re.sub(r"\[Start thinking\].*?\[End thinking\]", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text if text else raw_text[-500:].strip()

def run_prompt(model_path: Path, prompt: str, max_tokens: int = 64) -> dict:
    cmd = [
        str(LLAMA_CLI),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(max_tokens),
        "--simple-io",
        "--single-turn",
        "--temp", "0.2"
    ]
    
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        dt = time.time() - t0
        output = proc.stdout + proc.stderr
        
        # Parse speed stats
        prompt_speed = 0.0
        gen_speed = 0.0
        m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", output)
        if m:
            prompt_speed = float(m.group(1))
            gen_speed = float(m.group(2))
            
        cleaned = clean_output(output)
        return {
            "success": proc.returncode == 0,
            "response": cleaned,
            "raw_output": output,
            "latency_sec": round(dt, 2),
            "prompt_speed_tps": prompt_speed,
            "gen_speed_tps": gen_speed
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "response": "TIMEOUT (>120s)", "latency_sec": 120.0}
    except Exception as e:
        return {"success": False, "response": f"ERROR: {str(e)}", "latency_sec": 0.0}

def main():
    print("===================================================================")
    print("  llama-apu: Multi-Model Benchmark Suite (Easy to Frontier)       ")
    print("===================================================================")
    
    all_results = {}
    
    for mname in MODELS:
        mdir = DISTILL_BASE / mname
        cand = list(mdir.glob("*Stage4-Extra*.gguf"))
        if not cand:
            print(f"[!] Model {mname} not found at {mdir}, skipping.")
            continue
        model_path = cand[0]
        print(f"\n===================================================================")
        print(f"  TESTING MODEL: {mname}")
        print(f"  Path: {model_path.name}")
        print(f"===================================================================")
        
        all_results[mname] = {"model": mname, "path": str(model_path), "questions": {}}
        
        for q in QUESTIONS:
            qid = q["id"]
            print(f"\n[{q['difficulty']}] {q['category']}:")
            print(f"  Prompt: {q['prompt']}")
            res = run_prompt(model_path, q["prompt"])
            print(f"  Generated ({res.get('gen_speed_tps', 0)} t/s, {res.get('latency_sec')}s):")
            print(f"  >>> {res['response'][:200]}...")
            
            all_results[mname]["questions"][qid] = {
                "difficulty": q["difficulty"],
                "category": q["category"],
                "prompt": q["prompt"],
                "result": res
            }
            
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
