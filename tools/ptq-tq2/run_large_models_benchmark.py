#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Large Models TQ2_0 Benchmark Suite (Stage 3 Full vs Stage 4 Extra)
Evaluates large models (Qwen3.8-27B, Gemma-4-31B, occamy-1.0-with-mtp, Qwen3.8-Flash-Next)
by staging one model at a time to local NVMe SSD scratch, executing the benchmark,
and immediately pruning the local copy to preserve storage.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

LLAMA_CLI = Path("/home/fencer/.openclaw/workspace/projects/llama-apu/llama.cpp/build/bin/llama-cli")
DISTILL_BASE = Path("/mnt/Media/Downloads/model_testing/distill_test")
LOCAL_SCRATCH_DIR = Path("/home/fencer/.cache/llama-apu-distill-scratch/bench")
OUTPUT_JSON = DISTILL_BASE / "large_models_benchmark_results.json"

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
    "Qwen3.8-27B-Cold-Fusion",
    "Gemma-4-31B-it",
    "occamy-1.0-with-mtp",
    "Qwen3.8-Flash-Next"
]

STAGES = [
    ("Stage3-Full", "Full Distillation"),
    ("Stage4-Extra", "Extra Distillation")
]

def clean_output(raw_text: str) -> str:
    """Extracts generated text, removing header banners and footer stats."""
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
    return text if text else raw_text[-500:].strip()

def run_prompt(model_path: Path, prompt: str, max_tokens: int = 64) -> dict:
    cmd = [
        str(LLAMA_CLI),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(max_tokens),
        "-c", "512",
        "--simple-io",
        "--single-turn",
        "--temp", "0.2"
    ]
    
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        dt = time.time() - t0
        output = proc.stdout + proc.stderr
        
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
        return {"success": False, "response": "TIMEOUT (>300s)", "latency_sec": 300.0}
    except Exception as e:
        return {"success": False, "response": f"ERROR: {str(e)}", "latency_sec": 0.0}

def stage_file_to_scratch(source_path: Path, scratch_dir: Path) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    target_path = scratch_dir / source_path.name
    
    # Check free space
    stat = shutil.disk_usage(scratch_dir)
    src_size = source_path.stat().st_size
    print(f"[*] Staging {source_path.name} ({src_size / (1024**3):.2f} GiB) to NVMe scratch...")
    print(f"    Available on NVMe: {stat.free / (1024**3):.2f} GiB")
    
    if stat.free < (src_size + 10 * 1024**3):
        raise RuntimeError(f"Insufficient NVMe space: need {src_size/(1024**3):.1f} GiB + 10 GiB buffer, only {stat.free/(1024**3):.1f} GiB available.")
        
    t0 = time.time()
    subprocess.run(["cp", str(source_path), str(target_path)], check=True)
    elapsed = time.time() - t0
    rate = (src_size / (1024**2)) / max(elapsed, 0.001)
    print(f"[+] Staged in {elapsed:.1f}s ({rate:.1f} MB/s)")
    
    # Also copy sidecar .q4nx if present
    q4nx_src = source_path.with_suffix(".q4nx")
    if q4nx_src.exists():
        q4nx_dst = target_path.with_suffix(".q4nx")
        shutil.copyfile(q4nx_src, q4nx_dst)
        print(f"[+] Staged sidecar: {q4nx_src.name}")
        
    return target_path

def prune_scratch_file(target_path: Path):
    if target_path.exists():
        sz = target_path.stat().st_size
        target_path.unlink()
        print(f"[+] Pruned scratch file: {target_path.name} (reclaimed {sz / (1024**3):.2f} GiB)")
    q4nx = target_path.with_suffix(".q4nx")
    if q4nx.exists():
        q4nx.unlink()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run large models distillation benchmark")
    parser.add_argument("--force", action="store_true", help="Force rerun of all targets including already passed ones")
    parser.add_argument("--model", type=str, default="", help="Specific model to run (e.g. Qwen3.8-Flash-Next)")
    args = parser.parse_args()

    print("===================================================================")
    print("  llama-apu: Large Models Benchmark (Full vs Extra Distillation)  ")
    print("  Models: Qwen3.8-27B, Gemma-4-31B, occamy-1.0-with-mtp, Flash-Next")
    print("  Policy: Local NVMe Staging (One-at-a-time) + Immediate Pruning  ")
    print("===================================================================")
    sys.stdout.flush()
    
    results = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
            
    models_to_run = [m for m in MODELS if (not args.model or args.model.lower() in m.lower())]

    for mname in models_to_run:
        mdir = DISTILL_BASE / mname
        if not mdir.exists():
            print(f"[!] Directory not found: {mdir}, skipping.")
            continue
            
        for stage_key, stage_desc in STAGES:
            run_key = f"{mname}_{stage_key}"
            
            # Check if this target already succeeded completely
            if run_key in results and not args.force:
                stage_data = results[run_key]
                questions_dict = stage_data.get("questions", {})
                if len(questions_dict) == len(QUESTIONS) and all(q.get("result", {}).get("success", False) for q in questions_dict.values()):
                    print(f"[*] Skipping {run_key} — all {len(QUESTIONS)} questions already passed successfully.")
                    continue

            cand = list(mdir.glob(f"*{stage_key}*.gguf"))
            if not cand:
                print(f"[!] No file matching *{stage_key}*.gguf in {mdir}, skipping.")
                continue
            source_gguf = cand[0]
            
            print(f"\n" + "="*70)
            print(f"  TARGET: {mname} | {stage_desc}")
            print(f"  Source: {source_gguf}")
            print("="*70)
            sys.stdout.flush()
            
            scratch_path = None
            try:
                # 1. Stage to local NVMe scratch
                scratch_path = stage_file_to_scratch(source_gguf, LOCAL_SCRATCH_DIR)
                
                stage_results = {
                    "model": mname,
                    "stage": stage_key,
                    "stage_desc": stage_desc,
                    "source_path": str(source_gguf),
                    "questions": {}
                }
                
                # 2. Run questions
                for q in QUESTIONS:
                    qid = q["id"]
                    print(f"\n  [{q['difficulty']}] {q['category']}:")
                    print(f"    Prompt: {q['prompt']}")
                    sys.stdout.flush()
                    
                    res = run_prompt(scratch_path, q["prompt"])
                    print(f"    Speed: {res.get('gen_speed_tps', 0)} t/s (prompt: {res.get('prompt_speed_tps', 0)} t/s, latency: {res.get('latency_sec')}s)")
                    resp_preview = res['response'][:160].replace('\n', ' ')
                    print(f"    Response: {resp_preview}...")
                    sys.stdout.flush()
                    
                    stage_results["questions"][qid] = {
                        "difficulty": q["difficulty"],
                        "category": q["category"],
                        "prompt": q["prompt"],
                        "result": res
                    }
                    
                results[run_key] = stage_results
                
                # Save partial results
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n[+] Intermediate results saved to: {OUTPUT_JSON}")
                sys.stdout.flush()
                
            finally:
                # 3. Always prune scratch file immediately
                if scratch_path:
                    prune_scratch_file(scratch_path)
                    sys.stdout.flush()
                    
    print("\n" + "="*70)
    print(f"  ALL BENCHMARKS COMPLETE! Results saved to:")
    print(f"  {OUTPUT_JSON}")
    print("="*70)

if __name__ == "__main__":
    main()
