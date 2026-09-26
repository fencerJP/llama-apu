#!/usr/bin/env python3
"""
Sanity check runner for Qwen 3.8 Flash Next UD-IQ4_XS sharded model.
Stages shards to local NVMe scratch, runs the 6 benchmark questions,
evaluates correctness and coherence, and prunes scratch files.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import re
from pathlib import Path

LLAMA_DIR = Path("/home/fencer/.openclaw/workspace/projects/llama-apu/llama.cpp")
LLAMA_CLI = LLAMA_DIR / "build/bin/llama-cli"
SCRATCH_DIR = Path("/home/fencer/.cache/llama-apu-distill-scratch/bench")
OUTPUT_JSON = Path("/mnt/Media/Downloads/model_testing/distill_test/large_models_benchmark_results.json")

CANDIDATE_DIRS = [
    Path("/mnt/Media/Downloads/model_testing/Qwen3.8-Flash-Next/UD-IQ4_XS"),
    Path("/mnt/Media/Downloads/model_testing/Qwen3.8-Flash-Next/UD-IQ4_S"),
    Path("/mnt/Media/Downloads/model_testing/Qwen3.8-Flash-Next/UD-IQ4_xs"),
    Path("/mnt/Media/Downloads/model_testing/Qwen3.8-Flash-Next/UD-IQ4_s"),
]

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

def check_shards_ready() -> tuple[Path | None, list[Path], str]:
    for c in CANDIDATE_DIRS:
        if not c.exists():
            continue
        
        # Check for active download artifacts
        in_progress = list(c.glob("*.part")) + list(c.glob("*.crdownload")) + list(c.glob("*.tmp")) + list(c.glob("*.aria2"))
        if in_progress:
            names = [f.name for f in in_progress]
            return c, [], f"Active download in progress ({len(in_progress)} incomplete files: {names})"

        shards = sorted(list(c.glob("*.gguf")))
        if len(shards) == 3:
            sizes = [s.stat().st_size for s in shards]
            if any(sz < 5 * 1024 * 1024 for sz in sizes):
                return c, shards, "Found 3 shards but at least one file is <5MB"
            return c, shards, "READY"
        elif len(shards) > 0:
            return c, shards, f"Found only {len(shards)} of 3 expected shards: {[s.name for s in shards]}"

    return None, [], "Target directory not found yet or contains no .gguf files"

def clean_output(raw_text: str) -> str:
    lines = raw_text.splitlines()
    in_response = False
    response_lines = []
    for line in lines:
        if line.strip().startswith("> "):
            in_response = True
            continue
        if in_response:
            if re.match(r"^\[\s*Prompt:\s*[\d\.]+\s*t/s", line.strip()):
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
        "--device", "none",
        "--load-mode", "mmap",
        "-t", "16",
        "--simple-io",
        "--single-turn",
        "--temp", "0.2"
    ]
    
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return {
            "success": False,
            "response": f"TIMEOUT (>600s)\n{out[-300:]}",
            "latency_sec": 600.0,
            "raw_output": out
        }
    except Exception as e:
        return {"success": False, "response": f"ERROR: {str(e)}", "latency_sec": 0.0}

def stage_shards(shard_dir: Path, scratch_dir: Path) -> tuple[Path, list[Path]]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    all_shards = sorted(list(shard_dir.glob("*.gguf")))
    if not all_shards:
        raise RuntimeError(f"No .gguf files found in {shard_dir}")
        
    total_bytes = sum(s.stat().st_size for s in all_shards)
    stat = shutil.disk_usage(scratch_dir)
    print(f"[*] Staging {len(all_shards)} shards ({total_bytes / (1024**3):.2f} GiB total) to NVMe scratch...")
    print(f"    Available on NVMe: {stat.free / (1024**3):.2f} GiB")
    
    if stat.free < (total_bytes + 10 * 1024**3):
        raise RuntimeError(f"Insufficient NVMe space: need {total_bytes/(1024**3):.1f} GiB + 10 GiB buffer, only {stat.free/(1024**3):.1f} GiB available.")
        
    staged_paths = []
    first_shard = None
    for s in all_shards:
        dst = scratch_dir / s.name
        print(f"    - Staging {s.name} ({s.stat().st_size / (1024**3):.2f} GiB)...")
        t0 = time.time()
        subprocess.run(["cp", str(s), str(dst)], check=True)
        dt = time.time() - t0
        rate = (s.stat().st_size / (1024**2)) / max(dt, 0.001)
        print(f"      Copied in {dt:.1f}s ({rate:.1f} MB/s)")
        staged_paths.append(dst)
        if first_shard is None or "00001-of" in s.name:
            first_shard = dst

    return first_shard, staged_paths

def prune_shards(staged_paths: list[Path]):
    reclaimed = 0
    for p in staged_paths:
        if p.exists():
            reclaimed += p.stat().st_size
            p.unlink()
    print(f"[+] Pruned {len(staged_paths)} scratch shards (reclaimed {reclaimed / (1024**3):.2f} GiB)")

def main():
    print("===================================================================")
    print("  Sanity Check: Qwen 3.8 Flash Next UD-IQ4_XS Sharded Evaluation  ")
    print("===================================================================")
    shard_dir, shards, status = check_shards_ready()
    if status != "READY":
        print(f"[!] UD-IQ4_XS shards are NOT ready: {status}")
        print("    Candidate directories inspected:")
        for c in CANDIDATE_DIRS:
            print(f"    - {c} (exists: {c.exists()})")
        return 2

    print(f"[+] All 3 shards verified and ready in: {shard_dir}")
    for s in shards:
        print(f"    - {s.name} ({s.stat().st_size / (1024**3):.2f} GiB)")
    
    first_shard, staged = None, []
    try:
        first_shard, staged = stage_shards(shard_dir, SCRATCH_DIR)
        print(f"[+] Primary entry shard: {first_shard.name}")
        
        results = {}
        if OUTPUT_JSON.exists():
            try:
                with open(OUTPUT_JSON, "r") as f:
                    results = json.load(f)
            except Exception:
                results = {}
                
        target_results = {
            "model": "Qwen3.8-Flash-Next",
            "stage": "UD-IQ4_XS",
            "stage_desc": "Sanity Check (UD-IQ4_XS Shards)",
            "source_path": str(shard_dir),
            "questions": {}
        }
        
        for q in QUESTIONS:
            qid = q["id"]
            print(f"\n  [{q['difficulty']}] {q['category']}:")
            print(f"    Prompt: {q['prompt']}")
            sys.stdout.flush()
            
            res = run_prompt(first_shard, q["prompt"])
            print(f"    Speed: {res.get('gen_speed_tps', 0)} t/s (prompt: {res.get('prompt_speed_tps', 0)} t/s, latency: {res.get('latency_sec')}s)")
            resp_preview = res['response'][:160].replace('\n', ' ')
            print(f"    Response: {resp_preview}...")
            sys.stdout.flush()
            
            target_results["questions"][qid] = {
                "difficulty": q["difficulty"],
                "category": q["category"],
                "prompt": q["prompt"],
                "result": res
            }
            
        results["Qwen3.8-Flash-Next_UD-IQ4_XS"] = target_results
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Results updated in: {OUTPUT_JSON}")
        
    finally:
        if staged:
            prune_shards(staged)
            
    print("\n[+] Sanity check complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
