#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Workflow Quantization Benchmark Suite
Standardized evaluation harness across candidate low-quantization workflows.
Features:
- Automated staging of models from persistent storage (/mnt/Media/Downloads/model_testing)
  to local high-speed NVMe SSD (/home/fencer/.cache/llama-apu-distill-scratch/bench/)
  with automatic pruning after benchmark completion.
- Multi-tier prompt reasoning suite (Easy -> Frontier).
- Perplexity (PPL) measurement via llama-perplexity.
- Automated coherence, repetition loop, and ASCII garbage detection.
- Structured JSON output with complete scorecard metrics.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

LLAMA_DIR = Path(__file__).resolve().parent.parent.parent
BUILD_BIN = LLAMA_DIR / "build" / "bin"
LLAMA_CLI = BUILD_BIN / "llama-cli"
LLAMA_PERPLEXITY = BUILD_BIN / "llama-perplexity"

DEFAULT_SCRATCH_DIR = Path("/home/fencer/.cache/llama-apu-distill-scratch/bench")
DEFAULT_DATABANK_SAMPLES = Path("/home/fencer/databank/distill/calibration_samples.txt")

BENCHMARK_QUESTIONS = [
    {
        "id": "Q1_EASY",
        "difficulty": "Easy",
        "category": "General Knowledge",
        "prompt": "What is the capital of France, and what river runs through it?",
        "expected_keywords": ["paris", "seine"]
    },
    {
        "id": "Q2_EASY_MED",
        "difficulty": "Easy-Medium",
        "category": "Arithmetic",
        "prompt": "If you have 15 pencils and give 4 to your friend and 3 to your sister, how many pencils do you have left?",
        "expected_keywords": ["8", "eight"]
    },
    {
        "id": "Q3_MEDIUM",
        "difficulty": "Medium",
        "category": "Coding (Python)",
        "prompt": "Write a Python function is_even(n) that returns True if n is an even integer, and False otherwise.",
        "expected_keywords": ["def is_even", "% 2"]
    },
    {
        "id": "Q4_MED_HARD",
        "difficulty": "Medium-Hard",
        "category": "Logical Deduction",
        "prompt": "No reptiles have fur. All snakes are reptiles. Can a snake have fur? Answer with YES or NO and a brief explanation.",
        "expected_keywords": ["no"]
    },
    {
        "id": "Q5_HARD",
        "difficulty": "Hard",
        "category": "Mathematics",
        "prompt": "What is the derivative of f(x) = 3*x^2 + 5*x - 7 with respect to x?",
        "expected_keywords": ["6*x + 5", "6x + 5", "6x+5"]
    },
    {
        "id": "Q6_FRONTIER",
        "difficulty": "Frontier",
        "category": "Distributed Systems",
        "prompt": "Explain what the Byzantine Generals Problem is in distributed consensus and how many faulty nodes a system of N nodes can tolerate.",
        "expected_keywords": ["consensus", "fault", "3m + 1", "(n-1)/3", "f < n/3"]
    }
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
    # Remove thinking tags if present
    text = re.sub(r"\[Start thinking\].*?\[End thinking\]", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text if text else raw_text[-500:].strip()

def detect_gibberish_or_repetition(text: str) -> Dict[str, Any]:
    """
    Detects if the response collapsed into gibberish, unicode/ASCII noise,
    or degenerate repetitive n-gram loops.
    """
    if not text or len(text.strip()) == 0:
        return {"coherent": False, "reason": "Empty output"}
    
    # Check 1: Non-ASCII / high-entropy symbol ratio
    # If text is filled with random non-latin / control / corrupted unicode tokens
    weird_chars = len(re.findall(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]", text))
    if weird_chars > 3:
        return {"coherent": False, "reason": f"Corrupted tokens / replacement chars detected ({weird_chars})"}
    
    # Check 2: Repetitive word loops
    words = text.lower().split()
    if len(words) >= 8:
        # Check consecutive 3-grams
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        for tri in set(trigrams):
            if trigrams.count(tri) >= 4:
                return {"coherent": False, "reason": f"Degenerate repetition loop: '{tri}'"}
                
    # Check 3: Character repetition loop (e.g. aaaaaa or !!!!!!!!)
    if re.search(r"(.)\1{7,}", text):
        return {"coherent": False, "reason": "Character repetition loop"}

    return {"coherent": True, "reason": "OK"}

def run_prompt_evaluation(model_path: Path, prompt_entry: Dict[str, Any], max_tokens: int = 128, timeout_sec: int = 120) -> Dict[str, Any]:
    cmd = [
        str(LLAMA_CLI),
        "-m", str(model_path),
        "-p", prompt_entry["prompt"],
        "-n", str(max_tokens),
        "-c", "512",
        "--simple-io",
        "--single-turn",
        "--temp", "0.2"
    ]
    
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        dt = time.time() - t0
        output = proc.stdout + proc.stderr
        
        prompt_speed = 0.0
        gen_speed = 0.0
        m = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", output)
        if m:
            prompt_speed = float(m.group(1))
            gen_speed = float(m.group(2))
            
        cleaned = clean_output(output)
        coherence_info = detect_gibberish_or_repetition(cleaned)
        
        # Keyword matching check
        kw_matched = any(kw.lower() in cleaned.lower() for kw in prompt_entry.get("expected_keywords", []))
        
        return {
            "success": proc.returncode == 0 and coherence_info["coherent"],
            "response": cleaned,
            "raw_output": output,
            "latency_sec": round(dt, 2),
            "prompt_speed_tps": prompt_speed,
            "gen_speed_tps": gen_speed,
            "coherent": coherence_info["coherent"],
            "coherence_reason": coherence_info["reason"],
            "keyword_matched": kw_matched
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "response": f"TIMEOUT (>{timeout_sec}s)",
            "latency_sec": float(timeout_sec),
            "prompt_speed_tps": 0.0,
            "gen_speed_tps": 0.0,
            "coherent": False,
            "coherence_reason": "Timeout exceeded"
        }
    except Exception as e:
        return {
            "success": False,
            "response": f"ERROR: {str(e)}",
            "latency_sec": 0.0,
            "prompt_speed_tps": 0.0,
            "gen_speed_tps": 0.0,
            "coherent": False,
            "coherence_reason": str(e)
        }

def run_ppl_evaluation(model_path: Path, test_file: Path, max_chunks: int = 4, timeout_sec: int = 180) -> Dict[str, Any]:
    """Runs llama-perplexity over test chunks."""
    if not LLAMA_PERPLEXITY.exists() or not test_file.exists():
        return {"success": False, "ppl": None, "note": "Binary or test file missing"}
        
    cmd = [
        str(LLAMA_PERPLEXITY),
        "-m", str(model_path),
        "-f", str(test_file),
        "-c", "512",
        "--chunks", str(max_chunks)
    ]
    
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        dt = time.time() - t0
        output = proc.stdout + proc.stderr
        
        # Parse PPL from output: "Final estimate: PPL = 12.3456 +/- 0.1234"
        ppl = None
        m = re.search(r"PPL\s*=\s*([\d\.]+)", output)
        if m:
            ppl = float(m.group(1))
            
        return {
            "success": proc.returncode == 0 and (ppl is not None),
            "ppl": ppl,
            "latency_sec": round(dt, 2),
            "raw_output": output[-500:] if output else ""
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "ppl": None, "note": "Timeout exceeded"}
    except Exception as e:
        return {"success": False, "ppl": None, "note": str(e)}

def stage_and_evaluate_model(
    model_source_path: Path,
    workflow_id: str,
    scratch_dir: Path = DEFAULT_SCRATCH_DIR,
    ppl_file: Optional[Path] = DEFAULT_DATABANK_SAMPLES,
    skip_ppl: bool = False,
    timeout_sec: int = 120
) -> Dict[str, Any]:
    """
    Handles local NVMe staging & auto-pruning lifecycle for a single model:
    1. Copies model to scratch_dir if on persistent/remote storage.
    2. Runs full evaluation suite on local NVMe.
    3. Prunes (deletes) the local copy immediately.
    """
    model_source_path = Path(model_source_path)
    if not model_source_path.exists():
        raise FileNotFoundError(f"Source model does not exist: {model_source_path}")
        
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    is_already_local = (scratch_dir in model_source_path.parents) or ("/tmp" in str(model_source_path))
    
    if is_already_local:
        eval_model_path = model_source_path
        staged = False
    else:
        eval_model_path = scratch_dir / model_source_path.name
        print(f"\n[Staging] Copying {model_source_path.name} to local NVMe scratch ({scratch_dir})...")
        t_copy_0 = time.time()
        shutil.copy2(model_source_path, eval_model_path)
        print(f"[Staging] Copy completed in {time.time() - t_copy_0:.2f}s.")
        staged = True
        
    results = {
        "workflow_id": workflow_id,
        "model_name": model_source_path.name,
        "source_path": str(model_source_path),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions": {},
        "summary": {}
    }
    
    try:
        print(f"\n===================================================================")
        print(f"  BENCHMARKING: {model_source_path.name} [Workflow: {workflow_id}]")
        print(f"===================================================================")
        
        passed_questions = 0
        total_prompt_speed = 0.0
        total_gen_speed = 0.0
        speed_samples = 0
        
        for q in BENCHMARK_QUESTIONS:
            qid = q["id"]
            print(f"\n[{q['difficulty']}] {q['category']}:")
            print(f"  Prompt: {q['prompt']}")
            
            res = run_prompt_evaluation(eval_model_path, q, timeout_sec=timeout_sec)
            
            status_str = "PASS" if res["success"] else f"FAIL ({res['coherence_reason']})"
            print(f"  Result: {status_str} | Speed: {res['gen_speed_tps']} t/s | Latency: {res['latency_sec']}s")
            print(f"  Generated: >>> {res['response'][:160]}...")
            
            results["questions"][qid] = res
            
            if res["success"]:
                passed_questions += 1
            if res["gen_speed_tps"] > 0:
                total_gen_speed += res["gen_speed_tps"]
                total_prompt_speed += res["prompt_speed_tps"]
                speed_samples += 1
                
        avg_gen_speed = round(total_gen_speed / max(1, speed_samples), 2)
        avg_prompt_speed = round(total_prompt_speed / max(1, speed_samples), 2)
        
        # Run PPL check if requested
        ppl_result = None
        if not skip_ppl and ppl_file and ppl_file.exists():
            print(f"\n[Perplexity] Evaluating PPL on {ppl_file.name}...")
            ppl_result = run_ppl_evaluation(eval_model_path, ppl_file)
            print(f"[Perplexity] Result: PPL = {ppl_result.get('ppl')} ({ppl_result.get('latency_sec')}s)")
            
        results["summary"] = {
            "total_questions": len(BENCHMARK_QUESTIONS),
            "passed_questions": passed_questions,
            "pass_rate_pct": round(passed_questions / len(BENCHMARK_QUESTIONS) * 100.0, 1),
            "avg_gen_speed_tps": avg_gen_speed,
            "avg_prompt_speed_tps": avg_prompt_speed,
            "ppl_score": ppl_result.get("ppl") if ppl_result else None,
            "overall_status": "COHERENT_PASS" if passed_questions >= 4 else "COLLAPSED_FAIL"
        }
        
    finally:
        if staged and eval_model_path.exists():
            print(f"\n[Auto-Prune] Deleting staged NVMe copy: {eval_model_path}...")
            try:
                os.remove(eval_model_path)
                print("[Auto-Prune] Successfully pruned from local NVMe.")
            except Exception as e:
                print(f"[Auto-Prune Warning] Failed to delete {eval_model_path}: {e}")
                
    return results

def main():
    parser = argparse.ArgumentParser(description="Multi-Workflow Quantization Benchmark Harness")
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf model")
    parser.add_argument("--workflow", type=str, default="custom", help="Workflow identifier (e.g. WF1_IQ2, WF2_R2Q, WF3_UniSVQ, WF4_QuaRot_R2Q)")
    parser.add_argument("--scratch-dir", type=str, default=str(DEFAULT_SCRATCH_DIR), help="Local NVMe scratch staging directory")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save benchmark JSON output")
    parser.add_argument("--ppl-file", type=str, default=str(DEFAULT_DATABANK_SAMPLES), help="Text corpus file for PPL evaluation")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip PPL measurement for rapid checks")
    parser.add_argument("--timeout", type=int, default=120, help="Per-prompt timeout in seconds")
    
    args = parser.parse_args()
    
    res = stage_and_evaluate_model(
        model_source_path=Path(args.model),
        workflow_id=args.workflow,
        scratch_dir=Path(args.scratch_dir),
        ppl_file=Path(args.ppl_file) if args.ppl_file else None,
        skip_ppl=args.skip_ppl,
        timeout_sec=args.timeout
    )
    
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[+] Results written to: {out_path}")
    else:
        print("\n=== SUMMARY SCORECARD ===")
        print(json.dumps(res["summary"], indent=2))

if __name__ == "__main__":
    main()
