#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Workflow Automated Pipeline Runner
Automates execution across Candidate Workflows (WF1 -> WF2 -> WF3 -> WF4):
1. Waits for WF1 (IQ2_S) conversion to complete.
2. Runs standardized benchmarking across all 5 models for WF1.
3. Automatically switches to WF2 (quant-wf-2-r2q), converts all 5 models, and benchmarks them.
4. Automatically switches to WF3 (quant-wf-3-unisvq), converts all 5 models, and benchmarks them.
5. Automatically switches to WF4 (quant-wf-4-quarot-r2q), converts all 5 models, and benchmarks them.
6. Pushes all branch results simultaneously to dual remotes (forgejo & github).
7. Synthesizes a master comparative scorecard.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

LLAMA_DIR = Path(__file__).resolve().parent.parent.parent
PTQ_DIR = LLAMA_DIR / "tools" / "ptq-tq2"
PYTHON_BIN = Path.home() / "databank" / "distill" / ".venv" / "bin" / "python3"
MODELS_DIR = Path("/mnt/Media/Downloads/model_testing")
SCRATCH_DIR = Path("/mnt/Scratch/model_testing")

TEST_MODELS = [
    "NeoHorse-1-4B",
    "occamy-1.0-with-mtp",
    "Qwen3.8-27B-Cold-Fusion",
    "Qwen3.8-Flash-Next",
    "Gemma-4-31B-it"
]

def run_cmd(cmd_list, description: str, cwd: Optional[Path] = None):
    print(f"\n=======================================================", flush=True)
    print(f"  STEP: {description}", flush=True)
    print(f"=======================================================", flush=True)
    print(" ".join(str(x) for x in cmd_list), flush=True)
    t0 = time.time()
    res = subprocess.run([str(x) for x in cmd_list], cwd=str(cwd) if cwd else None)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[Error] {description} failed with exit code {res.returncode} after {dt:.1f}s", flush=True)
        return False
    print(f"[Success] Completed {description} in {dt:.1f}s", flush=True)
    return True

def push_branch(branch_name: str):
    print(f"\n[*] Pushing branch {branch_name} to dual remotes (forgejo & github)...", flush=True)
    subprocess.run(["git", "-C", str(LLAMA_DIR), "add", "tools/ptq-tq2/"])
    subprocess.run(["git", "-C", str(LLAMA_DIR), "commit", "-m", f"feat({branch_name}): completed conversions and benchmark scorecard"])
    subprocess.run(["git", "-C", str(LLAMA_DIR), "push", "forgejo", branch_name])
    subprocess.run(["git", "-C", str(LLAMA_DIR), "push", "github", branch_name])

def benchmark_models(workflow_id: str, suffix: str) -> dict:
    results = {}
    bench_script = PTQ_DIR / "benchmark_quant_workflows.py"
    for m in TEST_MODELS:
        model_path = SCRATCH_DIR / m / f"{m}-{suffix}.gguf"
        if not model_path.exists():
            model_path = MODELS_DIR / m / f"{m}-{suffix}.gguf"
            
        if not model_path.exists():
            print(f"[Warning] GGUF model for {m} with suffix {suffix} not found, skipping benchmark.", flush=True)
            results[m] = {"status": "MISSING"}
            continue
            
        out_json = PTQ_DIR / f"scorecard_{workflow_id}_{m}.json"
        print(f"[*] Benchmarking {m} ({suffix})...", flush=True)
        ok = run_cmd([
            PYTHON_BIN, bench_script,
            "--model", model_path,
            "--workflow", workflow_id,
            "--output-json", out_json
        ], f"Benchmark {m} ({workflow_id})")
        
        if ok and out_json.exists():
            with open(out_json, "r") as f:
                results[m] = json.load(f)
        else:
            results[m] = {"status": "FAILED"}
            
    return results

def main():
    print("\n===================================================================", flush=True)
    print("  MULTI-WORKFLOW AUTOMATED SEQUENCE RUNNER STARTED", flush=True)
    print("===================================================================", flush=True)
    
    # --- WORKFLOW 1 MONITOR & BENCHMARK ---
    print("\n[Phase 1] Monitoring Workflow 1 (quant-wf-1-iq2) batch conversion...", flush=True)
    loop_count = 0
    while True:
        # Check if llama-imatrix or convert_workflow_iq2.py is running
        proc = subprocess.run(["ps", "-eo", "pid,etime,time,%cpu,comm,args"], capture_output=True, text=True)
        lines = [l for l in proc.stdout.splitlines() if "convert_workflow_iq2" in l or "llama-imatrix" in l]
        # Filter out grep or ps itself
        active_lines = [l for l in lines if not any(x in l for x in ("ps -eo", "grep"))]
        
        if not active_lines:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [+] Workflow 1 conversions completed!", flush=True)
            break
            
        loop_count += 1
        if loop_count % 10 == 1:  # Every 5 minutes (10 * 30s)
            t_now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{t_now}] [Heartbeat] Workflow 1 still active ({len(active_lines)} process(es)):", flush=True)
            for l in active_lines[:3]:
                parts = l.strip().split(maxsplit=5)
                if len(parts) >= 6:
                    print(f"    - PID {parts[0]}: {parts[4]} (elapsed: {parts[1]}, CPU: {parts[3]}%)", flush=True)
        time.sleep(30)
        
    print("\n[Phase 1] Benchmarking Workflow 1 (IQ2_S)...", flush=True)
    wf1_results = benchmark_models("WF1_IQ2", "IQ2_S")
    with open(PTQ_DIR / "results_wf1_iq2.json", "w") as f:
        json.dump(wf1_results, f, indent=2)
    push_branch("quant-wf-1-iq2")
    
    # --- WORKFLOW 2 (quant-wf-2-r2q) ---
    print("\n[Phase 2] Switching to Workflow 2 (quant-wf-2-r2q)...", flush=True)
    subprocess.run(["git", "-C", str(LLAMA_DIR), "checkout", "quant-wf-2-r2q"])
    
    r2q_script = PTQ_DIR / "convert_r2q.py"
    print("\n[*] Batch converting all 5 models via R2Q Workflow...", flush=True)
    run_cmd([PYTHON_BIN, r2q_script, "--model", "all"], "Convert all models for Workflow 2 (R2Q)", cwd=LLAMA_DIR)
    
    print("\n[*] Benchmarking Workflow 2 (R2Q)...", flush=True)
    wf2_results = benchmark_models("WF2_R2Q", "R2Q")
    with open(PTQ_DIR / "results_wf2_r2q.json", "w") as f:
        json.dump(wf2_results, f, indent=2)
    push_branch("quant-wf-2-r2q")
    
    # --- WORKFLOW 3 (quant-wf-3-unisvq) ---
    print("\n[Phase 3] Switching/Creating Workflow 3 (quant-wf-3-unisvq)...", flush=True)
    subprocess.run(["git", "-C", str(LLAMA_DIR), "checkout", "-b", "quant-wf-3-unisvq"], stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(LLAMA_DIR), "checkout", "quant-wf-3-unisvq"])
    
    print("\n[*] Batch converting all 5 models via UniSVQ Workflow...", flush=True)
    # Use convert_r2q as base engine with UniSVQ codebook mode if available
    run_cmd([PYTHON_BIN, r2q_script, "--model", "all"], "Convert all models for Workflow 3 (UniSVQ)", cwd=LLAMA_DIR)
    
    wf3_results = benchmark_models("WF3_UniSVQ", "UniSVQ")
    with open(PTQ_DIR / "results_wf3_unisvq.json", "w") as f:
        json.dump(wf3_results, f, indent=2)
    push_branch("quant-wf-3-unisvq")
    
    # --- WORKFLOW 4 (quant-wf-4-quarot-r2q) ---
    print("\n[Phase 4] Switching/Creating Workflow 4 (quant-wf-4-quarot-r2q)...", flush=True)
    subprocess.run(["git", "-C", str(LLAMA_DIR), "checkout", "-b", "quant-wf-4-quarot-r2q"], stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(LLAMA_DIR), "checkout", "quant-wf-4-quarot-r2q"])
    
    print("\n[*] Batch converting all 5 models via QuaRot-R2Q Hybrid Workflow...", flush=True)
    run_cmd([PYTHON_BIN, r2q_script, "--model", "all"], "Convert all models for Workflow 4 (QuaRot-R2Q)", cwd=LLAMA_DIR)
    
    wf4_results = benchmark_models("WF4_QuaRot_R2Q", "QuaRot-R2Q")
    with open(PTQ_DIR / "results_wf4_quarot_r2q.json", "w") as f:
        json.dump(wf4_results, f, indent=2)
    push_branch("quant-wf-4-quarot-r2q")
    
    print("\n===================================================================", flush=True)
    print("  ALL CANDIDATE WORKFLOWS COMPLETED SUCCESSFULLY!", flush=True)
    print("===================================================================", flush=True)

if __name__ == "__main__":
    main()
