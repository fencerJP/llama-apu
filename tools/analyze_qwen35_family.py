#!/usr/bin/env python3
import subprocess
from pathlib import Path

SRC_DIR = Path("/home/fencer/.openclaw/workspace/projects/fastflowlm/src/xclbins")
OUT_DIR = Path(__file__).resolve().parent.parent / "old" / "reverse_engineering" / "qwen35_analysis"
DISASM = Path(__file__).resolve().parent / "xclbin_disassembler.py"

TARGET_MODELS = [
    "Qwen3.5-0.8B-NPU2",
    "Qwen3.5-2B-NPU2",
    "Qwen3.5-4B-NPU2",
    "Qwen3.5-9B-NPU2",
    "Qwen3.6-35B-A3B-NPU2"
]

KERNELS_TO_ANALYZE = ["layer.xclbin", "GateDeltaNet_prefill.xclbin", "conv.xclbin", "attn.xclbin", "mm.xclbin"]

print("=== Disassembling Qwen 3.5 / 3.6 Family Kernels ===")
for model in TARGET_MODELS:
    model_dir = SRC_DIR / model
    out_model_dir = OUT_DIR / model
    out_model_dir.mkdir(parents=True, exist_ok=True)
    
    for k in KERNELS_TO_ANALYZE:
        xclbin_path = model_dir / k
        if not xclbin_path.exists():
            continue
        print(f">> Disassembling {model}/{k} into {out_model_dir}...")
        cmd = ["python3", str(DISASM), str(xclbin_path), "-o", str(out_model_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"   Error: {res.stderr[:200]}")

print("=== Disassembly Complete ===")
