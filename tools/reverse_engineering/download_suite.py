import os
import sys
import subprocess
from pathlib import Path

DEST_ROOT = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/benchmark_suite")
DEST_ROOT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "name": "deepseek-r1-qwen3-8b",
        "downloads": [
            ("hf", "download", "FastFlowLM/DeepSeek-R1-0528-Qwen3-8B-NPU2", "model.q4nx", "layer.xclbin", "mm.xclbin", "attn.xclbin", "config.json"),
            ("hf", "download", "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"),
        ]
    },
    {
        "name": "qwen2.5-3b",
        "downloads": [
            ("hf", "download", "FastFlowLM/Qwen2.5-3B-Instruct-NPU2", "model.q4nx", "config.json"),
            ("hf", "download", "Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
        ]
    },
    {
        "name": "llama-3.2-3b",
        "downloads": [
            ("hf", "download", "FastFlowLM/Llama-3.2-3B-NPU2", "model.q4nx", "layer.xclbin", "mm.xclbin", "attn.xclbin", "config.json"),
            ("hf", "download", "lmstudio-community/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        ]
    },
    {
        "name": "qwen3.5-0.8b",
        "downloads": [
            ("hf", "download", "FastFlowLM/Qwen3.5-0.8B-NPU2", "model.q4nx", "config.json"),
            ("hf", "download", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
        ]
    },
    {
        "name": "gemma4",
        "downloads": [
            ("hf", "download", "FastFlowLM/Gemma4-E2B-IT-NPU2", "model.q4nx", "config.json"),
        ],
        "local_symlinks": [
            ("/opt/models/gemma-4-E4B-heretic.gguf", "gemma-4-E4B-heretic.gguf")
        ]
    }
]

print("=== Starting Model Download Suite ===")
for target in TARGETS:
    target_dir = DEST_ROOT / target["name"]
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>> Target: {target['name']} -> {target_dir}")
    
    for dl in target.get("downloads", []):
        cmd = ["hf", "download", dl[2]]
        # dl[3:] are files
        cmd.extend(dl[3:])
        cmd.extend(["--local-dir", str(target_dir)])
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Error downloading {dl[2]}: exit code {res.returncode}", file=sys.stderr)
            
    for src, dst_name in target.get("local_symlinks", []):
        dst = target_dir / dst_name
        if not dst.exists() and os.path.exists(src):
            print(f"Symlinking local model: {src} -> {dst}")
            dst.symlink_to(src)

print("\n=== Download Suite Complete ===")
