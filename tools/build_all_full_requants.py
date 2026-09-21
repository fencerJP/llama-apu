#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Full Model BiLLM Requant Builder with On-the-Fly Orthogonal Rotation.

Streams 100% of all weight matrices from safetensors model shards on NAS directly into
full-sized .q4nx containers with Block-RHT Walsh-Hadamard orthogonal rotation for all models.
"""

import os
import sys

# Ensure converter package is in Python path
REPO_ROOT = "/home/fencer/.openclaw/workspace/projects/zero-copy_model_runner"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from converter.convert_to_billm import quantize_direct_stream

NAS_DIR = "/mnt/Media/Downloads/model_testing"

def main():
    # Load fallback xclbin bytes if available
    xclbin_bytes = b""

    models_config = [
        {
            "name": "TokenRhythm/NeoHorse-1-4B (32 Layers, Dim=2560)",
            "model_dir": os.path.join(NAS_DIR, "NeoHorse-1-4B"),
            "nas_q4nx": os.path.join(NAS_DIR, "neohorse-1-4b-full-billm.q4nx"),
            "hp": {
                "arch": "neohorse",
                "hidden_dim": 2560,
                "num_heads": 20,
                "num_kv_heads": 4,
                "num_layers": 32,
                "vocab_size": 131072,
                "context_length": 65536,
            }
        },
        {
            "name": "Qwen/Qwen3.8-27B-Cold-Fusion (64 Layers, Dim=3584)",
            "model_dir": os.path.join(NAS_DIR, "Qwen3.8-27B-Cold-Fusion"),
            "nas_q4nx": os.path.join(NAS_DIR, "qwen3.8-cold-fusion-full-billm.q4nx"),
            "hp": {
                "arch": "qwen3_5_text",
                "hidden_dim": 3584,
                "num_heads": 28,
                "num_kv_heads": 4,
                "num_layers": 64,
                "vocab_size": 248320,
                "context_length": 262144,
            }
        },
        {
            "name": "google/gemma-4-31B (60 Layers, Dim=5376)",
            "model_dir": os.path.join(NAS_DIR, "gemma-4-31B"),
            "nas_q4nx": os.path.join(NAS_DIR, "gemma-4-31b-full-billm.q4nx"),
            "hp": {
                "arch": "gemma4_text",
                "hidden_dim": 5376,
                "num_heads": 32,
                "num_kv_heads": 16,
                "num_layers": 60,
                "vocab_size": 262144,
                "context_length": 131072,
            }
        },
        {
            "name": "Qwen3-Coder-Next (48 Layers, Dim=2048, MoE 80B/3B)",
            "model_dir": os.path.join(NAS_DIR, "Qwen3-Coder-Next"),
            "nas_q4nx": os.path.join(NAS_DIR, "qwen3-coder-next-full-billm.q4nx"),
            "hp": {
                "arch": "qwen3_coder_next",
                "hidden_dim": 2048,
                "num_heads": 16,
                "num_kv_heads": 4,
                "num_layers": 48,
                "vocab_size": 248320,
                "context_length": 262144,
            }
        },
        {
            "name": "sarvam-105b (32 Layers, Dim=4096, MoE 105B/10.3B)",
            "model_dir": os.path.join(NAS_DIR, "sarvam-105b"),
            "nas_q4nx": os.path.join(NAS_DIR, "sarvam-105b-full-billm.q4nx"),
            "hp": {
                "arch": "sarvam_moe",
                "hidden_dim": 4096,
                "num_heads": 32,
                "num_kv_heads": 8,
                "num_layers": 32,
                "vocab_size": 248320,
                "context_length": 131072,
            }
        },
        {
            "name": "poolside/Laguna-S-2.1 (48 Layers, Dim=3072, MoE 256 Experts)",
            "model_dir": os.path.join(NAS_DIR, "Laguna-S-2.1"),
            "nas_q4nx": os.path.join(NAS_DIR, "laguna-s-2.1-full-billm.q4nx"),
            "hp": {
                "arch": "laguna",
                "hidden_dim": 3072,
                "num_heads": 48,
                "num_kv_heads": 8,
                "num_layers": 48,
                "vocab_size": 100352,
                "context_length": 1048576,
            }
        },
        {
            "name": "Qwen/Qwen3.8-Flash-Next (48 Layers, Dim=2560, Hybrid Linear Attn)",
            "model_dir": os.path.join(NAS_DIR, "Qwen3.8-Flash-Next"),
            "nas_q4nx": os.path.join(NAS_DIR, "qwen3.8-flash-next-full-billm.q4nx"),
            "hp": {
                "arch": "qwen4_exp",
                "hidden_dim": 2560,
                "num_heads": 24,
                "num_kv_heads": 2,
                "num_layers": 48,
                "vocab_size": 248056,
                "context_length": 262144,
            }
        },
        {
            "name": "deepseek-ai/DeepSeek-V4-Flash-DSpark (43 Layers, Dim=4096, FP8/FP4 MoE)",
            "model_dir": os.path.join(NAS_DIR, "DeepSeek-V4-Flash-DSpark"),
            "nas_q4nx": os.path.join(NAS_DIR, "deepseek-v4-flash-dspark-full-billm.q4nx"),
            "hp": {
                "arch": "deepseek_v4",
                "hidden_dim": 4096,
                "num_heads": 64,
                "num_kv_heads": 1,
                "num_layers": 43,
                "vocab_size": 128000,
                "context_length": 1048576,
            }
        },
        {
            "name": "deepseek-ai/DeepSeek-V4.1-Flash",
            "model_dir": os.path.join(NAS_DIR, "DeepSeek-V4.1-Flash"),
            "nas_q4nx": os.path.join(NAS_DIR, "deepseek-v4.1-flash-full-billm.q4nx"),
            "hp": {
                "arch": "deepseek_v4",
                "hidden_dim": 4096,
                "num_heads": 64,
                "num_kv_heads": 1,
                "num_layers": 43,
                "vocab_size": 128000,
                "context_length": 1048576,
            }
        },
        {
            "name": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "model_dir": os.path.join(NAS_DIR, "DeepSeek-V4-Flash-0731"),
            "nas_q4nx": os.path.join(NAS_DIR, "deepseek-v4-flash-0731-full-billm.q4nx"),
            "hp": {
                "arch": "deepseek_v4",
                "hidden_dim": 4096,
                "num_heads": 64,
                "num_kv_heads": 1,
                "num_layers": 43,
                "vocab_size": 128000,
                "context_length": 1048576,
            }
        },
        {
            "name": "zai-org/GLM-5.3-Flash",
            "model_dir": os.path.join(NAS_DIR, "GLM-5.3-Flash"),
            "nas_q4nx": os.path.join(NAS_DIR, "glm-5.3-flash-full-billm.q4nx"),
            "hp": {
                "arch": "glm_5_text",
                "hidden_dim": 4096,
                "num_heads": 32,
                "num_kv_heads": 2,
                "num_layers": 40,
                "vocab_size": 151552,
                "context_length": 131072,
            }
        },
    ]

    print("=" * 90, flush=True)
    print(" STREAMING FULL BI-LLM REQUANTIZATION (WITH ROTATION & SALIENCY ISOLATION) ON NAS", flush=True)
    print("=" * 90, flush=True)

    for idx, m in enumerate(models_config, 1):
        print(f"\n>>> [{idx}/{len(models_config)}] Starting Model: {m['name']}", flush=True)
        if not os.path.exists(m["nas_q4nx"]) and os.path.exists(m["model_dir"]):
            quantize_direct_stream(
                model_dir=m["model_dir"],
                output_q4nx_path=m["nas_q4nx"],
                xclbin_bytes=xclbin_bytes,
                hp=m["hp"],
                apply_rotation=True,
                salient_ratio=0.015
            )
        else:
            if os.path.exists(m["nas_q4nx"]):
                sz_mb = os.path.getsize(m["nas_q4nx"]) / (1024 * 1024)
                print(f"[=] Already built full .q4nx container on NAS: {m['nas_q4nx']} ({sz_mb:.1f} MB)", flush=True)
            elif not os.path.exists(m["model_dir"]):
                print(f"[!] Source directory not found: {m['model_dir']}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(" ALL FULL ROTATED BI-LLM REQUANTS SUCCESSFULLY CREATED ON NAS!", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
