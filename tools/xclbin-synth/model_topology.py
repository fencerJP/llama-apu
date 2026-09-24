#!/usr/bin/env python3
"""Milestone 7.1.1: Model Graph Topology Extractor for AMD Ryzen AI APU (AIE2/AIE2P).

Extracts structural hyperparameters from GGUF containers or SafeTensors config.json
without virtual memory bloat or external dependencies.
"""

import json
import os
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelTopology:
    arch_name: str
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int
    context_length: int
    head_dim: int
    ffn_dim: int
    num_experts: int = 0


def parse_gguf_topology(path: str) -> ModelTopology:
    """Bounded, zero-dependency GGUF header parser."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a valid GGUF file: {path} (magic: {magic})")

        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        kv_data: Dict[str, Any] = {}
        # Bounded scan up to 2048 keys
        for _ in range(min(n_kv, 2048)):
            klen_bytes = f.read(8)
            if len(klen_bytes) < 8:
                break
            klen = struct.unpack("<Q", klen_bytes)[0]
            if klen > 512:
                break
            key = f.read(klen).decode("utf-8", errors="replace")

            vtype = struct.unpack("<I", f.read(4))[0]
            # GGUF types: 4=UINT32, 5=INT32, 6=FLOAT32, 8=STRING, 10=UINT64, 11=INT64
            if vtype in (4, 5):
                val = struct.unpack("<I", f.read(4))[0]
                kv_data[key] = val
            elif vtype == 6:
                val = struct.unpack("<f", f.read(4))[0]
                kv_data[key] = val
            elif vtype == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                if slen < 1024:
                    val = f.read(slen).decode("utf-8", errors="replace")
                    kv_data[key] = val
                else:
                    f.seek(slen, os.SEEK_CUR)
            elif vtype in (10, 11):
                val = struct.unpack("<Q", f.read(8))[0]
                kv_data[key] = val
            elif vtype == 9:  # ARRAY
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                # Skip array payload
                elem_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                if arr_type in elem_sizes:
                    f.seek(arr_len * elem_sizes[arr_type], os.SEEK_CUR)
                elif arr_type == 8:  # String array
                    for _ in range(min(arr_len, 4096)):
                        s_len_bytes = f.read(8)
                        if len(s_len_bytes) < 8:
                            break
                        sl = struct.unpack("<Q", s_len_bytes)[0]
                        f.seek(sl, os.SEEK_CUR)
                else:
                    break
            else:
                # Other types, stop scanning
                break

    arch = kv_data.get("general.architecture", "llama")
    hidden_dim = kv_data.get(f"{arch}.embedding_length", kv_data.get("llama.embedding_length", 2048))
    num_heads = kv_data.get(f"{arch}.attention.head_count", kv_data.get("llama.attention.head_count", 32))
    num_kv_heads = kv_data.get(f"{arch}.attention.head_count_kv", num_heads)
    num_layers = kv_data.get(f"{arch}.block_count", kv_data.get("llama.block_count", 16))
    ffn_dim = kv_data.get(f"{arch}.feed_forward_length", kv_data.get("llama.feed_forward_length", hidden_dim * 4))
    context_length = kv_data.get(f"{arch}.context_length", 8192)
    vocab_size = kv_data.get(f"{arch}.vocab_size", 128000)
    head_dim = hidden_dim // max(num_heads, 1)

    # MoE detection
    num_experts = kv_data.get(f"{arch}.expert_count", 0)

    return ModelTopology(
        arch_name=arch,
        hidden_dim=int(hidden_dim),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        num_layers=int(num_layers),
        vocab_size=int(vocab_size),
        context_length=int(context_length),
        head_dim=int(head_dim),
        ffn_dim=int(ffn_dim),
        num_experts=int(num_experts),
    )


def parse_safetensors_config(config_path: str) -> ModelTopology:
    """Parse Hugging Face style config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Handle text_config sub-dictionary if present
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        base_cfg = cfg["text_config"]
    else:
        base_cfg = cfg

    arch = base_cfg.get("model_type", "llama")
    hidden_dim = base_cfg.get("hidden_size", 2048)
    num_heads = base_cfg.get("num_attention_heads", 32)
    num_kv_heads = base_cfg.get("num_key_value_heads", num_heads)
    num_layers = base_cfg.get("num_hidden_layers", 16)
    ffn_dim = base_cfg.get("intermediate_size", hidden_dim * 4)
    context_length = base_cfg.get("max_position_embeddings", 8192)
    vocab_size = base_cfg.get("vocab_size", 128000)
    head_dim = base_cfg.get("head_dim", hidden_dim // max(num_heads, 1))
    num_experts = base_cfg.get("num_experts", base_cfg.get("n_routed_experts", 0))

    return ModelTopology(
        arch_name=arch,
        hidden_dim=int(hidden_dim),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        num_layers=int(num_layers),
        vocab_size=int(vocab_size),
        context_length=int(context_length),
        head_dim=int(head_dim),
        ffn_dim=int(ffn_dim),
        num_experts=int(num_experts),
    )


def extract_topology(path: str) -> ModelTopology:
    p = Path(path)
    if p.is_dir():
        cfg_file = p / "config.json"
        if cfg_file.is_file():
            return parse_safetensors_config(str(cfg_file))
        raise FileNotFoundError(f"config.json not found in model directory: {path}")

    if p.suffix.lower() == ".json":
        return parse_safetensors_config(str(p))
    elif p.suffix.lower() == ".gguf":
        return parse_gguf_topology(str(p))
    else:
        raise ValueError(f"Unsupported model file: {path} (expected .gguf, .json, or model directory)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: model_topology.py <model.gguf | config.json | model_dir>")
        sys.exit(1)
    topo = extract_topology(sys.argv[1])
    print(json.dumps(asdict(topo), indent=2))
