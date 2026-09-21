#!/usr/bin/env python3
import os
import re
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("/home/fencer/.openclaw/workspace/projects/zero-copy_model_runner/tools/reverse_engineering/qwen35_analysis")

MODELS = [
    "Qwen3.5-0.8B-NPU2",
    "Qwen3.5-2B-NPU2",
    "Qwen3.5-4B-NPU2",
    "Qwen3.5-9B-NPU2",
    "Qwen3.6-35B-A3B-NPU2"
]

KERNELS = ["layer", "GateDeltaNet_prefill", "conv", "attn", "mm"]

def parse_kernel_dossier(md_path):
    if not md_path.exists():
        return None
    text = md_path.read_text()
    
    info = {}
    m_sz = re.search(r"Container Size\*\*:\s*([0-9,]+ bytes)", text)
    if m_sz: info["size"] = m_sz.group(1)
    
    m_pdi = re.search(r"PDI Binary Size\*\*:\s*([0-9,]+ bytes)", text)
    if m_pdi: info["pdi_size"] = m_pdi.group(1)
    
    m_cdo_words = re.search(r"CDO Command Words\*\*:\s*`([^`]+)`", text)
    if m_cdo_words: info["cdo_words"] = m_cdo_words.group(1)
    
    # Extract CDO table
    opcodes = {}
    cdo_section = text.find("### CDO Hardware Instruction Breakdown")
    if cdo_section != -1:
        cdo_end = text.find("###", cdo_section + 35)
        if cdo_end == -1: cdo_end = text.find("##", cdo_section + 35)
        cdo_chunk = text[cdo_section:cdo_end]
        for line in cdo_chunk.splitlines():
            m_op = re.search(r"\|\s*\*\*`([^`]+)`\*\*\s*\|\s*([0-9,]+)\s*\|", line)
            if m_op:
                opcodes[m_op.group(1)] = int(m_op.group(2).replace(",", ""))
    info["opcodes"] = opcodes
    return info

print("==========================================================================================")
print("              QWEN 3.5 & 3.6 HARDWARE KERNEL OPCODE & ARCHITECTURE MATRIX                 ")
print("==========================================================================================")

for k in KERNELS:
    print(f"\n------------------------------------------------------------------------------------------")
    print(f"KERNEL: {k}.xclbin")
    print(f"------------------------------------------------------------------------------------------")
    print(f"{'Model':<22} | {'Container':<12} | {'PDI Size':<12} | {'CDO Words':<12} | {'Top Opcodes'}")
    print("-" * 90)
    for m in MODELS:
        md_file = OUT_DIR / m / f"{k}.md"
        data = parse_kernel_dossier(md_file)
        if not data:
            print(f"{m:<22} | {'MISSING':<12} | {'-':<12} | {'-':<12} | -")
            continue
        ops_str = ", ".join([f"{op}:{cnt}" for op, cnt in list(data["opcodes"].items())[:4]])
        print(f"{m:<22} | {data['size']:<12} | {data['pdi_size']:<12} | {data['cdo_words']:<12} | {ops_str}")

