#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path("/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/benchmark_suite")
MODELS = ["deepseek-r1-qwen3-8b", "qwen2.5-3b", "llama-3.2-3b", "qwen3.5-0.8b", "gemma4"]

def parse_dossier(path):
    if not path.exists(): return {}
    text = path.read_text()
    info = {}
    m_sz = re.search(r"Container Size\*\*:\s*([0-9,]+ bytes)", text)
    if m_sz: info["size"] = m_sz.group(1)
    m_uuid = re.search(r"XRT Container UUID\*\*:\s*`([^`]+)`", text)
    if m_uuid: info["uuid"] = m_uuid.group(1)
    m_cols = re.search(r"Spatial Column Width\*\*:\s*\*\*(\d+) columns\*\*", text)
    if m_cols: info["cols"] = m_cols.group(1)
    m_pdi = re.search(r"PDI Binary Size\*\*:\s*([0-9,]+ bytes)", text)
    if m_pdi: info["pdi_size"] = m_pdi.group(1)
    m_cdo = re.search(r"CDO Command Words\*\*:\s*`([^`]+)`", text)
    if m_cdo: info["cdo_words"] = m_cdo.group(1)
    m_meta = re.search(r"Model Dimensions\*\*:\s*([^\n]+)", text)
    if m_meta: info["meta"] = m_meta.group(1)
    m_arch = re.search(r"Architecture Tag\*\*:\s*`([^`]+)`", text)
    if m_arch: info["arch_tag"] = m_arch.group(1)
    m_sram = re.search(r"SRAM.*?(0x[0-9a-fA-F]+ \([0-9]+ KB\))\s*\|\s*([0-9.]+ MB)", text)
    if m_sram: info["sram_size"] = f"{m_sram.group(1)} ({m_sram.group(2)})"
    return info

print("| Model | Architectural Dimension | Built-in / Provided XCLBIN | Custom Synthesized XCLBIN |")
print("| :--- | :--- | :--- | :--- |")
for m in MODELS:
    orig = parse_dossier(ROOT / m / "layer.md")
    gen = parse_dossier(ROOT / m / "generated-layer.md")
    print(f"| **{m}** | Container Size | {orig.get('size','N/A')} | {gen.get('size','N/A')} |")
    print(f"| | SRAM Buffer Size | {orig.get('sram_size','N/A')} | {gen.get('sram_size','N/A')} |")
    print(f"| | Extended Metadata | {orig.get('arch_tag','N/A')}: {orig.get('meta','N/A')} | {gen.get('arch_tag','N/A')}: {gen.get('meta','N/A')} |")
    print(f"| | Spatial Column Width | {orig.get('cols','N/A')} columns (32 tiles) | {gen.get('cols','N/A')} columns (32 tiles) |")
    print(f"| | PDI Binary Size | {orig.get('pdi_size','N/A')} | {gen.get('pdi_size','N/A')} |")
    print(f"| | CDO Stream Length | {orig.get('cdo_words','N/A')} | {gen.get('cdo_words','N/A')} |")
