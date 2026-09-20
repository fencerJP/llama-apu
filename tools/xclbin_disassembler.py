#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
XCLBIN Disassembler & Reverse-Engineering Analysis Suite for AMD Ryzen AI NPU.

Deconstructs, disassembles, and reverse-engineers AMD XDNA hardware microcode
binaries (.xclbin), extracting spatial tile configurations, memory crossbars,
kernel interfaces, PDI boot headers, and CDO command streams into a structured
technical Markdown dossier (<xclbin_name>.md).

Usage:
    python3 xclbin_disassembler.py <path_to_file.xclbin> [--verbose] [-o <output_dir>]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


def log_verbose(msg: str, verbose: bool):
    if verbose:
        print(f"[VERBOSE] {msg}")


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_strings(data: bytes, min_len: int = 4) -> list:
    pattern = rb"[\x20-\x7e]{" + str(min_len).encode() + rb",}"
    found = re.findall(pattern, data)
    return [s.decode("ascii", errors="ignore") for s in found]


def parse_pdi_stream(pdi_path: Path, verbose: bool) -> dict:
    """Performs deep binary analysis of the AMD Versal / Ryzen AI PDI image."""
    log_verbose(f"Analyzing PDI binary: {pdi_path}", verbose)
    if not pdi_path.is_file():
        return {"error": f"PDI file not found: {pdi_path}"}

    with open(pdi_path, "rb") as f:
        pdi_bytes = f.read()

    pdi_size = len(pdi_bytes)
    result = {
        "size_bytes": pdi_size,
        "sha256": hashlib.sha256(pdi_bytes).hexdigest(),
        "has_sync_word": False,
        "identification": "Unknown",
        "cdo_found": False,
        "cdo_offset": None,
        "cdo_version": None,
        "cdo_word_length": None,
        "cdo_checksum": None,
        "cdo_commands": {},
        "aie_registers_configured": [],
        "dma_transfers": [],
        "embedded_elfs": [],
        "recovered_strings": [],
    }

    # 1. Inspect Boot Header sync pattern: 0x11223344 0x55667788 0x99aabbcc
    sync_marker = b"\x44\x33\x22\x11\x88\x77\x66\x55\xcc\xbb\xaa\x99"
    if sync_marker in pdi_bytes[:128]:
        result["has_sync_word"] = True

    # Check for PPDI / aie_image header
    if b"PPDI" in pdi_bytes[:128]:
        result["identification"] = "Versal PDI (PPDI Header)"
    elif b"aie_image" in pdi_bytes[:256]:
        result["identification"] = "AMD AIE Spatial Partition Image (aie_image)"

    # Search for embedded ELF binaries (0x7F 'E' 'L' 'F')
    elf_offset = 0
    while True:
        idx = pdi_bytes.find(b"\x7fELF", elf_offset)
        if idx == -1:
            break
        result["embedded_elfs"].append(idx)
        elf_offset = idx + 4

    # 2. Inspect CDO (Configuration Data Object) Stream
    cdo_idx = pdi_bytes.find(b"CDO\x00")
    if cdo_idx != -1:
        result["cdo_found"] = True
        result["cdo_offset"] = cdo_idx
        cdo_hdr = pdi_bytes[cdo_idx : cdo_idx + 16]
        if len(cdo_hdr) >= 16:
            magic, version, word_len, checksum = struct.unpack("<4sIII", cdo_hdr)
            result["cdo_version"] = f"0x{version:04x}"
            result["cdo_word_length"] = word_len
            result["cdo_checksum"] = f"0x{checksum:08x}"

            # Parse CDO commands
            cdo_data = pdi_bytes[cdo_idx:]
            offset = 16
            max_offset = min(len(cdo_data), word_len * 4)

            cmd_map = {
                0x00: "NOP",
                0x01: "WRITE",
                0x02: "MASK_WRITE",
                0x03: "MASK_POLL",
                0x04: "DMA_WRITE",
                0x05: "DMA_XFER",
                0x11: "AIE_STREAM_CONFIG",
                0x62: "AIE_LOCK_INIT",
            }

            while offset + 4 <= max_offset:
                hdr_word = struct.unpack("<I", cdo_data[offset : offset + 4])[0]
                cmd_id = hdr_word & 0xFF
                api_id = (hdr_word >> 8) & 0xFF
                cmd_payload_words = (hdr_word >> 16) & 0xFFFF

                cmd_name = cmd_map.get(cmd_id, f"CMD_0x{api_id:02x}_{cmd_id:02x}")
                result["cdo_commands"][cmd_name] = result["cdo_commands"].get(cmd_name, 0) + 1

                # If write or mask_write targeting AIE register space
                if cmd_name in ("WRITE", "MASK_WRITE") and offset + 8 <= max_offset:
                    target_addr = struct.unpack("<I", cdo_data[offset + 4 : offset + 8])[0]
                    # AMD AIE2 base address registers usually reside in 0x00200000 - 0x002FFFFF
                    if 0x00200000 <= target_addr <= 0x002FFFFF:
                        col = (target_addr >> 18) & 0x7F
                        reg = target_addr & 0x3FFFF
                        result["aie_registers_configured"].append({
                            "address": f"0x{target_addr:08x}",
                            "column": col,
                            "reg_offset": f"0x{reg:05x}",
                            "op": cmd_name,
                        })

                if cmd_name == "DMA_XFER" and offset + 12 <= max_offset:
                    dst_addr = struct.unpack("<I", cdo_data[offset + 4 : offset + 8])[0]
                    xfer_len = struct.unpack("<I", cdo_data[offset + 8 : offset + 12])[0]
                    result["dma_transfers"].append({
                        "dst_address": f"0x{dst_addr:08x}",
                        "length_bytes": xfer_len,
                    })

                step = (cmd_payload_words + 1) * 4
                if step == 0:
                    break
                offset += step

    # Extract printable strings
    all_strings = extract_strings(pdi_bytes, min_len=4)
    # Filter for meaningful strings
    filtered_strings = [
        s for s in all_strings
        if any(keyword in s.lower() for keyword in ["aie", "dpu", "xilinx", "versal", "kernel", "mlir", "layer", "qwen", "llama"])
    ]
    result["recovered_strings"] = filtered_strings[:20]

    return result


def disassemble_xclbin(xclbin_path: Path, output_dir: Path, verbose: bool) -> Path:
    """Disassembles an XCLBIN into its constituent sections and compiles a Markdown dossier."""
    xclbin_path = xclbin_path.resolve()
    if not xclbin_path.is_file():
        raise FileNotFoundError(f"Target XCLBIN does not exist: {xclbin_path}")

    stem_name = xclbin_path.stem
    output_md_path = output_dir.resolve() / f"{stem_name}.md"
    artifacts_dir = output_dir.resolve() / f"{stem_name}_extracted"
    os.makedirs(artifacts_dir, exist_ok=True)

    print("================================================================================")
    print(f"  AMD XDNA XCLBIN Reverse-Engineering Suite")
    print(f"  Target File:       {xclbin_path.name}")
    print(f"  Report Output:     {output_md_path}")
    print(f"  Extracted Assets:  {artifacts_dir}")
    print("================================================================================")

    log_verbose(f"Computing SHA-256 and querying xclbinutil...", verbose)
    file_size = xclbin_path.stat().st_size
    file_sha256 = sha256_file(xclbin_path)

    # 1. Execute xclbinutil --info
    info_proc = subprocess.run(
        ["xclbinutil", "--info", "--input", str(xclbin_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    raw_info = info_proc.stdout

    # Parse header fields from info
    uuid_match = re.search(r"UUID \(xclbin\):\s+([0-9a-fA-F-]+)", raw_info)
    xclbin_uuid = uuid_match.group(1) if uuid_match else "Unknown"

    version_match = re.search(r"Version:\s+([0-9.]+)", raw_info)
    xclbin_version = version_match.group(1) if version_match else "Unknown"

    sections_match = re.search(r"Sections:\s+([^\n]+)", raw_info)
    sections_list = [s.strip() for s in sections_match.group(1).split(",")] if sections_match else []

    log_verbose(f"Discovered UUID: {xclbin_uuid}", verbose)
    log_verbose(f"Discovered Sections: {', '.join(sections_list)}", verbose)

    # 2. Extract sections
    section_files = {
        "MEM_TOPOLOGY": artifacts_dir / "mem_topology.json",
        "IP_LAYOUT": artifacts_dir / "ip_layout.json",
        "CONNECTIVITY": artifacts_dir / "connectivity.json",
        "EMBEDDED_METADATA": artifacts_dir / "embedded_metadata.xml",
        "AIE_PARTITION": artifacts_dir / "aie_partition.json",
    }

    dump_cmd = ["xclbinutil", "--input", str(xclbin_path)]
    for sec_name, sec_path in section_files.items():
        fmt = "JSON" if sec_path.suffix == ".json" else "RAW"
        dump_cmd.extend(["--dump-section", f"{sec_name}:{fmt}:{sec_path.name}"])

    log_verbose(f"Running dump command in {artifacts_dir}...", verbose)
    dump_proc = subprocess.run(dump_cmd, cwd=artifacts_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if dump_proc.returncode != 0:
        log_verbose(f"Warning: xclbinutil returned non-zero during dump: {dump_proc.stderr}", verbose)

    # 3. Parse JSON / XML sections
    mem_topology_data = {}
    if section_files["MEM_TOPOLOGY"].is_file():
        try:
            with open(section_files["MEM_TOPOLOGY"]) as f:
                mem_topology_data = json.load(f).get("mem_topology", {})
        except Exception as e:
            log_verbose(f"Error parsing mem_topology: {e}", verbose)

    ip_layout_data = {}
    if section_files["IP_LAYOUT"].is_file():
        try:
            with open(section_files["IP_LAYOUT"]) as f:
                ip_layout_data = json.load(f).get("ip_layout", {})
        except Exception as e:
            log_verbose(f"Error parsing ip_layout: {e}", verbose)

    connectivity_data = {}
    if section_files["CONNECTIVITY"].is_file():
        try:
            with open(section_files["CONNECTIVITY"]) as f:
                connectivity_data = json.load(f).get("connectivity", {})
        except Exception as e:
            log_verbose(f"Error parsing connectivity: {e}", verbose)

    aie_partition_data = {}
    pdi_filename = None
    if section_files["AIE_PARTITION"].is_file():
        try:
            with open(section_files["AIE_PARTITION"]) as f:
                aie_partition_data = json.load(f).get("aie_partition", {})
                pdis = aie_partition_data.get("PDIs", [])
                if pdis and "file_name" in pdis[0]:
                    pdi_filename = pdis[0]["file_name"]
        except Exception as e:
            log_verbose(f"Error parsing aie_partition: {e}", verbose)

    embedded_metadata_parsed = {}
    if section_files["EMBEDDED_METADATA"].is_file():
        try:
            tree = ET.parse(section_files["EMBEDDED_METADATA"])
            root = tree.getroot()
            kernel = root.find(".//kernel")
            if kernel is not None:
                ext_data = kernel.find("extended-data")
                args = []
                for arg in kernel.findall("arg"):
                    args.append(arg.attrib)
                embedded_metadata_parsed = {
                    "kernel_name": kernel.attrib.get("name", "Unknown"),
                    "kernel_type": kernel.attrib.get("type", "Unknown"),
                    "extended_data": ext_data.attrib if ext_data is not None else {},
                    "arguments": args,
                }
        except Exception as e:
            log_verbose(f"Error parsing embedded_metadata: {e}", verbose)

    # 4. Check for and analyze PDI
    pdi_path = None
    if pdi_filename:
        cand = artifacts_dir / pdi_filename
        if cand.is_file():
            pdi_path = cand

    if not pdi_path:
        for f in artifacts_dir.glob("*.pdi"):
            pdi_path = f
            break

    pdi_analysis = parse_pdi_stream(pdi_path, verbose) if pdi_path else {}

    # Determine hardware architecture family
    col_width_str = aie_partition_data.get("partition", {}).get("column_width", "0")
    col_width = int(col_width_str) if col_width_str.isdigit() else 0

    if col_width == 5:
        target_arch = "AMD XDNA 1 / AIE2 (20-tile spatial array)"
        silicon_family = "Phoenix (Ryzen 7040 series) & Hawk Point (Ryzen 8040 series)"
        arch_id = "NPU1"
    elif col_width == 8:
        target_arch = "AMD XDNA 2 / AIE2P (32-tile spatial array)"
        silicon_family = "Strix Point (Ryzen AI 300 / HX 370), Krackan Point, Gorgon Point, & Strix Halo"
        arch_id = "NPU2"
    else:
        target_arch = f"Generic / Custom ({col_width} columns)"
        silicon_family = "Unknown / Heterogeneous APU"
        arch_id = "CUSTOM"

    log_verbose(f"Detected Target Architecture: {arch_id} ({target_arch})", verbose)

    # 5. Build Markdown Dossier
    print(f">> Generating structured engineering dossier: {output_md_path}...")
    with open(output_md_path, "w", encoding="utf-8") as md:
        md.write(f"# Reverse-Engineering Dossier: `{xclbin_path.name}`\n\n")
        md.write(f"**Analyzed on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        md.write(f"**Binary Source**: `{xclbin_path}`  \n")
        md.write(f"**SHA-256 Digest**: `{file_sha256}`  \n")
        md.write(f"**Container Size**: {file_size:,} bytes ({file_size / 1024:.2f} KB)  \n\n")

        md.write("---\n\n")
        md.write("## 1. Executive Hardware Architecture Summary\n\n")
        md.write(f"- **Target Generation**: **{arch_id}** (`{target_arch}`)\n")
        md.write(f"- **Supported Silicon Processors**: {silicon_family}\n")
        md.write(f"- **XRT Container UUID**: `{xclbin_uuid}`\n")
        md.write(f"- **Container Version**: `{xclbin_version}`\n")
        md.write(f"- **Spatial Column Width**: **{col_width} columns** ({col_width * 4} active spatial compute tiles)\n")
        ops_per_cycle = aie_partition_data.get("operations_per_cycle", "N/A")
        md.write(f"- **Operations per Cycle**: `{ops_per_cycle}`\n\n")

        md.write("---\n\n")
        md.write("## 2. Container Sections Inventory\n\n")
        md.write("| Section Name | Format / Type | Extracted Artifact Path |\n")
        md.write("| :--- | :--- | :--- |\n")
        for sec in sections_list:
            art_file = section_files.get(sec)
            art_desc = f"`{art_file.name}`" if art_file and art_file.is_file() else "*Embedded in PDI / In-Memory*"
            md.write(f"| **`{sec}`** | Binary / XRT Header | {art_desc} |\n")
        md.write("\n")

        md.write("---\n\n")
        md.write("## 3. Physical Memory Configuration (`MEM_TOPOLOGY`)\n\n")
        mem_data = mem_topology_data.get("m_mem_data", [])
        if mem_data:
            md.write("| Bank Index | Memory Tag | Type | Base Address | Size (KB) | Size (MB) | Bank Used |\n")
            md.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for idx, b in enumerate(mem_data):
                tag = b.get("m_tag", "UNKNOWN")
                m_type = b.get("m_type", "MEM_DRAM")
                base = b.get("m_base_address", "0x0")
                sz_kb_hex = b.get("m_sizeKB", "0x0")
                sz_kb = int(sz_kb_hex, 16) if sz_kb_hex.startswith("0x") else int(sz_kb_hex)
                sz_mb = sz_kb / 1024.0
                used = "YES" if b.get("m_used") == "1" else "NO"
                md.write(f"| **{idx}** | `{tag}` | `{m_type}` | `{base}` | `{sz_kb_hex}` ({sz_kb} KB) | {sz_mb:.1f} MB | {used} |\n")
            md.write("\n")
        else:
            md.write("*No `MEM_TOPOLOGY` bank descriptors found.*  \n\n")

        md.write("---\n\n")
        md.write("## 4. DPU Kernel Layout & Execution Interface (`IP_LAYOUT` & `EMBEDDED_METADATA`)\n\n")
        ips = ip_layout_data.get("m_ip_data", [])
        if ips:
            md.write("### IP Layout Instances\n\n")
            md.write("| Instance Name | Kernel ID | Subtype | Base Address |\n")
            md.write("| :--- | :--- | :--- | :--- |\n")
            for ip in ips:
                md.write(f"| `{ip.get('m_name')}` | `{ip.get('m_kernel_id')}` | `{ip.get('m_subtype')}` | `{ip.get('m_base_address')}` |\n")
            md.write("\n")

        if embedded_metadata_parsed:
            md.write("### Kernel Function Signature & Arguments\n\n")
            md.write(f"- **Kernel Name**: `{embedded_metadata_parsed.get('kernel_name')}`\n")
            md.write(f"- **Type**: `{embedded_metadata_parsed.get('kernel_type')}`\n")
            ext = embedded_metadata_parsed.get("extended_data", {})
            if ext:
                md.write(f"- **Architecture Tag**: `{ext.get('arch', 'N/A')}`\n")
                md.write(f"- **Model Dimensions**: Hidden Dim=`{ext.get('hidden_dim', 'N/A')}`, Heads=`{ext.get('num_heads', 'N/A')}`, KV Heads=`{ext.get('num_kv_heads', 'N/A')}`, Layers=`{ext.get('layers', 'N/A')}`\n")

            md.write("\n| Argument ID | Name | Type | Qualifier | Size | Host Offset |\n")
            md.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for a in embedded_metadata_parsed.get("arguments", []):
                q = "Value (Scalar)" if a.get("addressQualifier") == "0" else "Pointer (Buffer Object)"
                md.write(f"| `{a.get('id')}` | `{a.get('name')}` | `{a.get('type')}` | {q} | `{a.get('size')}` | `{a.get('hostOffset')}` |\n")
            md.write("\n")

        md.write("---\n\n")
        md.write("## 5. Crossbar Memory Connectivity (`CONNECTIVITY`)\n\n")
        conn = connectivity_data.get("m_connection", [])
        if conn:
            md.write("Routes kernel buffer arguments directly to specific physical memory banks:\n\n")
            md.write("| Kernel Argument Index | IP Layout Index | Target Memory Bank Index | Memory Bank Tag |\n")
            md.write("| :--- | :--- | :--- | :--- |\n")
            for c in conn:
                arg_idx = c.get("arg_index")
                mem_idx = int(c.get("mem_data_index", "0"))
                bank_tag = mem_data[mem_idx].get("m_tag", "UNKNOWN") if mem_idx < len(mem_data) else "UNKNOWN"
                md.write(f"| Argument `bo{int(arg_idx)-3}` (arg `{arg_idx}`) | `{c.get('m_ip_layout_index')}` | Bank `{mem_idx}` | **`{bank_tag}`** |\n")
            md.write("\n")

            md.write("```mermaid\ngraph LR\n")
            md.write('    subgraph "Kernel Arguments"\n')
            for c in conn:
                arg_idx = c.get("arg_index")
                md.write(f'        Arg{arg_idx}["Argument {arg_idx}"]\n')
            md.write("    end\n")
            md.write('    subgraph "Physical Memory Banks"\n')
            for idx, b in enumerate(mem_data):
                md.write(f'        Bank{idx}["Bank {idx}: {b.get("m_tag")} ({b.get("m_type")})"]\n')
            md.write("    end\n")
            for c in conn:
                arg_idx = c.get("arg_index")
                mem_idx = c.get("mem_data_index")
                md.write(f"    Arg{arg_idx} --> Bank{mem_idx}\n")
            md.write("```\n\n")

        md.write("---\n\n")
        md.write("## 6. Spatial AIE Partition Geometry (`AIE_PARTITION`)\n\n")
        md.write(f"- **Partition Name**: `{aie_partition_data.get('name', 'N/A')}`\n")
        md.write(f"- **Active Column Width**: **{col_width}**\n")
        start_cols = aie_partition_data.get("partition", {}).get("start_columns", [])
        md.write(f"- **Start Columns**: `{start_cols}`\n")
        md.write(f"- **Inference Fingerprint**: `{aie_partition_data.get('inference_fingerprint', 'N/A')}`\n")
        md.write(f"- **Pre/Post Fingerprint**: `{aie_partition_data.get('pre_post_fingerprint', 'N/A')}`\n\n")

        pdis = aie_partition_data.get("PDIs", [])
        if pdis:
            md.write("### Embedded Hardware PDI Reference\n\n")
            for p in pdis:
                md.write(f"- **PDI UUID**: `{p.get('uuid')}`\n")
                md.write(f"- **Target PDI File**: `{p.get('file_name')}`\n")
                for cdo in p.get("cdo_groups", []):
                    md.write(f"  - **CDO Group**: `{cdo.get('name')}` (Type: `{cdo.get('type')}`, PDI ID: `{cdo.get('pdi_id')}`, Kernel IDs: `{cdo.get('dpu_kernel_ids')}`)\n")
            md.write("\n")

        md.write("---\n\n")
        md.write("## 7. Deep PDI & CDO Microcode Disassembly\n\n")
        if pdi_analysis and "error" not in pdi_analysis:
            md.write(f"- **PDI Binary Size**: {pdi_analysis.get('size_bytes', 0):,} bytes\n")
            md.write(f"- **BootROM Sync Header**: {'VALID (0x11223344 ...)' if pdi_analysis.get('has_sync_word') else 'NOT PRESENT'}\n")
            md.write(f"- **PDI Identification**: `{pdi_analysis.get('identification')}`\n")
            md.write(f"- **CDO Stream Offset**: `0x{pdi_analysis.get('cdo_offset', 0):x}`\n")
            md.write(f"- **CDO Header Version**: `{pdi_analysis.get('cdo_version')}`\n")
            md.write(f"- **CDO Command Words**: `{pdi_analysis.get('cdo_word_length'):,} words` ({pdi_analysis.get('cdo_word_length', 0) * 4:,} bytes)\n\n")

            cmd_breakdown = pdi_analysis.get("cdo_commands", {})
            if cmd_breakdown:
                md.write("### CDO Hardware Instruction Breakdown\n\n")
                md.write("| Command Opcode / ID | Count | Purpose in Spatial Interconnect |\n")
                md.write("| :--- | :--- | :--- |\n")
                descriptions = {
                    "WRITE": "Direct register / memory write to tile registers",
                    "MASK_WRITE": "Atomic bitmasked register configuration (locks, stream switches)",
                    "MASK_POLL": "Hardware synchronization polling (barrier wait)",
                    "DMA_XFER": "Tile DMA channel transfer descriptor initialization",
                    "DMA_WRITE": "Direct bulk data transfer into local 64 KB tile SRAM",
                    "AIE_STREAM_CONFIG": "Inter-tile stream switch crossbar circuit configuration",
                    "AIE_LOCK_INIT": "Mutual exclusion lock / semaphore initialization",
                    "NOP": "Instruction alignment padding",
                }
                for cmd, cnt in sorted(cmd_breakdown.items(), key=lambda x: x[1], reverse=True):
                    desc = descriptions.get(cmd, "Custom / Proprietary AIE microcode command")
                    md.write(f"| **`{cmd}`** | {cnt:,} | {desc} |\n")
                md.write("\n")

            regs = pdi_analysis.get("aie_registers_configured", [])
            if regs:
                md.write("### AI Engine Tile Registers Configured (Sample)\n\n")
                md.write("| Target Address | Tile Column | Register Offset | Operation |\n")
                md.write("| :--- | :--- | :--- | :--- |\n")
                for r in regs[:10]:
                    md.write(f"| `{r['address']}` | Col {r['column']} | `{r['reg_offset']}` | `{r['op']}` |\n")
                if len(regs) > 10:
                    md.write(f"| ... | ... | ... | *(Total {len(regs)} registers configured)* |\n")
                md.write("\n")

            strings = pdi_analysis.get("recovered_strings", [])
            if strings:
                md.write("### Recovered Symbol & Internal Strings\n\n")
                for s in strings:
                    md.write(f"- `{s}`\n")
                md.write("\n")
        else:
            md.write(f"*PDI analysis unavailable: {pdi_analysis.get('error', 'No PDI extracted')}*  \n\n")

        md.write("---\n\n")
        md.write("## 8. Cross-APU Binary Portability Assessment\n\n")
        if arch_id == "NPU2":
            md.write("### Compatibility Status: **AMD XDNA 2 (AIE2P)**\n")
            md.write("- [x] **Strix Point (Ryzen AI 9 HX 370 / 365)**: Native execution (32 spatial tiles active).\n")
            md.write("- [x] **Krackan Point / Gorgon Point**: Native execution (identical tile architecture).\n")
            md.write("- [x] **Strix Halo (Ryzen AI Max+ 395)**: Compatible in 32-tile spatial compatibility mode.\n")
            md.write("- [ ] **Phoenix / Hawk Point (Ryzen 7040 / 8040)**: **INCOMPATIBLE** (Hardware mismatch: NPU1 uses 20 tiles / 5 columns and AIE2 VLIW ISA).\n")
        elif arch_id == "NPU1":
            md.write("### Compatibility Status: **AMD XDNA 1 (AIE2)**\n")
            md.write("- [x] **Phoenix (Ryzen 7040)**: Native execution (20 spatial tiles active).\n")
            md.write("- [x] **Hawk Point (Ryzen 8040)**: Native execution.\n")
            md.write("- [ ] **Strix Point / Gorgon / Halo**: **INCOMPATIBLE** (Hardware mismatch: NPU2 expects 32 tiles / 8 columns and AIE2P VLIW ISA).\n")
        md.write("\n")

        md.write("---\n\n")
        md.write("## 9. Zero-Copy Model Runner Deployment Checklist\n\n")
        md.write("1. **Embedding into `.q4nx` Header**:\n")
        md.write(f"   ```bash\n   ./build/bin/apu-run -m <model>.gguf -x {xclbin_path} --embed-xclbin\n   ```\n")
        md.write("2. **Shared KV-Cache DMA-BUF Sizing**:\n")
        md.write(f"   - Minimum recommended DMA-BUF allocation: **64 MB** (64-byte aligned).\n")
        md.write("3. **Direct PDI Microcode Extraction**:\n")
        if pdi_path and pdi_path.is_file():
            md.write(f"   - PDI extracted to: `{pdi_path}` ({pdi_path.stat().st_size:,} bytes).\n")
            md.write(f"   - Can be used directly as a hardware template in `XclbinBuilder::new(TargetHardware::{'Npu2Aie2p' if arch_id == 'NPU2' else 'Npu1Aie2'}, ...)`.\n")

    print(f">> Reverse-engineering analysis complete! Dossier generated at: {output_md_path}")
    return output_md_path


def main():
    parser = argparse.ArgumentParser(
        description="Deconstruct and reverse-engineer AMD XDNA XCLBIN microcode binaries into structured Markdown dossiers."
    )
    parser.add_argument("xclbin", type=str, help="Path to target .xclbin file")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory for generated .md report (defaults to directory of input xclbin)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose step-by-step diagnostic telemetry")

    args = parser.parse_args()
    xclbin_path = Path(args.xclbin)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = xclbin_path.parent

    try:
        disassemble_xclbin(xclbin_path, output_dir, args.verbose)
    except Exception as e:
        print(f"\n[Error] Failed to reverse-engineer {xclbin_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
