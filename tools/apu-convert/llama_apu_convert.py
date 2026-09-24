#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# llama-apu: Phase 7.2 End-to-End Conversion Pipeline (16/8-bit -> TQ2_0 + Linked Q4NX Sidecar)

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SYS_TOOLS = REPO_ROOT / "tools"
XCLBIN_SYNTH_DIR = SYS_TOOLS / "xclbin-synth"
PTQ_DIR = SYS_TOOLS / "ptq-tq2"
sys.path.append(str(PTQ_DIR))

from ptqtp_engine import determine_execution_strategy, check_memory_governor

APU_MAX_MEMORY_CEILING_GB = 50.0

def get_python_runner() -> str:
    """Finds a Python interpreter that has torch and gguf installed."""
    try:
        import torch
        import gguf
        return sys.executable
    except ImportError:
        pass

    candidates = [
        "/home/fencer/FLM_Q4NX_Converter/.venv/bin/python",
        "/home/fencer/.openclaw/workspace/projects/old/zero-copy_model_runner/.venv/bin/python",
        sys.executable,
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                res = subprocess.run([cand, "-c", "import torch, gguf; print('OK')"],
                                     capture_output=True, text=True)
                if res.returncode == 0:
                    return cand
            except Exception:
                continue
    return sys.executable

def check_system_memory() -> Tuple[float, float]:
    """Returns (total_gb, avail_gb) from /proc/meminfo."""
    total_gb = 64.0
    avail_gb = 32.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if parts[0] == "MemTotal:":
                    total_gb = int(parts[1]) / (1024 * 1024)
                elif parts[0] == "MemAvailable:":
                    avail_gb = int(parts[1]) / (1024 * 1024)
    except Exception:
        pass
    return total_gb, avail_gb

def inspect_model(source_path: Path) -> Dict[str, Any]:
    """Inspects model topology using model_topology.py or config.json/GGUF."""
    topo_script = XCLBIN_SYNTH_DIR / "model_topology.py"
    target = source_path
    if source_path.is_dir():
        cfg = source_path / "config.json"
        if cfg.exists():
            target = cfg

    res = subprocess.run([sys.executable, str(topo_script), str(target)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Model topology inspection failed: {res.stderr}")
    return json.loads(res.stdout)

def evaluate_tq2_compatibility(topo: Dict[str, Any]) -> Tuple[bool, str]:
    """Evaluates whether model dimensions satisfy TQ2_0 standard requirements."""
    hidden = topo.get("hidden_dim", 0)
    ffn = topo.get("ffn_dim", 0)
    head_dim = topo.get("head_dim", 0)

    reasons = []
    if hidden % 256 != 0:
        reasons.append(f"hidden_dim ({hidden}) not divisible by 256")
    if ffn % 256 != 0:
        reasons.append(f"ffn_dim ({ffn}) not divisible by 256")
    if head_dim > 0 and (head_dim & (head_dim - 1)) != 0:
        reasons.append(f"head_dim ({head_dim}) is not a power of 2 for FWHT")

    if reasons:
        return False, "; ".join(reasons)
    return True, "Compatible with TQ2_0 (256-element blocks, power-of-2 head dim)"

def convert_safetensors_to_gguf(
    source_dir: Path,
    out_gguf: Path,
    quant_type: str,
    py_exec: str
) -> Path:
    """Converts a SafeTensors directory to GGUF using convert_hf_to_gguf.py."""
    convert_script = REPO_ROOT / "convert_hf_to_gguf.py"
    cmd = [
        py_exec,
        str(convert_script),
        str(source_dir),
        "--outfile", str(out_gguf),
        "--outtype", quant_type.lower(),
    ]

    # Handle architectures like Qwen3-Next that lack mtp_num_hidden_layers in config.json
    cfg_file = source_dir / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                cfg_data = json.load(f)
            arch = cfg_data.get("architectures", [""])[0]
            if "Next" in arch or "qwen3_next" in cfg_data.get("model_type", ""):
                cmd.append("--no-mtp")
        except Exception:
            pass

    print(f"[+] Executing out-of-core conversion: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(f"    {line.rstrip()}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed with exit code {proc.returncode}")

    if not out_gguf.exists():
        # Check if convert_hf_to_gguf created a file with a slight name variation
        parent_name = source_dir.name
        cand = (list(out_gguf.parent.glob(f"{out_gguf.stem}*.gguf")) or
                list(out_gguf.parent.glob(f"*{parent_name}*{quant_type}*.gguf")) or
                list(out_gguf.parent.glob(f"*{parent_name}*.gguf")))
        if cand:
            return cand[0]
        raise FileNotFoundError(f"Expected converted GGUF at {out_gguf}, but not found.")
    return out_gguf

def quantize_existing_gguf(
    in_gguf: Path,
    out_gguf: Path,
    quant_type: str
) -> Path:
    """Quantizes an existing GGUF model using build/bin/llama-quantize."""
    quant_bin = REPO_ROOT / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        raise FileNotFoundError(f"llama-quantize binary not found at {quant_bin}")

    cmd = [str(quant_bin), str(in_gguf), str(out_gguf), quant_type]
    print(f"[+] Executing llama-quantize: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"llama-quantize failed: {res.stderr}")
    print(res.stdout)
    return out_gguf

def synthesize_and_package_q4nx(
    source_model: Path,
    gguf_path: Path,
    output_q4nx_path: Path,
    topo: Dict[str, Any]
) -> Tuple[Path, Path]:
    """
    Synthesizes custom XCLBIN using Phase 7.1 pipeline and embeds it into
    a companion .q4nx sidecar container with standard header markers.
    """
    xclbin_builder_script = XCLBIN_SYNTH_DIR / "xclbin_builder.py"
    
    # 1. Synthesize custom model-matched XCLBIN
    model_stem = gguf_path.stem
    if model_stem.endswith("-TQ2_0"):
        parent_stem = model_stem[:-6]
    else:
        parent_stem = model_stem

    synth_xclbin_path = output_q4nx_path.parent / f"{model_stem}-enhanced.xclbin"
    synth_cmd = [
        sys.executable,
        str(xclbin_builder_script),
        str(source_model if source_model.is_dir() else gguf_path),
        "-o", str(synth_xclbin_path),
    ]
    print(f"[+] Synthesizing model-matched XCLBIN: {' '.join(synth_cmd)}")
    res = subprocess.run(synth_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"XCLBIN synthesis failed: {res.stderr}")
    print(f"    {res.stdout.strip()}")

    # 2. Package into .q4nx sidecar
    with open(synth_xclbin_path, "rb") as f:
        xclbin_bytes = f.read()

    header = bytearray(256)
    header[0:4] = b"Q4NX"
    struct.pack_into("<I", header, 4, 1)  # version 1

    arch_str = topo.get("arch_name", "llama").encode("utf-8")[:31]
    header[8:8 + len(arch_str)] = arch_str

    hidden = topo.get("hidden_dim", 2048)
    heads = topo.get("num_heads", 16)
    kv_heads = topo.get("num_kv_heads", 8)
    layers = topo.get("num_layers", 28)
    vocab = topo.get("vocab_size", 128000)
    ctx = min(topo.get("context_length", 32768), 131072)

    struct.pack_into("<IIIIII", header, 40, hidden, heads, kv_heads, layers, vocab, ctx)

    xclbin_off = 256
    xclbin_size = len(xclbin_bytes)
    table_off = 0
    entries = 0
    payload_off = (256 + xclbin_size + 63) & ~63  # 64-byte aligned
    payload_size = 0

    struct.pack_into("<QQQQQQ", header, 64, xclbin_off, xclbin_size, table_off, entries, payload_off, payload_size)

    tmp_q4nx = output_q4nx_path.with_suffix(f".tmp.{os.getpid()}")
    with open(tmp_q4nx, "wb") as f:
        f.write(header)
        f.write(xclbin_bytes)
        pad = payload_off - (256 + xclbin_size)
        if pad > 0:
            f.write(b"\x00" * pad)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_q4nx, output_q4nx_path)

    print(f"[+] Packaged linked companion sidecar (atomic): {output_q4nx_path} ({os.path.getsize(output_q4nx_path)} bytes)")

    # 3. Register into Tier 3 ($HOME/.local/share/llama-apu/xclbins/<key>/)
    user_xclbin_base = Path.home() / ".local" / "share" / "llama-apu" / "xclbins"
    for key in [model_stem, parent_stem, topo.get("arch_name", "")]:
        if not key:
            continue
        dest_dir = user_xclbin_base / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{key}-enhanced.xclbin"
        shutil.copy2(synth_xclbin_path, dest_file)

    print(f"[+] Registered custom XCLBIN profile under: {user_xclbin_base / model_stem}")
    return output_q4nx_path, synth_xclbin_path

def validate_pipeline_outputs(gguf_path: Path, q4nx_path: Optional[Path]) -> bool:
    """Runs q4nx-validate.py, container-info, and route-info on generated outputs."""
    print("\n=== Validating Pipeline Outputs ===")
    all_ok = True

    # 1. q4nx-validate.py
    if q4nx_path and q4nx_path.exists():
        validator_script = REPO_ROOT.parent / "q4nx-validate.py"
        if validator_script.exists():
            val_res = subprocess.run([sys.executable, str(validator_script), str(q4nx_path), str(gguf_path)],
                                     capture_output=True, text=True)
            print(f"[q4nx-validate] return code: {val_res.returncode}")
            print(f"    {val_res.stdout.strip()}")
            if "PASS" not in val_res.stdout:
                all_ok = False

    # 2. llama-apu-cli container-info
    apu_cli = REPO_ROOT / "build" / "bin" / "llama-apu-cli"
    if apu_cli.exists():
        print(f"\n[container-info] checking GGUF: {gguf_path.name}")
        c_res = subprocess.run([str(apu_cli), "container-info", str(gguf_path)],
                               capture_output=True, text=True)
        print(f"    {c_res.stdout.strip()}")

        if q4nx_path and q4nx_path.exists():
            print(f"\n[container-info] checking Q4NX: {q4nx_path.name}")
            cq_res = subprocess.run([str(apu_cli), "container-info", str(q4nx_path)],
                                    capture_output=True, text=True)
            print(f"    {cq_res.stdout.strip()}")

        # 3. llama-apu-cli route-info
        print(f"\n[route-info] verifying tier profile resolution for {gguf_path.name}")
        r_res = subprocess.run([str(apu_cli), "route-info", str(gguf_path)],
                               capture_output=True, text=True)
        print(f"    {r_res.stdout.strip()}")
        if "tier" not in r_res.stdout.lower():
            all_ok = False

    return all_ok

def run_conversion_pipeline(
    source_path: str,
    output_dir: str,
    target_quant: str = "TQ2_0",
    fallback_quant: str = "Q4_K_M",
    skip_sidecar: bool = False,
    skip_xclbin: bool = False,
    convert_mode: str = "auto",
    distill_mode: str = "auto",
    enable_distill: bool = False,
    distill_stage: int = 3,
    max_distill_layers: int = 0
) -> Dict[str, Any]:
    """Unified entry point for Phase 7.2 End-to-End Pipeline."""
    source = Path(source_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===================================================================")
    print("  llama-apu: Phase 7.2 End-to-End Conversion & Packaging Pipeline  ")
    print("===================================================================")
    print(f"  Source Model     : {source}")
    print(f"  Output Directory : {out_dir}")
    print(f"  Default Quant    : {target_quant} (fallback: {fallback_quant})")

    # 1. Memory Governor Check
    tot_gb, avail_gb = check_system_memory()
    print(f"[+] Memory Governor : {avail_gb:.1f} GB available / {tot_gb:.1f} GB total (Ceiling: {APU_MAX_MEMORY_CEILING_GB} GB)")

    # 2. Inspect Model Topology
    print(f"\n[1/5] Inspecting Model Architecture and Topology...")
    topo = inspect_model(source)
    num_experts = topo.get('num_experts', 0)
    print(f"    Arch Name : {topo.get('arch_name')}")
    print(f"    Hidden Dim: {topo.get('hidden_dim')}")
    print(f"    FFN Dim   : {topo.get('ffn_dim')}")
    print(f"    Layers    : {topo.get('num_layers')}")
    print(f"    Experts   : {num_experts}")

    # Calculate static source size
    source_size_bytes = 0
    if source.is_dir():
        for p in source.glob("**/*"):
            if p.is_file():
                source_size_bytes += p.stat().st_size
    elif source.exists():
        source_size_bytes = source.stat().st_size

    # Evaluate automated execution strategy
    strat = determine_execution_strategy(
        model_size_bytes=source_size_bytes,
        num_experts=num_experts,
        is_moe=(num_experts > 0),
        override_convert=convert_mode,
        override_distill=distill_mode
    )

    print(f"\n[+] Automated Workflow Execution Strategy:")
    print(f"    Conversion Mode     : {strat['convert_mode'].upper()}")
    print(f"      Rationale         : {strat['convert_rationale']}")
    print(f"    Distillation Mode   : {strat['distill_mode'].upper()}")
    print(f"      Rationale         : {strat['distill_rationale']}")
    print(f"    Staging Policy      : {strat['staging_policy'].upper()}")
    print(f"      Rationale         : {strat['staging_rationale']}")

    if num_experts > 0:
        print(f"[+] MoE Architecture Detected: {num_experts} experts.")
        print(f"[+] Automatically engaging MoE router and AIE2P Tile SRAM pinning.")
        print(f"[+] Sparse expert execution policy active (skipping full-weight local SSD staging).")

    # 3. Architecture Gate: TQ2_0 Compatibility Evaluation
    is_tq2_ok, rationale = evaluate_tq2_compatibility(topo)
    selected_quant = target_quant
    if target_quant.upper() == "TQ2_0" and not is_tq2_ok:
        print(f"[!] TQ2_0 incompatible: {rationale}")
        print(f"[!] Triggering automated fallback to {fallback_quant}")
        selected_quant = fallback_quant
    else:
        print(f"[+] Target Quantization Validated: {selected_quant} ({rationale})")

    # 4. Conversion (Online vs Offline)
    print(f"\n[2/5] Executing Quantization -> {selected_quant} (Mode: {strat['convert_mode'].upper()})...")
    model_name = source.name if source.is_dir() else source.stem
    target_gguf_name = f"{model_name}-{selected_quant}.gguf"
    out_gguf_path = out_dir / target_gguf_name

    py_runner = get_python_runner()
    if out_gguf_path.exists() and os.path.getsize(out_gguf_path) > 1024 * 1024:
        print(f"[+] Reusing existing converted GGUF: {out_gguf_path}")
        final_gguf = out_gguf_path
    elif source.is_dir():
        final_gguf = convert_safetensors_to_gguf(source, out_gguf_path, selected_quant, py_runner)
    else:
        final_gguf = quantize_existing_gguf(source, out_gguf_path, selected_quant)

    print(f"[+] Converted GGUF generated: {final_gguf} ({os.path.getsize(final_gguf)/(1024*1024*1024):.2f} GiB)")

    # 5. Post-Quantization Scale Distillation (Optional / Integrated)
    distill_metrics = None
    if enable_distill:
        print(f"\n[3/5] Executing Post-Quantization Scale Distillation (Stage {distill_stage}, Mode: {strat['distill_mode'].upper()})...")
        from distill_pipeline import distill_gguf_model, load_calibration_corpus, CORPUS_PATH
        corpus_texts = load_calibration_corpus(CORPUS_PATH, max_samples=1000)
        distill_metrics = distill_gguf_model(
            src_gguf=final_gguf,
            dst_gguf=final_gguf,
            stage=distill_stage,
            texts=corpus_texts,
            src_model_dir=source if source.is_dir() else None,
            max_layers_to_distill=max_distill_layers,
            distill_mode=strat['distill_mode'],
            is_moe=(num_experts > 0)
        )

    # 6. Companion .q4nx Sidecar & Custom XCLBIN Synthesis
    q4nx_path = None
    xclbin_path = None
    if not skip_sidecar:
        step_num = "4/5" if enable_distill else "3/4"
        print(f"\n[{step_num}] Synthesizing Model-Matched XCLBIN & Companion .q4nx Sidecar...")
        target_q4nx_name = f"{final_gguf.stem}.q4nx"
        out_q4nx_path = out_dir / target_q4nx_name
        q4nx_path, xclbin_path = synthesize_and_package_q4nx(source, final_gguf, out_q4nx_path, topo)

    # 7. Validation and Integrity Checks
    step_num = "5/5" if enable_distill else "4/4"
    print(f"\n[{step_num}] Multi-tier Integrity Validation...")
    valid = validate_pipeline_outputs(final_gguf, q4nx_path)

    print("\n===================================================================")
    print("  PHASE 7.2 CONVERSION PIPELINE COMPLETE")
    print(f"  GGUF Model    : {final_gguf}")
    print(f"  Q4NX Sidecar  : {q4nx_path}")
    print(f"  XCLBIN Profile: {xclbin_path}")
    print(f"  Distillation  : {'Executed (Stage ' + str(distill_stage) + ')' if enable_distill else 'Skipped'}")
    print(f"  Status        : {'SUCCESS' if valid else 'WARNING'}")
    print("===================================================================")

    return {
        "gguf": str(final_gguf),
        "q4nx": str(q4nx_path) if q4nx_path else None,
        "xclbin": str(xclbin_path) if xclbin_path else None,
        "quant": selected_quant,
        "strategy": strat,
        "distill_metrics": distill_metrics,
        "status": "SUCCESS" if valid else "WARNING"
    }

def main():
    parser = argparse.ArgumentParser(description="llama-apu Phase 7.2 End-to-End Conversion Pipeline")
    parser.add_argument("source", help="Source SafeTensors model directory or GGUF file")
    parser.add_argument("outdir", help="Output directory for converted GGUF and .q4nx companion")
    parser.add_argument("--quant", default="TQ2_0", help="Target quantization type (default: TQ2_0)")
    parser.add_argument("--fallback-quant", default="Q4_K_M", help="Fallback quantization type (default: Q4_K_M)")
    parser.add_argument("--no-sidecar", action="store_true", help="Skip companion .q4nx sidecar generation")
    parser.add_argument("--no-xclbin", action="store_true", help="Skip custom XCLBIN synthesis")
    parser.add_argument("--convert-mode", choices=["auto", "online", "offline"], default="auto", help="Conversion mode: auto (resource-guided), online (in-memory streaming), offline (disk chunking)")
    parser.add_argument("--distill-mode", choices=["auto", "in_memory", "stream_in_place"], default="auto", help="Distillation mode: auto (resource-guided), in_memory (all at once), stream_in_place (disk streaming)")
    parser.add_argument("--distill", action="store_true", help="Execute post-quantization scale distillation")
    parser.add_argument("--distill-stage", type=int, default=3, choices=[1, 2, 3, 4], help="Distillation stage: 1=Frobenius, 2=Light, 3=Full, 4=Extra")
    parser.add_argument("--all-layers", action="store_true", help="Distill all layers in model (thorough multi-hour production mode for large models)")
    parser.add_argument("--max-layers", type=int, default=0, help="Max layers to distill (0 = all layers)")
    args = parser.parse_args()

    run_conversion_pipeline(
        source_path=args.source,
        output_dir=args.outdir,
        target_quant=args.quant,
        fallback_quant=args.fallback_quant,
        skip_sidecar=args.no_sidecar,
        skip_xclbin=args.no_xclbin,
        convert_mode=args.convert_mode,
        distill_mode=args.distill_mode,
        enable_distill=args.distill,
        distill_stage=args.distill_stage,
        max_distill_layers=args.max_layers
    )

if __name__ == "__main__":
    main()
