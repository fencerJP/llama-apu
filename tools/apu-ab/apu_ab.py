#!/usr/bin/env python3
"""apu_ab — llama-apu Phase 10.6 local A/B profiling harness.

Institutionalises the manual A/B protocol used through Phase 10 (see
docs/phase10-rocm-optimization.md §11): interleaved arms, cooldown, medians,
GPU temp/clock logging, and a significant-vs-optional verdict.

Subcommands
-----------
  bench    Interleaved A/B of two commands; median pp/tg + verdict.
  profile  Run a command under rocprofv3 and save a kernel summary.
  diff     Diff two saved profile summaries.

Examples
--------
  # hipBLASLt prefill ON vs OFF, on the frozen 4B model
  tools/apu-ab/apu_ab.py bench \
    --a "LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib llama-bench -m M -ngl 99 -p 512 -n 128 -r 5" \
    --b "LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib LLAMA_APU_GEMM_BACKEND=hipblaslt LLAMA_APU_HIPBLASLT_PREFILL=1 llama-bench -m M -ngl 99 -p 512 -n 128 -r 5"

  tools/apu-ab/apu_ab.py profile --cmd "llama-bench -m M -ngl 99 -p 128 -n 8 -r 1" --out /tmp/prof-a
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROCM = "/opt/rocm/core-10.1"
TEMP = "/sys/class/drm/card1/device/hwmon/hwmon8/temp1_input"
SCLK = "/sys/class/drm/card1/device/pp_dpm_sclk"

_ROW = re.compile(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([0-9.]+)")


def _gpu_temp_c():
    try:
        return round(int(Path(TEMP).read_text().strip()) / 1000)
    except Exception:
        return None


def _sclk_mhz():
    try:
        for line in Path(SCLK).read_text().splitlines():
            if "*" in line:
                return line.split()[1]
    except Exception:
        pass
    return None


def _run(cmd):
    """Run a shell command, return (stdout+stderr, rc)."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout + p.stderr, p.returncode


def _parse_metrics(out):
    """Extract {pp512: 483.6, tg128: 11.3} from llama-bench table output."""
    m = {}
    for key, val in _ROW.findall(out):
        m[key] = float(val)
    return m


def cmd_bench(args):
    runs = {"a": [], "b": []}
    for i in range(args.rounds):
        for arm, cmd in (("a", args.a), ("b", args.b)):
            t = _gpu_temp_c()
            print(f"[round {i+1}] arm {arm} start (gpu {t}C sclk {_sclk_mhz()})", file=sys.stderr)
            out, rc = _run(cmd)
            if rc != 0:
                print(f"  WARN arm {arm} rc={rc}", file=sys.stderr)
            metrics = _parse_metrics(out)
            if not metrics:
                print(f"  WARN arm {arm}: no metrics parsed (hang/crash?)", file=sys.stderr)
            runs[arm].append(metrics)
            time.sleep(args.cooldown)
    _report(runs, args)


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _report(runs, args):
    keys = sorted({k for arm in runs.values() for m in arm for k in m})
    print("\n=== apu_ab result ===")
    print(f"rounds={args.rounds} cooldown={args.cooldown}s")
    med = {}
    for arm in ("a", "b"):
        med[arm] = {k: _median([m.get(k) for m in runs[arm]]) for k in keys}
    header = "metric".ljust(10) + "A".rjust(12) + "B".rjust(12) + "delta".rjust(12)
    print(header)
    print("-" * len(header))
    for k in keys:
        a, b = med["a"].get(k), med["b"].get(k)
        if a is None or b is None:
            print(f"{k.ljust(10)}{str(a).rjust(12)}{str(b).rjust(12)}{'n/a'.rjust(12)}")
            continue
        pct = (b - a) / a * 100.0 if a else 0.0
        print(f"{k.ljust(10)}{f'{a:.2f}'.rjust(12)}{f'{b:.2f}'.rjust(12)}{f'{pct:+.1f}%'.rjust(12)}")
    print("\nper-round raw:")
    for arm in ("a", "b"):
        for i, m in enumerate(runs[arm]):
            print(f"  {arm}{i+1}: " + " ".join(f"{k}={m.get(k)}" for k in keys))
    if args.json:
        Path(args.json).write_text(json.dumps({"median": med, "raw": runs, "args": vars(args)}, indent=2))
        print(f"\nwrote {args.json}")
    print("\nverdict rule (per Dan): a significant improvement should become the default; "
          "otherwise the feature stays optional. Compare `delta` against run-to-run noise first.")


def cmd_profile(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = f"LD_LIBRARY_PATH={ROCM}/lib {args.cmd}"
    # kernel trace (timing runs are profiler-free; this is a separate pass)
    trace = out / "kernel_trace.csv"
    full = f"{ROCM}/bin/rocprofv3 --kernel-trace --output-format csv -o {out/'rocprof'} -- {cmd}"
    print(f"profiling: {full}", file=sys.stderr)
    txt, rc = _run(full)
    (out / "rocprof.log").write_text(txt)
    print(f"rc={rc}; logs in {out}")
    # summarise any csv produced
    csvs = list(out.glob("*.csv")) + list(out.glob("**/*.csv"))
    summary = {"files": [str(c) for c in csvs]}
    for c in csvs:
        try:
            lines = c.read_text(errors="ignore").splitlines()
            summary[c.name] = {"rows": max(0, len(lines) - 1)}
        except Exception:
            pass
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_diff(args):
    a, b = Path(args.a), Path(args.b)
    ta, tb = (a / "kernel_trace.csv"), (b / "kernel_trace.csv")
    def load(p):
        if not p.is_file():
            c = list(p.glob("**/*kernel*trace*.csv")) or list(p.glob("**/*.csv"))
            p = c[0] if c else None
        if not p:
            return None, None
        return p, p.read_text(errors="ignore")
    pa, da = load(a); pb, db = load(b)
    print(f"A: {pa}\nB: {pb}")
    if da is None or db is None:
        print("missing trace CSVs; run `profile` for each arm first")
        sys.exit(2)
    la, lb = da.splitlines(), db.splitlines()
    print(f"rows: A={max(0,len(la)-1)} B={max(0,len(lb)-1)}")
    print("(full kernel-level diff is profile-tool specific; summaries above)")


def main():
    ap = argparse.ArgumentParser(description="llama-apu local A/B harness (Phase 10.6)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="interleaved A/B benchmark")
    b.add_argument("--a", required=True)
    b.add_argument("--b", required=True)
    b.add_argument("--rounds", type=int, default=3)
    b.add_argument("--cooldown", type=int, default=10)
    b.add_argument("--json", default=None)
    b.set_defaults(func=cmd_bench)

    p = sub.add_parser("profile", help="rocprofv3 kernel-trace a command")
    p.add_argument("--cmd", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_profile)

    d = sub.add_parser("diff", help="diff two profile outputs")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
