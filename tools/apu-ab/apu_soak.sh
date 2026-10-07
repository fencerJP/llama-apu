#!/usr/bin/env bash
# llama-apu Phase 10.7 soak — repeated inference, sampling RSS + GPU temp/GTT for drift.
# Usage: apu_soak.sh <minutes> <model> <bin-dir>
set -u
MIN="${1:-60}"; MODEL="$2"; BIN="${3:-build-rocm101/bin}"
LD=/opt/rocm/core-10.1/lib
LOG=/tmp/apu-soak-$(date +%Y%m%d-%H%M%S).log
SAMPLES=/tmp/apu-soak-samples.txt
: > "$LOG"; : > "$SAMPLES"
END=$(( $(date +%s) + MIN*60 ))
round=0
while [ "$(date +%s)" -lt "$END" ]; do
  round=$((round+1))
  echo "=== round $round @ $(date +%T) ===" >> "$LOG"
  LD_LIBRARY_PATH=$LD "$BIN/llama-bench" -m "$MODEL" -ngl 99 -p 512 -n 256 -r 40 >> "$LOG" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
    t=$(awk '{printf "%.0f",$1/1000}' /sys/class/drm/card1/device/hwmon/hwmon8/temp1_input 2>/dev/null)
    echo "$(date +%s) rss_kb=${rss:-0} temp_c=${t:-0}" >> "$SAMPLES"
    sleep 20
  done
  wait "$pid"
done
echo "SOAK_DONE rounds=$round log=$LOG" >> "$LOG"
echo "SOAK_DONE log=$LOG samples=$SAMPLES"
python3 - "$SAMPLES" <<'PY'
import sys, statistics
rows=[l.split() for l in open(sys.argv[1]) if 'rss_kb=' in l]
rss=[int(r[1].split('=')[1]) for r in rows if int(r[1].split('=')[1])>0]
if rss:
    print(f"samples={len(rss)} rss_first_kb={rss[0]} rss_last_kb={rss[-1]} rss_min={min(rss)} rss_max={max(rss)} rss_median={statistics.median(rss):.0f}")
    print(f"drift_first_to_last_MB={(rss[-1]-rss[0])/1024:.1f}")
else:
    print("no samples")
PY
