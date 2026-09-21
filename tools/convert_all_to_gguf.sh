#!/usr/bin/env bash
# convert_all_to_gguf.sh
# Converts every model directory in MODEL_DIR to billm-quant gguf-embedded q4nx models
# using the llama-convert CLI utility.

set -euo pipefail

LLAMA_CONVERT_BIN="/home/fencer/.openclaw/workspace/projects/zero-copy_model_runner/target/release/llama-convert"
if [[ ! -x "$LLAMA_CONVERT_BIN" ]]; then
    if command -v llama-convert &>/dev/null; then
        LLAMA_CONVERT_BIN="llama-convert"
    else
        echo "Error: llama-convert binary not found!" >&2
        exit 1
    fi
fi

MODEL_DIR="/mnt/Media/Downloads/model_testing"

echo "=== llama-convert: batch BiLLM-quant GGUF-embedded Q4NX conversion ==="
echo "Source dir : $MODEL_DIR"
echo "Binary     : $LLAMA_CONVERT_BIN"
echo ""

for model_path in "$MODEL_DIR"/*/; do
    model_name="$(basename "$model_path")"

    # Skip if no config.json (not a HF model directory)
    if [[ ! -f "$model_path/config.json" ]]; then
        echo "[SKIP] $model_name — no config.json"
        continue
    fi

    clean_name="$(echo "$model_name" | tr '[:upper:]' '[:lower:]')"
    output_path="$MODEL_DIR/${clean_name}-billm.q4nx"

    echo "──────────────────────────────────────────"
    echo "[CONVERT] $model_name -> $(basename "$output_path")"
    "$LLAMA_CONVERT_BIN" convert --input "$model_path" --output "$output_path"
    echo "[DONE]    $model_name"
    echo ""
done

echo "=== All conversions complete ==="
