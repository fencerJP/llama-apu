#!/usr/bin/env bash
# convert_all_to_gguf.sh
# Converts every model directory in MODEL_DIR to GGUF (outtype=auto)
# using the updated convert_hf_to_gguf.py from the local llama.cpp tree.

set -euo pipefail

LLAMA_CPP_DIR="/home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp"
CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
PYTHON="/home/fencer/.openclaw/workspace/projects/zero-copy_model_runner/.venv/bin/python3"
MODEL_DIR="/mnt/Media/Downloads/model_testing"

export PYTHONPATH="$LLAMA_CPP_DIR/gguf-py:$LLAMA_CPP_DIR:${PYTHONPATH:-}"

echo "=== llama-convert: batch GGUF conversion (outtype=auto) ==="
echo "Source dir : $MODEL_DIR"
echo "Script     : $CONVERT_SCRIPT"
echo ""

for model_path in "$MODEL_DIR"/*/; do
    model_name="$(basename "$model_path")"

    # Skip if no config.json (not a HF model directory)
    if [[ ! -f "$model_path/config.json" ]]; then
        echo "[SKIP] $model_name — no config.json"
        continue
    fi

    echo "──────────────────────────────────────────"
    echo "[CONVERT] $model_name"
    "$PYTHON" "$CONVERT_SCRIPT" "$model_path" --outtype auto
    echo "[DONE]    $model_name"
    echo ""
done

echo "=== All conversions complete ==="
