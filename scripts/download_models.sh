#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Detached Safetensors Downloader for Model Testing on NAS / Local Storage

set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: $0 [DEST_DIR]"
    echo "Downloads designated testing models using huggingface-cli / hf."
    echo "Default destination: /mnt/Media/Downloads/model_testing (or \$MODEL_TEST_DIR)"
    exit 0
fi

DEST_BASE="${1:-${MODEL_TEST_DIR:-/mnt/Media/Downloads/model_testing}}"
LOG_FILE="${DEST_BASE}/download_progress.log"

mkdir -p "${DEST_BASE}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

# Resolve Hugging Face CLI binary or module invocation dynamically
resolve_hf_cli() {
    if command -v hf >/dev/null 2>&1; then
        echo "hf"
    elif [ -x "$HOME/.local/bin/hf" ]; then
        echo "$HOME/.local/bin/hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli"
    elif [ -x "$HOME/.local/bin/huggingface-cli" ]; then
        echo "$HOME/.local/bin/huggingface-cli"
    elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        echo "python3 -m huggingface_hub.cli.hf_cli"
    else
        echo ""
    fi
}

HF_CMD=$(resolve_hf_cli)
if [ -z "$HF_CMD" ]; then
    echo -e "\033[0;31mError: huggingface-cli or hf command not found.\033[0m"
    echo -e "Please install huggingface_hub via: \033[1mpip install -U huggingface_hub\033[0m"
    exit 1
fi

log "Starting detached safetensors download sequence to ${DEST_BASE}"
log "Using Hugging Face downloader command: ${HF_CMD}"

declare -A MODELS=(
    ["gemma-4-31B"]="google/gemma-4-31B"
    ["Qwen3.8-27B-Cold-Fusion"]="DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
    ["Qwen3-Coder-Next"]="Qwen/Qwen3-Coder-Next"
    ["sarvam-105b"]="sarvamai/sarvam-105b"
    ["Laguna-S-2.1"]="poolside/Laguna-S-2.1"
    ["Qwen3.8-Flash-Next"]="Qwen/Qwen3.8-Flash-Next"
    ["DeepSeek-V4-Flash-DSpark"]="deepseek-ai/DeepSeek-V4-Flash-DSpark"
    ["DeepSeek-V4.1-Flash"]="deepseek-ai/DeepSeek-V4.1-Flash"
    ["DeepSeek-V4-Flash-0731"]="deepseek-ai/DeepSeek-V4-Flash-0731"
    ["GLM-5.3-Flash"]="zai-org/GLM-5.3-Flash"
)

# Ordered sequence starting with Gemma-4-31B and Cold-Fusion
ORDER=(
    "gemma-4-31B"
    "Qwen3.8-27B-Cold-Fusion"
    "Qwen3-Coder-Next"
    "sarvam-105b"
    "Laguna-S-2.1"
    "Qwen3.8-Flash-Next"
    "DeepSeek-V4-Flash-DSpark"
    "DeepSeek-V4.1-Flash"
    "DeepSeek-V4-Flash-0731"
    "GLM-5.3-Flash"
)

for NAME in "${ORDER[@]}"; do
    REPO="${MODELS[$NAME]}"
    TARGET_DIR="${DEST_BASE}/${NAME}"
    log "========================================================"
    log "Starting download: ${REPO} -> ${TARGET_DIR}"
    log "========================================================"

    mkdir -p "${TARGET_DIR}"

    if $HF_CMD download "${REPO}" \
        --local-dir "${TARGET_DIR}" \
        >> "${LOG_FILE}" 2>&1; then
        log "Successfully completed: ${REPO}"
    else
        log "WARNING: Download error for ${REPO}. See ${LOG_FILE} for details."
    fi
done

log "All model download tasks completed."
