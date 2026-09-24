#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# llama-apu: Automated System Installer

set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"
BIN_DIR="${PREFIX}/bin"
SHARE_DIR="${PREFIX}/share/llama-apu"
XCLBIN_DIR="${SHARE_DIR}/xclbins"
UDEV_DIR="/etc/udev/rules.d"
SYSTEMD_DIR="/etc/systemd/system"

echo "==================================================================="
echo "  llama-apu: Heterogeneous APU Inference Engine Installer          "
echo "==================================================================="
echo "  Target Prefix : ${PREFIX}"
echo "  Binary Dir    : ${BIN_DIR}"
echo "  XCLBIN Share  : ${XCLBIN_DIR}"
echo "==================================================================="

# Ensure root permissions for system paths
IS_ROOT=0
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=1
fi

# 1. Install Binaries
echo "[*] Installing core runtime binaries to ${BIN_DIR}..."
mkdir -p "${BIN_DIR}"
BINARIES=(
    "llama"
    "llama-cli"
    "llama-server"
    "llama-apu-cli"
    "llama-bench"
    "llama-quantize"
    "llama-perplexity"
    "llama-tokenize"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_BIN="${ROOT_DIR}/build/bin"

for b in "${BINARIES[@]}"; do
    if [ -f "${BUILD_BIN}/${b}" ]; then
        install -m 0755 "${BUILD_BIN}/${b}" "${BIN_DIR}/${b}"
        echo "    [+] Installed ${b}"
    elif [ -f "${SCRIPT_DIR}/bin/${b}" ]; then
        install -m 0755 "${SCRIPT_DIR}/bin/${b}" "${BIN_DIR}/${b}"
        echo "    [+] Installed ${b}"
    fi
done

# 2. Setup Hardware Profile Registry
echo "[*] Setting up hardware graph profile (XCLBIN) directories..."
mkdir -p "${XCLBIN_DIR}"
if [ -d "${HOME}/.local/share/llama-apu/xclbins" ]; then
    cp -rn "${HOME}/.local/share/llama-apu/xclbins/"* "${XCLBIN_DIR}/" 2>/dev/null || true
    echo "    [+] Synchronized pre-compiled XCLBIN profiles to ${XCLBIN_DIR}"
fi

# 3. Install udev rules
if [ "${IS_ROOT}" -eq 1 ] && [ -d "${UDEV_DIR}" ]; then
    echo "[*] Installing udev hardware permission rules..."
    install -m 0644 "${SCRIPT_DIR}/99-amdxdna-apu.rules" "${UDEV_DIR}/99-amdxdna-apu.rules"
    udevadm control --reload-rules || true
    udevadm trigger || true
    echo "    [+] Udev rules applied."
fi

# 4. Install systemd unit
if [ "${IS_ROOT}" -eq 1 ] && [ -d "${SYSTEMD_DIR}" ]; then
    echo "[*] Installing systemd service unit..."
    install -m 0644 "${SCRIPT_DIR}/llama-server.service" "${SYSTEMD_DIR}/llama-server.service"
    systemctl daemon-reload || true
    echo "    [+] Service unit installed (use 'systemctl enable --now llama-server' to activate)."
fi

echo ""
echo "==================================================================="
echo "  llama-apu Installation Complete!                                 "
echo "  Run 'llama-apu-cli apu-doctor' to verify system readiness.       "
echo "==================================================================="
