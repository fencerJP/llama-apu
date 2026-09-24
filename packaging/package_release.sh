#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# llama-apu: Release Tarball Packager

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
VERSION="0.8.0-apu"
PKG_NAME="llama-apu-${VERSION}-linux-x86_64"
DIST_DIR="${ROOT_DIR}/dist"
STAGE_DIR="${DIST_DIR}/${PKG_NAME}"

echo "==================================================================="
echo "  llama-apu: Packaging Production Release                          "
echo "==================================================================="
echo "  Version   : ${VERSION}"
echo "  Artifact  : ${PKG_NAME}.tar.gz"
echo "==================================================================="

rm -rf "${STAGE_DIR}" "${DIST_DIR}/${PKG_NAME}.tar.gz"
mkdir -p "${STAGE_DIR}/bin"
mkdir -p "${STAGE_DIR}/xclbins"
mkdir -p "${STAGE_DIR}/packaging"

# 1. Copy Compiled Binaries
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

for b in "${BINARIES[@]}"; do
    if [ -f "${BUILD_DIR}/bin/${b}" ]; then
        cp "${BUILD_DIR}/bin/${b}" "${STAGE_DIR}/bin/"
        strip "${STAGE_DIR}/bin/${b}" 2>/dev/null || true
    fi
done

# Copy shared libraries
cp -a "${BUILD_DIR}/bin/"*.so* "${STAGE_DIR}/bin/" 2>/dev/null || true

# 2. Bundle 37+ Pre-compiled Hardware Graph Profiles
if [ -d "${HOME}/.local/share/llama-apu/xclbins" ]; then
    cp -r "${HOME}/.local/share/llama-apu/xclbins/"* "${STAGE_DIR}/xclbins/" 2>/dev/null || true
fi

# 3. Copy Packaging & Configuration Assets
cp "${SCRIPT_DIR}/install.sh" "${STAGE_DIR}/"
cp "${SCRIPT_DIR}/99-amdxdna-apu.rules" "${STAGE_DIR}/packaging/"
cp "${SCRIPT_DIR}/llama-server.service" "${STAGE_DIR}/packaging/"
cp "${ROOT_DIR}/README.md" "${STAGE_DIR}/"
cp "${ROOT_DIR}/CHANGELOG.md" "${STAGE_DIR}/"
cp "${ROOT_DIR}/ACKNOWLEDGEMENTS.md" "${STAGE_DIR}/"
cp "${ROOT_DIR}/LICENSE" "${STAGE_DIR}/"

chmod +x "${STAGE_DIR}/install.sh"
chmod +x "${STAGE_DIR}/bin/"*

# 4. Generate SHA-256 Manifest
cd "${STAGE_DIR}"
find . -type f ! -name "SHA256SUMS" -exec sha256sum {} + > SHA256SUMS
cd "${DIST_DIR}"

# 5. Create Compressed Tarball
tar -czf "${PKG_NAME}.tar.gz" "${PKG_NAME}"
sha256sum "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256"

echo "==================================================================="
echo "  [+] Release package generated successfully:                      "
echo "      ${DIST_DIR}/${PKG_NAME}.tar.gz"
echo "  [+] Size: $(du -h "${DIST_DIR}/${PKG_NAME}.tar.gz" | cut -f1)"
echo "  [+] Checksum: $(cat "${DIST_DIR}/${PKG_NAME}.tar.gz.sha256")"
echo "==================================================================="
