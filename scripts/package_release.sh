#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# AMD Ryzen AI APU Zero-Copy Release Packaging Script

set -euo pipefail

VERSION="${1:-v1.0.0}"
ARCH="linux-x86_64"
BUNDLE_NAME="llama-apu-${VERSION}-${ARCH}"
OUTPUT_DIR="dist"

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RESET="\033[0m"

echo -e "${BOLD}${BLUE}============================================================${RESET}"
echo -e "${BOLD}${BLUE} Packaging ${BUNDLE_NAME}${RESET}"
echo -e "${BOLD}${BLUE}============================================================${RESET}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_CPP_DIR="$(cd "$REPO_ROOT/../llamacpp-update/llama.cpp" 2>/dev/null || cd "$REPO_ROOT" && pwd)"

# 1. Build Rust Release Artifacts
echo -e "\n${BOLD}[1/5] Building Rust APU Backend Engine...${RESET}"
cd "$REPO_ROOT"
RUSTFLAGS="-C target-cpu=native" cargo build --release

# 2. Build C++ llama.cpp Release Artifacts
echo -e "\n${BOLD}[2/5] Building C++ llama.cpp Front-End...${RESET}"
cd "$LLAMA_CPP_DIR"
cmake -B build -DLLAMA_APU_BACKEND=ON -DAPU_BACKEND_DIR="$REPO_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

# 3. Assemble Release Directory
echo -e "\n${BOLD}[3/5] Assembling Release Files...${RESET}"
cd "$REPO_ROOT"
rm -rf "$OUTPUT_DIR/${BUNDLE_NAME}"
mkdir -p "$OUTPUT_DIR/${BUNDLE_NAME}"/{bin,lib,include,etc/udev/rules.d,etc/systemd/system,etc/default,docs}

# Binaries
cp "$REPO_ROOT/target/release/apu-doctor" "$OUTPUT_DIR/${BUNDLE_NAME}/bin/"
cp "$REPO_ROOT/target/release/apu-model" "$OUTPUT_DIR/${BUNDLE_NAME}/bin/"

for bin in llama-cli llama-server apu-run llama-bench llama-quantize; do
    if [ -f "$LLAMA_CPP_DIR/build/bin/$bin" ]; then
        cp "$LLAMA_CPP_DIR/build/bin/$bin" "$OUTPUT_DIR/${BUNDLE_NAME}/bin/"
    fi
done

# Create convenience alias symlink
cd "$OUTPUT_DIR/${BUNDLE_NAME}/bin"
ln -sf llama-cli llama
cd "$REPO_ROOT"

# Libraries & Headers
if [ -f "$REPO_ROOT/target/release/libzero_copy_model_runner.so" ]; then
    cp "$REPO_ROOT/target/release/libzero_copy_model_runner.so" "$OUTPUT_DIR/${BUNDLE_NAME}/lib/"
fi
if [ -f "$REPO_ROOT/target/release/libzero_copy_model_runner.a" ]; then
    cp "$REPO_ROOT/target/release/libzero_copy_model_runner.a" "$OUTPUT_DIR/${BUNDLE_NAME}/lib/"
fi
cp -r "$REPO_ROOT/include/"* "$OUTPUT_DIR/${BUNDLE_NAME}/include/"
cp "$LLAMA_CPP_DIR/include/llama.h" "$OUTPUT_DIR/${BUNDLE_NAME}/include/" 2>/dev/null || true

# Configurations
cp "$REPO_ROOT/scripts/99-amdxdna-apu.rules" "$OUTPUT_DIR/${BUNDLE_NAME}/etc/udev/rules.d/"
cp "$REPO_ROOT/scripts/llama-server.service" "$OUTPUT_DIR/${BUNDLE_NAME}/etc/systemd/system/"
cp "$REPO_ROOT/scripts/llama-server.default" "$OUTPUT_DIR/${BUNDLE_NAME}/etc/default/llama-server"

# Documentation
for doc in README.md CLI_GUIDE.md QUANTIZATION.md HARDWARE_SUPPORT.md CHANGELOG.md; do
    if [ -f "$REPO_ROOT/$doc" ]; then
        cp "$REPO_ROOT/$doc" "$OUTPUT_DIR/${BUNDLE_NAME}/docs/"
    fi
done
if [ -f "$REPO_ROOT/docs/TROUBLESHOOTING.md" ]; then
    cp "$REPO_ROOT/docs/TROUBLESHOOTING.md" "$OUTPUT_DIR/${BUNDLE_NAME}/docs/"
fi
if [ -f "$LLAMA_CPP_DIR/docs/backend/APU.md" ]; then
    cp "$LLAMA_CPP_DIR/docs/backend/APU.md" "$OUTPUT_DIR/${BUNDLE_NAME}/docs/"
fi

# Top-level Readme & License
cp "$REPO_ROOT/README.md" "$OUTPUT_DIR/${BUNDLE_NAME}/" 2>/dev/null || true
cp "$REPO_ROOT/CLI_GUIDE.md" "$OUTPUT_DIR/${BUNDLE_NAME}/" 2>/dev/null || true

# 4. Generate Precompiled Bundle Installer
cat << 'INSTALLER_EOF' > "$OUTPUT_DIR/${BUNDLE_NAME}/install.sh"
#!/usr/bin/env bash
# Turnkey Installer for Pre-compiled llama-apu Bundle
set -euo pipefail

DEST="/usr/local/bin"
if [ ! -w "$DEST" ] && [ "${EUID:-$(id -u)}" -ne 0 ]; then
    DEST="$HOME/.local/bin"
    mkdir -p "$DEST"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing llama-apu binaries to $DEST..."
cp -a "$SCRIPT_DIR/bin/"* "$DEST/"

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    if [ -d "/etc/udev/rules.d" ]; then
        cp "$SCRIPT_DIR/etc/udev/rules.d/99-amdxdna-apu.rules" /etc/udev/rules.d/
        udevadm control --reload-rules && udevadm trigger || true
    fi
    if [ -d "/etc/systemd/system" ]; then
        cp "$SCRIPT_DIR/etc/systemd/system/llama-server.service" /etc/systemd/system/
        [ ! -f "/etc/default/llama-server" ] && cp "$SCRIPT_DIR/etc/default/llama-server" /etc/default/
        systemctl daemon-reload || true
    fi
else
    echo "Note: Run with sudo to install udev hardware rules and systemd service."
fi

echo "Installation complete! Verifying hardware:"
"$DEST/apu-doctor" || true
INSTALLER_EOF
chmod +x "$OUTPUT_DIR/${BUNDLE_NAME}/install.sh"

# 5. Create Compressed Tarball & SHA-256 Checksum
echo -e "\n${BOLD}[4/5] Creating Compressed Tarball...${RESET}"
cd "$OUTPUT_DIR"
tar -czvf "${BUNDLE_NAME}.tar.gz" "${BUNDLE_NAME}"

echo -e "\n${BOLD}[5/5] Generating Checksum...${RESET}"
sha256sum "${BUNDLE_NAME}.tar.gz" > "${BUNDLE_NAME}.tar.gz.sha256"

echo -e "\n${BOLD}${GREEN}============================================================${RESET}"
echo -e "${BOLD}${GREEN} Release Package Created Successfully!${RESET}"
echo -e "${BOLD}${GREEN} Bundle: ${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz${RESET}"
echo -e "${BOLD}${GREEN} SHA256: $(cat "${BUNDLE_NAME}.tar.gz.sha256")${RESET}"
echo -e "${BOLD}${GREEN}============================================================${RESET}"
