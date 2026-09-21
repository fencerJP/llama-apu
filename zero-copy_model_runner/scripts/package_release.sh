#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# AMD Ryzen AI APU Zero-Copy Release Packaging Script

set -euo pipefail

VERSION="${1:-0.5.0}"
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
echo -e "\n${BOLD}[1/5] Building Rust APU Backend Engine & Utilities...${RESET}"
cd "$REPO_ROOT"
RUSTFLAGS="-C target-cpu=native" cargo build --release --bins

# 2. Build C++ llama.cpp Release Artifacts
echo -e "\n${BOLD}[2/5] Building C++ llama.cpp Front-End...${RESET}"
if [ -d "$LLAMA_CPP_DIR" ] && [ -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]; then
    cd "$LLAMA_CPP_DIR"
    cmake -B build -DLLAMA_APU_BACKEND=ON -DAPU_BACKEND_DIR="$REPO_ROOT" -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$(nproc)"
fi

# 3. Assemble Release Directory
echo -e "\n${BOLD}[3/5] Assembling Release Files...${RESET}"
cd "$REPO_ROOT"
rm -rf "$OUTPUT_DIR/${BUNDLE_NAME}"
mkdir -p "$OUTPUT_DIR/${BUNDLE_NAME}"/{bin,lib,include,converter,scripts,etc/udev/rules.d,etc/systemd/system,etc/default,docs}

# Binaries (including utilities & unified multiplexer)
for tool in apu-doctor apu-model apu-synth llama-convert; do
    if [ -f "$REPO_ROOT/target/release/$tool" ]; then
        cp "$REPO_ROOT/target/release/$tool" "$OUTPUT_DIR/${BUNDLE_NAME}/bin/"
    fi
done

if [ -d "$LLAMA_CPP_DIR/build/bin" ]; then
    for bin in llama llama-cli llama-server apu-run llama-bench llama-quantize; do
        if [ -f "$LLAMA_CPP_DIR/build/bin/$bin" ]; then
            cp "$LLAMA_CPP_DIR/build/bin/$bin" "$OUTPUT_DIR/${BUNDLE_NAME}/bin/"
        fi
    done
fi

# Libraries & Headers
if [ -f "$REPO_ROOT/target/release/libzero_copy_model_runner.so" ]; then
    cp "$REPO_ROOT/target/release/libzero_copy_model_runner.so" "$OUTPUT_DIR/${BUNDLE_NAME}/lib/"
fi
if [ -f "$REPO_ROOT/target/release/libzero_copy_model_runner.a" ]; then
    cp "$REPO_ROOT/target/release/libzero_copy_model_runner.a" "$OUTPUT_DIR/${BUNDLE_NAME}/lib/"
fi
if [ -d "$REPO_ROOT/include" ]; then
    cp -r "$REPO_ROOT/include/"* "$OUTPUT_DIR/${BUNDLE_NAME}/include/"
fi
if [ -f "$LLAMA_CPP_DIR/include/llama.h" ]; then
    cp "$LLAMA_CPP_DIR/include/llama.h" "$OUTPUT_DIR/${BUNDLE_NAME}/include/"
fi

# Converter Tools & Requirements
if [ -d "$REPO_ROOT/converter" ]; then
    cp -r "$REPO_ROOT/converter/"*.py "$OUTPUT_DIR/${BUNDLE_NAME}/converter/" 2>/dev/null || true
    cp -r "$REPO_ROOT/converter/requirements.txt" "$OUTPUT_DIR/${BUNDLE_NAME}/converter/" 2>/dev/null || true
fi
if [ -f "$REPO_ROOT/requirements.txt" ]; then
    cp "$REPO_ROOT/requirements.txt" "$OUTPUT_DIR/${BUNDLE_NAME}/"
fi

# Configurations & Helper Scripts
cp "$REPO_ROOT/scripts/99-amdxdna-apu.rules" "$OUTPUT_DIR/${BUNDLE_NAME}/etc/udev/rules.d/"
cp "$REPO_ROOT/scripts/llama-server.service" "$OUTPUT_DIR/${BUNDLE_NAME}/etc/systemd/system/"
cp "$REPO_ROOT/scripts/llama-server.default" "$OUTPUT_DIR/${BUNDLE_NAME}/etc/default/llama-server"
cp "$REPO_ROOT/scripts/download_models.sh" "$OUTPUT_DIR/${BUNDLE_NAME}/scripts/"

# Documentation
for doc in README.md CLI_GUIDE.md QUANTIZATION.md HARDWARE_SUPPORT.md QUICKSTART.md CHANGELOG.md; do
    if [ -f "$REPO_ROOT/$doc" ]; then
        cp "$REPO_ROOT/$doc" "$OUTPUT_DIR/${BUNDLE_NAME}/docs/"
    fi
done
if [ -f "$REPO_ROOT/docs/TROUBLESHOOTING.md" ]; then
    cp "$REPO_ROOT/docs/TROUBLESHOOTING.md" "$OUTPUT_DIR/${BUNDLE_NAME}/docs/"
fi
if [ -f "$REPO_ROOT/docs/AMD_APU_DEVELOPER_REFERENCE.md" ]; then
    cp "$REPO_ROOT/docs/AMD_APU_DEVELOPER_REFERENCE.md" "$OUTPUT_DIR/${BUNDLE_NAME}/docs/"
fi
if [ -f "$LLAMA_CPP_DIR/docs/backend/APU.md" ]; then
    cp "$LLAMA_CPP_DIR/docs/backend/APU.md" "$OUTPUT_DIR/${BUNDLE_NAME}/docs/"
fi

# XCLBIN Profiles
if [ -d "$REPO_ROOT/xclbins" ]; then
    echo "Copying XDNA 2 hardware profile bank..."
    cp -r "$REPO_ROOT/xclbins" "$OUTPUT_DIR/${BUNDLE_NAME}/"
fi

# Top-level Readme & License
cp "$REPO_ROOT/README.md" "$OUTPUT_DIR/${BUNDLE_NAME}/" 2>/dev/null || true
cp "$REPO_ROOT/CLI_GUIDE.md" "$OUTPUT_DIR/${BUNDLE_NAME}/" 2>/dev/null || true

# 4. Generate Precompiled Bundle Installer
cat << 'INSTALLER_EOF' > "$OUTPUT_DIR/${BUNDLE_NAME}/install.sh"
#!/usr/bin/env bash
# Turnkey Installer for Pre-compiled AMD Ryzen AI APU llama.cpp Bundle
set -euo pipefail

DEST="/usr/local/bin"
XCLBINS_DEST="/usr/local/share/llama-apu/xclbins"
if [ ! -w "$DEST" ] && [ "${EUID:-$(id -u)}" -ne 0 ]; then
    DEST="$HOME/.local/bin"
    XCLBINS_DEST="$HOME/.local/share/llama-apu/xclbins"
    mkdir -p "$DEST"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing AMD Ryzen AI APU binaries to $DEST..."
cp -a "$SCRIPT_DIR/bin/"* "$DEST/"

echo "Registering and installing XCLBIN hardware profiles to $XCLBINS_DEST..."
mkdir -p "$XCLBINS_DEST"
if [ -d "$SCRIPT_DIR/xclbins" ]; then
    cp -r "$SCRIPT_DIR/xclbins/"* "$XCLBINS_DEST/"
fi

# Install Python requirements if python3 and pip are available
if [ -f "$SCRIPT_DIR/requirements.txt" ] && command -v python3 >/dev/null 2>&1; then
    PIP_CMD=""
    if command -v pip3 >/dev/null 2>&1; then
        PIP_CMD="pip3"
    elif command -v pip >/dev/null 2>&1; then
        PIP_CMD="pip"
    elif python3 -m pip --version >/dev/null 2>&1; then
        PIP_CMD="python3 -m pip"
    fi
    if [ -n "$PIP_CMD" ]; then
        echo "Installing Python converter dependencies..."
        $PIP_CMD install -r "$SCRIPT_DIR/requirements.txt" --break-system-packages 2>/dev/null || \
        $PIP_CMD install -r "$SCRIPT_DIR/requirements.txt" --user 2>/dev/null || \
        $PIP_CMD install -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || true
    fi
fi

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
    if [ -d "/etc/udev/rules.d" ] && [ -f "$SCRIPT_DIR/etc/udev/rules.d/99-amdxdna-apu.rules" ]; then
        cp "$SCRIPT_DIR/etc/udev/rules.d/99-amdxdna-apu.rules" /etc/udev/rules.d/
        udevadm control --reload-rules && udevadm trigger || true
    fi
    if [ -d "/etc/systemd/system" ] && [ -f "$SCRIPT_DIR/etc/systemd/system/llama-server.service" ]; then
        cp "$SCRIPT_DIR/etc/systemd/system/llama-server.service" /etc/systemd/system/
        [ ! -f "/etc/default/llama-server" ] && [ -f "$SCRIPT_DIR/etc/default/llama-server" ] && cp "$SCRIPT_DIR/etc/default/llama-server" /etc/default/
        systemctl daemon-reload || true
    fi
else
    echo "Note: To install system udev rules and systemd daemon, re-run with sudo."
fi

# Check User Hardware Permissions
CURRENT_GROUPS=$(id -Gn 2>/dev/null || echo "")
if [[ "$CURRENT_GROUPS" != *"render"* ]] || [[ "$CURRENT_GROUPS" != *"video"* ]]; then
    echo "Notice: Add your user to the 'render' and 'video' groups for non-root hardware access:"
    echo "  sudo usermod -a -G render,video $(whoami)"
fi

echo "Installation complete! Verifying hardware diagnostics:"
if [ -x "$DEST/apu-doctor" ]; then
    "$DEST/apu-doctor" || true
fi
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
