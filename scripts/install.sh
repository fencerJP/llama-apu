#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# AMD Ryzen AI APU Zero-Copy llama.cpp Unified Installer

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo -e "${BOLD}${BLUE}============================================================${RESET}"
echo -e "${BOLD}${BLUE} AMD Ryzen AI APU Unified llama.cpp Installer${RESET}"
echo -e "${BOLD}${BLUE}============================================================${RESET}"

# 1. System & Architecture Pre-flight
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo -e "${RED}Error: Only x86_64 architecture (AMD Ryzen AI APUs) is supported.${RESET}"
    exit 1
fi

OS=$(uname -s)
if [ "$OS" != "Linux" ]; then
    echo -e "${RED}Error: Only Linux is supported.${RESET}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_CPP_DIR="$(cd "$REPO_ROOT/../llamacpp-update/llama.cpp" 2>/dev/null || cd "$REPO_ROOT" && pwd)"

# 2. Determine Destination Directory
INSTALL_DIR="/usr/local/bin"
if [ ! -w "$INSTALL_DIR" ]; then
    if [ "${EUID:-$(id -u)}" -ne 0 ]; then
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    fi
fi

echo -e "${CYAN}Target installation directory: ${BOLD}${INSTALL_DIR}${RESET}"

# 3. Build Rust APU Backend
echo -e "\n${BOLD}[1/4] Building Rust Hardware Acceleration Engine (libzero_copy_model_runner)...${RESET}"
cd "$REPO_ROOT"
if command -v cargo >/dev/null 2>&1; then
    RUSTFLAGS="-C target-cpu=native" cargo build --release
else
    echo -e "${YELLOW}Warning: cargo not found. Checking for pre-compiled binaries...${RESET}"
    if [ ! -f "$REPO_ROOT/target/release/libzero_copy_model_runner.a" ]; then
        echo -e "${RED}Error: cargo is required to build the APU backend from source.${RESET}"
        exit 1
    fi
fi

# 4. Build C++ llama.cpp with APU Backend Enabled
echo -e "\n${BOLD}[2/4] Building C++ llama.cpp Front-End with Zero-Copy APU Backend...${RESET}"
if [ -d "$LLAMA_CPP_DIR" ] && [ -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]; then
    cd "$LLAMA_CPP_DIR"
    cmake -B build -DLLAMA_APU_BACKEND=ON -DAPU_BACKEND_DIR="$REPO_ROOT"
    cmake --build build --config Release -j"$(nproc)"
else
    echo -e "${YELLOW}Notice: llama.cpp directory not found at $LLAMA_CPP_DIR; skipping C++ build step.${RESET}"
fi

# 5. Install Binaries
echo -e "\n${BOLD}[3/4] Installing Executables to $INSTALL_DIR...${RESET}"

# Install Rust utilities
for tool in apu-doctor apu-model; do
    SRC="$REPO_ROOT/target/release/$tool"
    if [ -f "$SRC" ]; then
        echo -e "  -> Installing ${GREEN}$tool${RESET}"
        cp "$SRC" "$INSTALL_DIR/$tool"
        chmod +x "$INSTALL_DIR/$tool"
    fi
done

# Install C++ binaries from llama.cpp build
if [ -d "$LLAMA_CPP_DIR/build/bin" ]; then
    for bin in llama-cli llama-server apu-run llama-bench llama-quantize; do
        SRC="$LLAMA_CPP_DIR/build/bin/$bin"
        if [ -f "$SRC" ]; then
            echo -e "  -> Installing ${GREEN}$bin${RESET}"
            cp "$SRC" "$INSTALL_DIR/$bin"
            chmod +x "$INSTALL_DIR/$bin"
        fi
    done
    # Symlink 'llama' convenience alias
    if [ -f "$INSTALL_DIR/llama-cli" ]; then
        echo -e "  -> Creating convenience alias ${GREEN}llama${RESET} -> llama-cli"
        ln -sf "$INSTALL_DIR/llama-cli" "$INSTALL_DIR/llama"
    fi
fi

# 6. Install & Register XCLBIN Hardware Profiles
XCLBINS_DEST="/usr/local/share/llama-apu/xclbins"
if [ ! -w "/usr/local/share" ] && [ "${EUID:-$(id -u)}" -ne 0 ]; then
    XCLBINS_DEST="$HOME/.local/share/llama-apu/xclbins"
fi
echo -e "\n${BOLD}[4/5] Registering & Installing System XCLBIN Hardware Profiles...${RESET}"
echo -e "  -> Target XCLBIN directory: ${GREEN}$XCLBINS_DEST${RESET}"
mkdir -p "$XCLBINS_DEST"
if [ -d "$REPO_ROOT/xclbins" ]; then
    cp -r "$REPO_ROOT/xclbins/"* "$XCLBINS_DEST/"
    echo -e "  -> Successfully registered $(ls -1 "$XCLBINS_DEST" | wc -l) XDNA 2 hardware profiles"
fi

# 7. System Configurations (Udev & Systemd)
echo -e "\n${BOLD}[5/5] Configuring System Permissions & Services...${RESET}"

# Udev rules
UDEV_SRC="$REPO_ROOT/scripts/99-amdxdna-apu.rules"
if [ -f "$UDEV_SRC" ]; then
    if [ "${EUID:-$(id -u)}" -eq 0 ] && [ -d "/etc/udev/rules.d" ]; then
        echo -e "  -> Installing udev rules to /etc/udev/rules.d/99-amdxdna-apu.rules"
        cp "$UDEV_SRC" /etc/udev/rules.d/99-amdxdna-apu.rules
        udevadm control --reload-rules || true
        udevadm trigger || true
    else
        echo -e "${YELLOW}  -> To install udev rules for non-root hardware access, run:${RESET}"
        echo -e "     sudo cp $UDEV_SRC /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger"
    fi
fi

# Systemd service
SERVICE_SRC="$REPO_ROOT/scripts/llama-server.service"
DEFAULT_SRC="$REPO_ROOT/scripts/llama-server.default"
if [ "${EUID:-$(id -u)}" -eq 0 ] && [ -d "/etc/systemd/system" ]; then
    echo -e "  -> Installing systemd service unit to /etc/systemd/system/llama-server.service"
    cp "$SERVICE_SRC" /etc/systemd/system/llama-server.service
    if [ -d "/etc/default" ] && [ ! -f "/etc/default/llama-server" ]; then
        cp "$DEFAULT_SRC" /etc/default/llama-server
    fi
    systemctl daemon-reload || true
    echo -e "  -> Systemd service ready. Configure /etc/default/llama-server and run: ${BOLD}systemctl enable --now llama-server${RESET}"
fi

# PATH advice
if [ "$INSTALL_DIR" = "$HOME/.local/bin" ]; then
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo -e "\n${YELLOW}Notice: Add ~/.local/bin to your PATH in ~/.bashrc or ~/.zshrc:${RESET}"
        echo -e "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
fi

echo -e "\n${BOLD}${GREEN}============================================================${RESET}"
echo -e "${BOLD}${GREEN} Installation Complete!${RESET}"
echo -e "${BOLD}${GREEN}============================================================${RESET}"

# 7. Post-install Hardware Diagnostic Probe
echo -e "\n${BOLD}Executing Hardware Diagnostic Verification:${RESET}"
if [ -x "$INSTALL_DIR/apu-doctor" ]; then
    "$INSTALL_DIR/apu-doctor" || true
fi
