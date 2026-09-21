#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# AMD Ryzen AI APU Zero-Copy llama.cpp Unified Installer & Dependency Provisioner

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
RESET="\033[0m"

NON_INTERACTIVE=false
INSTALL_PYTHON=true
DRY_RUN=false

print_usage() {
    echo -e "Usage: $0 [OPTIONS]"
    echo -e "Options:"
    echo -e "  -y, --yes          Non-interactive mode (automatically install missing system packages)"
    echo -e "  --no-python        Skip Python dependencies installation"
    echo -e "  --dry-run          Check dependencies and system requirements without installing"
    echo -e "  -h, --help         Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)
            NON_INTERACTIVE=true
            shift
            ;;
        --no-python)
            INSTALL_PYTHON=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${RESET}"
            print_usage
            exit 1
            ;;
    esac
done

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

# 2. Dependency Audit & Auto-Provisioning
echo -e "\n${BOLD}[1/6] Performing System & External Dependency Audit...${RESET}"

DISTRO="unknown"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO="${ID:-unknown}"
fi

MISSING_DEPS=()
MISSING_RUST=false

# Check C/C++ compiler
if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    MISSING_DEPS+=("compiler")
fi

# Check CMake
if ! command -v cmake >/dev/null 2>&1; then
    MISSING_DEPS+=("cmake")
fi

# Check pkg-config
if ! command -v pkg-config >/dev/null 2>&1 && ! command -v pkgconf >/dev/null 2>&1; then
    MISSING_DEPS+=("pkg-config")
fi

# Check libdrm headers
if ! pkg-config --exists libdrm 2>/dev/null && [ ! -f /usr/include/libdrm/drm.h ] && [ ! -f /usr/include/drm/drm.h ]; then
    MISSING_DEPS+=("libdrm-dev")
fi

# Check OpenSSL headers (optional for HTTPS, highly recommended)
if ! pkg-config --exists openssl 2>/dev/null && [ ! -f /usr/include/openssl/ssl.h ]; then
    MISSING_DEPS+=("libssl-dev")
fi

# Check Python 3
if ! command -v python3 >/dev/null 2>&1; then
    MISSING_DEPS+=("python3")
fi

# Check pip
if ! command -v pip3 >/dev/null 2>&1 && ! command -v pip >/dev/null 2>&1; then
    MISSING_DEPS+=("python3-pip")
fi

# Check Rust / Cargo
if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
    if [ ! -f "$REPO_ROOT/target/release/libzero_copy_model_runner.a" ]; then
        MISSING_RUST=true
    fi
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Detected missing system dependencies: ${MISSING_DEPS[*]}${RESET}"
    
    PKG_INSTALL_CMD=""
    case "$DISTRO" in
        ubuntu|debian|pop|linuxmint|elementary)
            DEB_PKGS=()
            for dep in "${MISSING_DEPS[@]}"; do
                case "$dep" in
                    compiler) DEB_PKGS+=("build-essential") ;;
                    cmake) DEB_PKGS+=("cmake") ;;
                    pkg-config) DEB_PKGS+=("pkg-config") ;;
                    libdrm-dev) DEB_PKGS+=("libdrm-dev") ;;
                    libssl-dev) DEB_PKGS+=("libssl-dev") ;;
                    python3) DEB_PKGS+=("python3" "python3-venv") ;;
                    python3-pip) DEB_PKGS+=("python3-pip") ;;
                esac
            done
            PKG_INSTALL_CMD="apt-get update && apt-get install -y ${DEB_PKGS[*]}"
            ;;
        fedora|rhel|centos|rocky|almalinux)
            RPM_PKGS=()
            for dep in "${MISSING_DEPS[@]}"; do
                case "$dep" in
                    compiler) RPM_PKGS+=("gcc" "gcc-c++" "make") ;;
                    cmake) RPM_PKGS+=("cmake") ;;
                    pkg-config) RPM_PKGS+=("pkgconf-pkg-config") ;;
                    libdrm-dev) RPM_PKGS+=("libdrm-devel") ;;
                    libssl-dev) RPM_PKGS+=("openssl-devel") ;;
                    python3) RPM_PKGS+=("python3") ;;
                    python3-pip) RPM_PKGS+=("python3-pip") ;;
                esac
            done
            PKG_INSTALL_CMD="dnf install -y ${RPM_PKGS[*]}"
            ;;
        arch|manjaro|endeavouros)
            ARCH_PKGS=()
            for dep in "${MISSING_DEPS[@]}"; do
                case "$dep" in
                    compiler) ARCH_PKGS+=("base-devel") ;;
                    cmake) ARCH_PKGS+=("cmake") ;;
                    pkg-config) ARCH_PKGS+=("pkgconf") ;;
                    libdrm-dev) ARCH_PKGS+=("libdrm") ;;
                    libssl-dev) ARCH_PKGS+=("openssl") ;;
                    python3) ARCH_PKGS+=("python") ;;
                    python3-pip) ARCH_PKGS+=("python-pip") ;;
                esac
            done
            PKG_INSTALL_CMD="pacman -Sy --noconfirm ${ARCH_PKGS[*]}"
            ;;
        opensuse*|suse)
            SUSE_PKGS=()
            for dep in "${MISSING_DEPS[@]}"; do
                case "$dep" in
                    compiler) SUSE_PKGS+=("patterns-devel-base-devel_basis" "gcc-c++") ;;
                    cmake) SUSE_PKGS+=("cmake") ;;
                    pkg-config) SUSE_PKGS+=("pkg-config") ;;
                    libdrm-dev) SUSE_PKGS+=("libdrm-devel") ;;
                    libssl-dev) SUSE_PKGS+=("libopenssl-devel") ;;
                    python3) SUSE_PKGS+=("python3") ;;
                    python3-pip) SUSE_PKGS+=("python3-pip") ;;
                esac
            done
            PKG_INSTALL_CMD="zypper install -y ${SUSE_PKGS[*]}"
            ;;
        *)
            echo -e "${YELLOW}Could not determine package manager for distro '$DISTRO'. Please install missing packages manually.${RESET}"
            ;;
    esac

    if [ -n "$PKG_INSTALL_CMD" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "  [Dry-Run] Would execute: sudo $PKG_INSTALL_CMD"
        elif [ "$NON_INTERACTIVE" = true ] || [ "${EUID:-$(id -u)}" -eq 0 ]; then
            echo -e "  -> Automatically installing required system packages..."
            if [ "${EUID:-$(id -u)}" -eq 0 ]; then
                sh -c "$PKG_INSTALL_CMD"
            elif command -v sudo >/dev/null 2>&1; then
                sudo sh -c "$PKG_INSTALL_CMD"
            fi
        else
            echo -e "${CYAN}Would you like to install the missing packages now? [y/N]${RESET}"
            read -r -p "Run: sudo $PKG_INSTALL_CMD ? " response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                sudo sh -c "$PKG_INSTALL_CMD"
            else
                echo -e "${YELLOW}Proceeding without automated package installation. Builds may fail if headers are missing.${RESET}"
            fi
        fi
    fi
else
    echo -e "  -> ${GREEN}All required system build packages and headers are present.${RESET}"
fi

# Check / Install Rust
if [ "$MISSING_RUST" = true ]; then
    echo -e "\n${YELLOW}Rust toolchain (cargo/rustc) is missing.${RESET}"
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [Dry-Run] Would install Rust via https://sh.rustup.rs"
    elif [ "$NON_INTERACTIVE" = true ]; then
        echo -e "  -> Installing Rust toolchain via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo -e "${CYAN}Would you like to install Rust toolchain via rustup now? [y/N]${RESET}"
        read -r -p "Install rustup? " resp
        if [[ "$resp" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            echo -e "${RED}Error: Rust toolchain is required to compile libzero_copy_model_runner.${RESET}"
            exit 1
        fi
    fi
else
    echo -e "  -> ${GREEN}Rust toolchain is ready ($(rustc --version 2>/dev/null || echo 'Prebuilt'))${RESET}"
fi

# Install Python requirements
if [ "$INSTALL_PYTHON" = true ] && [ "$DRY_RUN" = false ]; then
    echo -e "\n${BOLD}[2/6] Checking Python Converter Dependencies...${RESET}"
    if [ -f "$REPO_ROOT/requirements.txt" ] && command -v python3 >/dev/null 2>&1; then
        PIP_CMD=""
        if command -v pip3 >/dev/null 2>&1; then
            PIP_CMD="pip3"
        elif command -v pip >/dev/null 2>&1; then
            PIP_CMD="pip"
        elif python3 -m pip --version >/dev/null 2>&1; then
            PIP_CMD="python3 -m pip"
        fi

        if [ -n "$PIP_CMD" ]; then
            echo -e "  -> Installing Python requirements from requirements.txt..."
            $PIP_CMD install -r "$REPO_ROOT/requirements.txt" --break-system-packages 2>/dev/null || \
            $PIP_CMD install -r "$REPO_ROOT/requirements.txt" --user 2>/dev/null || \
            $PIP_CMD install -r "$REPO_ROOT/requirements.txt" 2>/dev/null || \
            echo -e "${YELLOW}Notice: Non-critical python dependency install skipped. Use a venv for manual installation.${RESET}"
        fi
    fi
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "\n${GREEN}Dry-run completed successfully.${RESET}"
    exit 0
fi

# 3. Determine Destination Directory
INSTALL_DIR="/usr/local/bin"
if [ ! -w "$INSTALL_DIR" ]; then
    if [ "${EUID:-$(id -u)}" -ne 0 ]; then
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    fi
fi

echo -e "\n${CYAN}Target installation directory: ${BOLD}${INSTALL_DIR}${RESET}"

# 4. Build Rust APU Backend
echo -e "\n${BOLD}[3/6] Building Rust Hardware Acceleration Engine (libzero_copy_model_runner)...${RESET}"
cd "$REPO_ROOT"
if command -v cargo >/dev/null 2>&1; then
    RUSTFLAGS="-C target-cpu=native" cargo build --release --bins
else
    echo -e "${YELLOW}Warning: cargo not found. Checking for pre-compiled binaries...${RESET}"
    if [ ! -f "$REPO_ROOT/target/release/libzero_copy_model_runner.a" ]; then
        echo -e "${RED}Error: cargo is required to build the APU backend from source.${RESET}"
        exit 1
    fi
fi

# 5. Build C++ llama.cpp with APU Backend Enabled
echo -e "\n${BOLD}[4/6] Building C++ llama.cpp Front-End with Zero-Copy APU Backend...${RESET}"
if [ -d "$LLAMA_CPP_DIR" ] && [ -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]; then
    cd "$LLAMA_CPP_DIR"
    cmake -B build -DLLAMA_APU_BACKEND=ON -DAPU_BACKEND_DIR="$REPO_ROOT"
    cmake --build build --config Release -j"$(nproc)"
else
    echo -e "${YELLOW}Notice: llama.cpp directory not found at $LLAMA_CPP_DIR; skipping C++ build step.${RESET}"
fi

# 6. Install Binaries
echo -e "\n${BOLD}[5/6] Installing Executables to $INSTALL_DIR...${RESET}"

# Install Rust utilities
for tool in apu-doctor apu-model apu-synth; do
    SRC="$REPO_ROOT/target/release/$tool"
    if [ -f "$SRC" ]; then
        echo -e "  -> Installing ${GREEN}$tool${RESET}"
        cp "$SRC" "$INSTALL_DIR/$tool"
        chmod +x "$INSTALL_DIR/$tool"
    fi
done

# Install C++ binaries from llama.cpp build (including unified 'llama' multiplexer)
if [ -d "$LLAMA_CPP_DIR/build/bin" ]; then
    for bin in llama llama-cli llama-server apu-run llama-bench llama-quantize; do
        SRC="$LLAMA_CPP_DIR/build/bin/$bin"
        if [ -f "$SRC" ]; then
            echo -e "  -> Installing ${GREEN}$bin${RESET}"
            cp "$SRC" "$INSTALL_DIR/$bin"
            chmod +x "$INSTALL_DIR/$bin"
        fi
    done
fi

# 7. Install & Register XCLBIN Hardware Profiles
XCLBINS_DEST="/usr/local/share/llama-apu/xclbins"
if [ ! -w "/usr/local/share" ] && [ "${EUID:-$(id -u)}" -ne 0 ]; then
    XCLBINS_DEST="$HOME/.local/share/llama-apu/xclbins"
fi
echo -e "\n${BOLD}[6/6] Registering System Hardware Profiles & Configuring Services...${RESET}"
echo -e "  -> Target XCLBIN directory: ${GREEN}$XCLBINS_DEST${RESET}"
mkdir -p "$XCLBINS_DEST"
if [ -d "$REPO_ROOT/xclbins" ]; then
    cp -r "$REPO_ROOT/xclbins/"* "$XCLBINS_DEST/"
    echo -e "  -> Successfully registered $(ls -1 "$XCLBINS_DEST" | wc -l) XDNA 2 hardware profiles"
fi

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

# Validate User Hardware Permissions (render / video groups)
CURRENT_GROUPS=$(id -Gn 2>/dev/null || echo "")
if [[ "$CURRENT_GROUPS" != *"render"* ]] || [[ "$CURRENT_GROUPS" != *"video"* ]]; then
    echo -e "\n${YELLOW}Notice: Current user ($(whoami)) is missing 'render' or 'video' group membership.${RESET}"
    echo -e "To grant non-root zero-copy DMA access to AMDGPU and XDNA NPU devices, run:"
    echo -e "  ${BOLD}sudo usermod -a -G render,video $(whoami)${RESET}"
    echo -e "Then log out and log back in for changes to take effect."
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

# Post-install Hardware Diagnostic Probe
echo -e "\n${BOLD}Executing Hardware Diagnostic Verification:${RESET}"
if [ -x "$INSTALL_DIR/apu-doctor" ]; then
    "$INSTALL_DIR/apu-doctor" || true
fi
