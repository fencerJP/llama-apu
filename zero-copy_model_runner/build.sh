#!/usr/bin/env bash
# One-command build script for AMD Ryzen AI APU llama.cpp
set -euo pipefail
echo "Building AMD Ryzen AI APU Zero-Copy Model Runner & llama.cpp..."
RUSTFLAGS="-C target-cpu=native" cargo build --release
if [ -d "../llamacpp-update/llama.cpp" ]; then
    cd ../llamacpp-update/llama.cpp
    cmake -B build -DLLAMA_APU_BACKEND=ON
    cmake --build build --config Release -j$(nproc)
    echo "Build complete! Binaries located in: $(pwd)/build/bin/"
fi
