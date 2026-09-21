# Quickstart Guide: AMD Ryzen AI APU llama.cpp Backend

Welcome to the **AMD Ryzen AI APU Zero-Copy Backend for llama.cpp**. This guide provides end-to-end instructions for setting up, verifying, and running production inference on AMD Ryzen AI APU systems powered by AMD XDNA 2 silicon.

---

## 1. System Requirements & Prerequisites

### Hardware
- **Processor**: AMD Ryzen AI 300 Series APU with AMD XDNA 2 NPU (Strix Point, Gorgon Point, Krackan Point, or Strix Halo).
- **RAM**: Minimum 16 GB unified LPDDR5X/DDR5 system memory (32 GB+ recommended for models $\ge$ 7B).

### Software & Drivers
- **Operating System**: Linux (Ubuntu 24.04 LTS, Debian 12+, Fedora 40+, Arch Linux, or openSUSE Tumbleweed).
- **Linux Kernel**: 6.10 or newer (with AMDXDNA NPU driver `amdxdna.ko` enabled).
- **ROCm / HIP**: ROCm 6.2+ or 7.0+ installed (for RDNA 3.5 iGPU acceleration).
- **Device Nodes & Permissions**:
  - `/dev/dri/renderD128` (AMDGPU iGPU render node)
  - `/dev/kfd` (ROCm compute interface)
  - `/dev/accel/accel0` (XDNA 2 NPU accelerator node)

Ensure your user account belongs to the `render`, `video`, and `kfd` groups:
```bash
sudo usermod -aG render,video,kfd $USER
# Log out and back in, or run:
newgrp render
```

---

## 2. Installation & Build

### Option A: Turnkey Automated Installer (Recommended)
The unified installer automatically audits system dependencies (across `apt`, `dnf`, `pacman`, and `zypper`), provisions missing compilers and headers (`libdrm-dev`, `libssl-dev`, `cmake`), configures the Rust and Python environments, compiles both Rust and C++ components, registers XDNA 2 hardware profiles, and installs systemd services:

```bash
git clone https://github.com/amd/zero-copy-model-runner.git
cd zero-copy-model-runner

# Interactive or automated installation
./scripts/install.sh -y
```

### Option B: Pre-compiled Release Bundle
Download the pre-compiled binary release bundle:
```bash
tar -xzf llama-apu-0.4.0-linux-x86_64.tar.gz
cd llama-apu-0.4.0-linux-x86_64
sudo ./install.sh
```

### Option C: Manual Build from Source
```bash
# 1. Build the Rust APU acceleration backend & CLI utilities
cd zero-copy_model_runner
RUSTFLAGS="-C target-cpu=native" cargo build --release --bins

# 2. Build upstream llama.cpp frontend with APU backend enabled
cd ../llamacpp-update/llama.cpp
cmake -B build -DLLAMA_APU_BACKEND=ON -DAPU_BACKEND_DIR=$(pwd)/../zero-copy_model_runner
cmake --build build --config Release -j$(nproc)
```

This compiles:
- `target/release/libzero_copy_model_runner.so` (and `.a`)
- `target/release/apu-doctor` (Hardware environment diagnostic)
- `target/release/apu-model` (Turnkey GGUF/Q4NX inspector and converter)
- `target/release/apu-synth` (Standalone XCLBIN hardware graph synthesizer)
- `llama`, `llama-cli`, `llama-server`, `apu-run`, `llama-bench`, `llama-quantize`

---

## 3. Verify Hardware with `apu-doctor`

Before launching inference, run `apu-doctor` to verify AMD APU hardware nodes, driver permissions, and instruction sets:

```bash
apu-doctor
```

Expected diagnostic output:
```text
============================================================
 AMD Ryzen AI APU Hardware & Runtime Diagnostic (Doctor)
============================================================
[ PASS ] Zen 5 CPU & AVX-512 SIMD: AMD Ryzen AI 9 HX 470 w/ Radeon 890M (AVX-512: Supported)
[ PASS ] AMDGPU DRM Render Node (/dev/dri/renderD128): Present
[ PASS ] AMDGPU KFD Compute Node (/dev/kfd): Present
[ PASS ] AMD XDNA 2 NPU Node (/dev/accel/accel0): Present (AMD XDNA 2 AIE2P Silicon)
[ PASS ] ROCm / HIP Runtime Stack: Path: /opt/rocm (7.2.4)
[ PASS ] User Hardware Permissions (render, video groups): render: OK, video: OK
============================================================
Status: All checks passed. System ready for zero-copy APU inference!
============================================================
```

---

## 4. Running Upstream `llama-cli` & `apu-run`

### Standard Upstream CLI with Native Accuracy
```bash
llama-cli -m /models/qwen2.5-0.5b-instruct-q8_0.gguf -p "What is the capital of France?" -n 32
```
Output:
> **The capital of France is Paris.**

### Turnkey Heterogeneous Execution (`apu-run`)
`apu-run` coordinates prompt prefill on the RDNA 3.5 iGPU and decode on the XDNA 2 NPU:
```bash
apu-run -m /models/my-model.q4nx -p "Explain quantum computing." -n 64 --verbose
```

---

## 5. Model Quantization & Ingestion

### BiLLM Direct-to-Disk Streaming Quantizer
```bash
# Install Python dependencies
pip install -r requirements.txt

# Stream and quantize Safetensors into 1.08 bpw BiLLM .q4nx container with orthogonal rotation
python3 converter/convert_to_billm.py --model-id /path/to/safetensors/ --output /models/model.q4nx
```

### Synthesizing Custom XCLBIN Graphs (`apu-synth`)
```bash
# Synthesize custom NPU2 AIE2P execution graph directly from model topology
apu-synth /models/model.q4nx /usr/local/share/llama-apu/xclbins/custom_model.xclbin enhanced
```

---

## 6. Running the OpenAI-Compatible Server (`llama-server`)

Launch the upstream `llama-server` with APU acceleration enabled:
```bash
llama-server -m /models/qwen2.5-0.5b-instruct-q8_0.gguf --host 0.0.0.0 --port 8080 -c 4096
```

### Querying via `curl`
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What are the advantages of an APU?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

---

## 7. Troubleshooting

| Issue / Error | Cause | Resolution |
| :--- | :--- | :--- |
| `Permission denied: /dev/kfd` or `/dev/accel/accel0` | User lacks group permissions | Run `sudo usermod -aG render,video,kfd $USER` and log back in. |
| `UnsupportedSiliconError: XDNA 1 detected` | Machine uses older Phoenix / Hawk Point silicon | XDNA 1 is deprecated. Run with CPU AVX-512 or upgrade to XDNA 2. |
| `Out of memory: GEM allocation failed` | Context window (`-c`) exceeds available UMA RAM | Lower `-c 2048` or switch to a smaller model size. |
| `Missing package headers (e.g. libdrm-dev)` | Build dependencies not installed | Run `./scripts/install.sh -y` to auto-install dependencies for your distro. |
