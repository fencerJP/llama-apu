# Quickstart Guide: AMD Ryzen AI APU llama.cpp Backend

Welcome to the **AMD Ryzen AI APU Zero-Copy Backend for llama.cpp**. This guide provides end-to-end instructions for setting up, verifying, and running production inference on AMD Ryzen AI APU systems powered by AMD XDNA 2 silicon.

---

## 1. System Requirements & Prerequisites

### Hardware
- **Processor**: AMD Ryzen AI 300 Series APU with AMD XDNA 2 NPU (Strix Point, Gorgon Point, Krackan Point, or Strix Halo).
- **RAM**: Minimum 16 GB unified LPDDR5X/DDR5 system memory (32 GB+ recommended for models $\ge$ 7B).

### Software & Drivers
- **Operating System**: Linux (Ubuntu 24.04 LTS, Fedora 40+, Arch Linux, or openSUSE Tumbleweed).
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

The architecture couples the **upstream C++ `llama.cpp` frontend** with the **Rust `apu-backend` acceleration library**.

### Step 1: Build the Rust Acceleration Backend & Tooling
```bash
git clone https://github.com/amd/zero-copy-model-runner.git
cd zero-copy-model-runner

# Compile the native C ABI shared/static library and diagnostic utilities
RUSTFLAGS="-C target-cpu=native" cargo build --release
```
This builds:
- `target/release/libzero_copy_model_runner.so` (and `.a`)
- `target/release/apu-doctor` (Hardware environment diagnostic)
- `target/release/apu-model` (Turnkey GGUF/Q4NX converter and XCLBIN stamper)

### Step 2: Build Upstream `llama.cpp` with APU Backend
```bash
cd /path/to/llama.cpp
cmake -B build -DLLAMA_APU_BACKEND=ON
cmake --build build --config Release -j$(nproc)
```
This builds upstream `llama-cli`, `llama-server`, and the `apu-run` reference runner.

---

## 3. Verify Hardware with `apu-doctor`

Before launching inference, run `apu-doctor` to ensure your AMD APU hardware nodes, driver permissions, and instruction sets are operational:

```bash
./target/release/apu-doctor
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

## 5. Model Quantization & Ingestion (`apu-model`)

The runtime utilizes `.q4nx` containers designed for the 4-bit AIE2P tile microcode inside the XCLBIN.

### Converting GGUF Models (Supported Range: Q4 through Q16)
You can ingest models quantized across the supported **Q4 through Q16** spectrum (`Q4_0`, `Q4_K_M`, `IQ4_NL`, `IQ4_XS`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, `BF16`). Sub-4-bit quantizations (`IQ1`, `IQ2`, `Q3`) are explicitly not recommended and rejected due to severe perplexity loss (see [QUANTIZATION.md](QUANTIZATION.md) for the full breakdown):

```bash
# Convert an IQ4_NL GGUF model to an optimized .q4nx container
apu-model convert --input model-iq4_nl.gguf --output model.q4nx

# Pre-compile / stamp an XDNA 2 hardware execution graph
apu-model stamp --model model.q4nx --target gorgon-point
```

*Note: During `.q4nx` creation, `IQ4_NL` non-linear codebooks are dequantized and re-packed into the AIE2P 4-bit tile-interleaved memory layout, retaining the enhanced perplexity and fine-tuning accuracy of the non-linear quantization while running at full NPU hardware speed.*

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
