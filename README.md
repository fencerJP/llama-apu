# llama-apu: AMD Ryzen AI APU Zero-Copy Runtime for llama.cpp

[![License: MIT / Apache 2.0](https://img.shields.io/badge/License-MIT_%2F_Apache_2.0-blue.svg)](LICENSE)
[![Target Hardware](https://img.shields.io/badge/Target-AMD_Ryzen_AI_(XDNA_2_+_RDNA_3.5)-red.svg)](HARDWARE_SUPPORT.md)
[![Upstream Fork](https://img.shields.io/badge/Upstream_Fork-ggml--org%2Fllama.cpp-brightgreen.svg)](https://github.com/ggml-org/llama.cpp)
[![Release](https://img.shields.io/github/v/release/fencerJP/llama-apu?color=orange)](https://github.com/fencerJP/llama-apu/releases)

**`llama-apu`** is a specialized, production-grade fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) engineered exclusively for **AMD Ryzen AI APUs** (AMD Strix Point, Gorgon Point, Krackan Point, and Strix Halo). 

It introduces a high-performance **heterogeneous zero-copy runtime architecture** that orchestrates compute-heavy prompt prefill on the **RDNA 3.5 iGPU**, memory-bound autoregressive decode across the **XDNA 2 NPU (32-tile AIE2P)**, and SIMD sampling on **Zen 5 AVX-512 CPU cores**, unified through kernel `dma-buf` memory handoffs and DRM timeline fences.

---

## Upstream llama.cpp vs. llama-apu: Key Architectural Differences

While upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) focuses on generic CPU and discrete GPU offloading across diverse platforms, **`llama-apu`** custom-tailors the execution pipeline specifically for AMD Unified Memory Architecture (UMA) APU silicon:

| Feature / Subsystem | Upstream `llama.cpp` | `llama-apu` (This Fork) |
| :--- | :--- | :--- |
| **Pipeline Acceleration** | Single accelerator (CPU or GPU) or static layer split | **Phase-partitioned heterogeneous pipeline**: Compute-heavy Prefill on iGPU $\to$ Memory-bound Decode on NPU $\to$ Sampling on AVX-512 CPU |
| **Cross-Device Handoff** | Host-side memory copies (`memcpy` through host RAM) | **Zero-Copy Linux Prime `dma-buf`**: Direct cross-accelerator KV cache binding in physical LPDDR5X DRAM |
| **Synchronization** | CPU polling and host-side synchronization | **Kernel DRM timeline fences (`drm_syncobj`)**: Non-blocking asynchronous hardware execution without CPU spinloops |
| **NPU (XDNA 2) Offload** | Unsupported or generic NPU shims | **Native XDNA 2 AIE2P tile streaming**: Optimized 32-tile dataflow execution via `/dev/accel/accel0` |
| **Hardware Microcode** | None / manual external setup | **Bundled 37-profile XDNA 2 XCLBIN bank**: Auto-resolved and auto-discovered in system search paths |
| **CLI & Stage Overrides** | Generic `-ngl` / `-t` flags | **Granular stage routing**: `--tokenize`, `--prefill`, `--decode`, `--gpu-based`, `--cpu-based`, `--npu-based`, `--apu-xclbin`, `--apu-verbose` |
| **Quantization Policy** | Sub-1-bit to 8-bit generic quants | **Q4–Q16 alignment matrix**: Native hardware support for Q4 through Q16 (`Q4_K_M`, `IQ4_NL`, `Q5_K_M`, `Q8_0`, `BF16`, `F16`), rejecting unaligned sub-4-bit quants that break tile memory strides |
| **Hardware Diagnostics** | External or ad-hoc scripts | **Integrated `apu-doctor` & `apu-model`**: Built-in verification for kernel nodes (`renderD128`, `accel0`), permissions, and model inspection |

---

## Supported AMD Silicon Matrix

| Silicon Family | iGPU Prefill Engine | NPU Decode Engine | UMA Bandwidth | Target Profile |
| :--- | :--- | :--- | :--- | :--- |
| **AMD Strix Point** (HX 370 / 365) | 16 CUs RDNA 3.5 | 32-tile AIE2P (50 TOPS) | 136 GB/s LPDDR5X | Prefill: iGPU $\to$ Decode: NPU |
| **AMD Gorgon Point** (HX 470) | 16 CUs RDNA 3.5 | 32-tile AIE2P (55 TOPS) | 136 GB/s LPDDR5X | Prefill: iGPU $\to$ Decode: NPU |
| **AMD Krackan Point** (Ryzen AI 7) | 8 CUs RDNA 3.5 | 16-tile AIE2P (32 TOPS) | 120 GB/s LPDDR5X | Prefill: iGPU $\to$ Decode: NPU |
| **AMD Strix Halo** (MAX+ 395) | 40 CUs RDNA 3.5 | 32-tile AIE2P (55 TOPS) | 273 GB/s (256-bit) | Prefill: iGPU $\to$ Decode: iGPU / NPU |

*For complete architectural specifications and kernel driver requirements, see [HARDWARE_SUPPORT.md](HARDWARE_SUPPORT.md).*

---

## Quick Start

### 1. Installation

#### Option A: Pre-compiled Release Bundle (Recommended)
Download the latest pre-compiled bundle from the [Releases page](https://github.com/fencerJP/llama-apu/releases):
```bash
tar -xzf llama-apu-0.2.1-linux-x86_64.tar.gz
cd llama-apu-0.2.1-linux-x86_64
sudo ./install.sh
```

#### Option B: Build from Source
```bash
# 1. Build the Rust APU backend engine
cd zero-copy_model_runner
RUSTFLAGS="-C target-cpu=native" cargo build --release

# 2. Build the C++ frontend with APU backend enabled
cd ../llamacpp-update/llama.cpp
cmake -B build -DLLAMA_APU_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 2. Verify Hardware Environment
```bash
apu-doctor
```

### 3. Run Inference with the Native Multiplexer
```bash
# Single-prompt generation
llama cli -m /path/to/model.gguf -p "Explain zero-copy memory architecture." -n 128

# Interactive conversation mode
llama cli -m /path/to/model.gguf -cnv

# Dedicated binary syntax is also supported
llama-cli -m /path/to/model.gguf -p "What is the capital of France?" -n 64
```

### 4. Launch OpenAI-Compatible API Server
```bash
llama serve -m /path/to/model.gguf --host 0.0.0.0 --port 8080 -c 4096
```

---

## APU Stage Controls & Runtime Flags

| Flag | Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `--tokenize` | `cpu`, `gpu`, `npu` | `cpu` | Selects accelerator for prompt tokenization and subword splitting. |
| `--prefill` | `gpu`, `cpu`, `npu` | `gpu` | Selects accelerator for batched GEMM prompt evaluation (RDNA 3.5 iGPU). |
| `--decode` | `npu`, `gpu`, `cpu` | `npu` | Selects accelerator for memory-bound token generation (XDNA 2 NPU). |
| `--gpu-based` | Preset flag | `disabled` | Macro preset: routes tokenization, prefill, and decode to the RDNA 3.5 iGPU. |
| `--cpu-based` | Preset flag | `disabled` | Macro preset: routes all stages to host Zen 5 CPU cores via AVX-512. |
| `--npu-based` | Preset flag | `disabled` | Macro preset: routes execution across the XDNA 2 NPU. |
| `--apu-xclbin <PATH>` | File path (`.xclbin`) | Auto-resolved | Explicit override for XDNA 2 hardware microcode profile. |
| `--apu-verbose` | Boolean | `false` | Enables real-time DMA-BUF buffer allocation and DRM timeline fence telemetry. |

*See [CLI_GUIDE.md](CLI_GUIDE.md) for full parameter specifications and REST API documentation.*

---

## Small-scale preliminary xclbin comparison test results

We evaluated 10 neural model configurations across 3 XCLBIN hardware binary variants (Vendor Built-in, Custom Enhanced with 64MB SRAM, Custom Mimic) on host AMD Ryzen AI silicon (**AMD Ryzen AI 9 HX 470 APU**) using a standardized 180-token systems architecture prompt.

| Model Name | Model ID | Built-in TTFT | Built-in t/s | Enhanced TTFT | Enhanced t/s | Mimic TTFT | Mimic t/s | Output Quality & Acceptability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen2.5-0.5B-Instruct** | `qwen2.5-0.5b` | 6,032.8 ms | 14.3 t/s | 6,052.6 ms | 15.2 t/s | 6,388.9 ms | 14.1 t/s | **Acceptable (High)**: Structured PUM breakdown |
| **Llama-3.2-1B-Instruct** | `llama-3.2-1b` | 12,676.1 ms | 7.3 t/s | 12,587.4 ms | 8.0 t/s | 12,413.8 ms | 8.3 t/s | **Acceptable (High)**: Coherent systems intro |
| **Gemma-2-2B-IT** | `gemma-2-2b` | 3,692.6 ms | 13.4 t/s | 3,685.3 ms | 13.2 t/s | 3,550.9 ms | 12.7 t/s | **Acceptable (High)**: Direct technical headers |
| **Spark-X2.5-1.7B** | `spark-x2.5-1.7b` | 18,627.5 ms | 4.8 t/s | 19,191.9 ms | 4.6 t/s | 18,269.2 ms | 4.8 t/s | **Acceptable (Coherent)**: Chain-of-thought analysis |
| **K2-Horizon-1B-BF16** | `k2-horizon-1b` | 13,188.4 ms | 5.9 t/s | 13,382.4 ms | 6.1 t/s | 12,638.9 ms | 6.2 t/s | **Acceptable (Coherent)**: Systems architecture reasoning |
| **Qwen3.5-0.8B-Q4_K_M** | `qwen3.5-0.8b` | 584.7 ms | 40.0 t/s | 582.1 ms | 39.7 t/s | 545.4 ms | 43.0 t/s | **Acceptable (Coherent)**: High-speed linear attention |
| **Qwen2.5-3B-Instruct** | `qwen2.5-3b` | 4,044.0 ms | 9.9 t/s | 3,991.3 ms | 9.5 t/s | 3,931.6 ms | 9.8 t/s | **Acceptable (Coherent)**: Detailed comparative analysis |
| **Llama-3.2-3B-Instruct** | `llama-3.2-3b` | 3,838.0 ms | 9.2 t/s | 3,719.0 ms | 9.7 t/s | 3,592.8 ms | 10.8 t/s | **Acceptable (High)**: Comprehensive technical breakdown |
| **Gemma-4-E4B** | `gemma-4-e4b` | 6,026.1 ms | 7.8 t/s | 5,799.4 ms | 8.2 t/s | 5,745.3 ms | 8.3 t/s | **Acceptable (Coherent)**: Structured planning and points |
| **DeepSeek-R1-0528-Qwen3-8B** | `deepseek-r1-qwen3-8b` | 5,575.8 ms | 4.9 t/s | 5,768.0 ms | 5.0 t/s | 5,878.6 ms | 5.0 t/s | **Acceptable (Fluent)**: Detailed reasoning process |

---

## Documentation Index

- [CLI Reference Guide](CLI_GUIDE.md): Full command-line options, stage flags, and OpenAI API endpoint documentation.
- [AMD APU Developer & Architecture Reference Guide](docs/AMD_APU_DEVELOPER_REFERENCE.md): In-depth guide covering silicon architecture, `.xclbin` graph compilation, Linux kernel UAPI (`dma-buf`), and timeline fences.
- [Quantization Matrix](QUANTIZATION.md): Supported Q4–Q16 format specifications and sub-4-bit rejection policies.
- [Hardware Support Matrix](HARDWARE_SUPPORT.md): Per-silicon architecture breakdown, driver nodes, and memory subsystem tuning.
- [Changelog](CHANGELOG.md): Version history, updates, and release notes.
- [APU Architecture Guide](docs/backend/APU.md): Deep-dive into DMA-BUF memory bridges and DRM timeline fences.

---

## Attribution & License

- **Original Project**: This software is based on and derived from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors, licensed under the [MIT License](LICENSE).
- **APU Zero-Copy Backend**: Developed under Apache License, Version 2.0 / MIT.
- **Attribution Policy**: Upstream file naming and core C/C++ data structures (`ggml.h`, `ggml_tensor`) are maintained intact as proper open-source attribution.
