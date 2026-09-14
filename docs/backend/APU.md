# AMD Ryzen AI APU Zero-Copy Backend for llama.cpp

This document describes the **AMD Ryzen AI APU Zero-Copy Backend (`apu-backend`)** integrated into `llama.cpp`.

---

## Overview

The APU backend provides zero-copy heterogeneous execution on **AMD Ryzen AI APUs** (AMD XDNA 2 AIE2P Silicon):
- **Prompt Prefill (Batched GEMM)**: Offloaded to the **RDNA 3.5 iGPU** via ROCm / HIP.
- **Autoregressive Token Decode (GEMV)**: Executed across the 32 AIE2P tiles on the **AMD XDNA 2 NPU** (`/dev/accel/accel0`).
- **Unified Memory Handoff**: Cross-accelerator KV cache handoff through Linux Prime `dma-buf` without host RAM copying.
- **Fence Synchronization**: Synchronized via Linux DRM synchronization objects (`drm_syncobj`) and timeline fences.

---

## Supported Hardware

| Silicon Family | iGPU (Prefill) | NPU (Decode) | Memory Bandwidth |
| :--- | :--- | :--- | :--- |
| **AMD Strix Point** (HX 370 / 365) | 16 CUs RDNA 3.5 | 32-tile AIE2P (50 TOPS) | 136 GB/s LPDDR5X |
| **AMD Gorgon Point** (HX 470) | 16 CUs RDNA 3.5 | 32-tile AIE2P (55 TOPS) | 136 GB/s LPDDR5X |
| **AMD Krackan Point** (Ryzen AI 7) | 8 CUs RDNA 3.5 | 16-tile AIE2P (32 TOPS) | 120 GB/s LPDDR5X |
| **AMD Strix Halo** (MAX+ 395) | 40 CUs RDNA 3.5 | 32-tile AIE2P (55 TOPS) | 273 GB/s (256-bit) |

*Note: Legacy AMD XDNA 1 processors (Phoenix / Hawk Point) lack BlockFP16 hardware and are formally unsupported.*

---

## Build Instructions

```bash
# Build the Rust apu-backend library
cd ../../zero-copy_model_runner
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Build llama.cpp with APU backend
cd ../llamacpp-update/llama.cpp
cmake -B build -DLLAMA_APU_BACKEND=ON
cmake --build build --config Release -j$(nproc)
```

---

## User-Facing Parameters & Runtime Options

### APU Stage Overrides & Presets

| Flag | Type / Acceptable Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `--tokenize` | `cpu`, `gpu`, `npu` | `cpu` | Target accelerator for prompt tokenization and subword splitting. |
| `--prefill` | `gpu`, `cpu`, `npu` | `gpu` | Target accelerator for prompt evaluation / prefill (RDNA 3.5 iGPU). |
| `--decode` | `npu`, `gpu`, `cpu` | `npu` | Target accelerator for autoregressive decode (XDNA 2 NPU). |
| `--gpu-based` | Preset flag | `disabled` | Run all stages on RDNA 3.5 iGPU (`--tokenize gpu --prefill gpu --decode gpu`). |
| `--cpu-based` | Preset flag | `disabled` | Run all stages on host Zen 5 CPU (`--tokenize cpu --prefill cpu --decode cpu`). |
| `--npu-based` | Preset flag | `disabled` | Run all stages on XDNA 2 NPU (`--tokenize npu --prefill npu --decode npu`). |
| `--apu-xclbin <PATH>` | File path (`.xclbin`) | Auto-resolved | Override path to XCLBIN hardware graph microcode. |
| `--apu-verbose` | Boolean flag | `false` | Enable detailed zero-copy DMA-BUF memory and timeline fence telemetry. |

### Quantization Support Spectrum (Q4 to Q16)

- **Supported Formats**: `Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `IQ4_NL`, `IQ4_XS`, `Q5_0`, `Q5_1`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, `F16`, `BF16`, `F32`.
- **Unsupported Policy**: Sub-4-bit formats (`IQ1_*`, `IQ2_*`, `Q2_K`, `IQ3_*`, `Q3_K_*`) are explicitly rejected due to severe perplexity loss and unaligned memory strides breaking AIE2P tile DMAs.

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

All runs exhibited 100% stability, zero memory leaks, and complete response intelligibility.
