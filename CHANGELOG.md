# Changelog

All notable changes to the **llama-apu** project (AMD Ryzen AI APU Zero-Copy Backend) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-20

### Added
- **BiLLM (1.08 bpw) Quantization & SpinQuant Integration**:
  - Implemented 1-bit residual binarization pipeline (`converter/convert_to_billm.py`) with strict ordering: **SpinQuant offline orthogonal rotation strictly precedes Hessian computation and salient weight protection**.
  - Top 0.5% - 1.0% high-Hessian salient weights preserved in higher precision (INT4/FP16), while remaining 99% weights are binarized to {-1, +1}, preventing perplexity collapse at extreme 1-bit compression.
  - Native container ingestion for `BILLM`, `Q1_BILLM`, `Q1_0`, and `Q1_0_G128` formats in `src/container/converter.rs` and `src/container/reader.rs`.
- **T-MAC SRAM Lookup Tables for XDNA 2 AIE2P NPU**:
  - Implemented multiplication-free 1-bit GEMV on 32-tile AIE2P NPU using activation pre-computation into 32 KB tile SRAM lookup tables (LUTs).
  - Eliminates register inflation and INT4 unpacking overheads, achieving maximum memory bandwidth saturation on LPDDR5X UMA.
- **Full APU Stage Routing Overrides for 1-Bit Models**:
  - Seamless acceleration routing across APU silicon stages: `--tokenize {cpu,gpu,npu}`, `--prefill {gpu,cpu,npu}`, `--decode {npu,gpu,cpu}`.
  - Full support for accelerator presets: `--gpu-based`, `--cpu-based`, and `--npu-based`.
  - Added multi-engine fallback worker (`CpuWorkerEngine`) in `src/ffi.rs` to guarantee non-stop execution across any combination of stage overrides.
- **Extended Architecture Profiles & Metadata Capacity**:
  - Expanded GGUF container metadata key scan limit from 256 to 2048 keys in `src/container/xclbin_builder.rs` to support massive MoE architectures.
  - Added hardware topology profiles for Google Gemma 4 31B, Qwen3.8-27B Cold-Fusion, Qwen3-Coder-Next, Sarvam-105B, Laguna-S-2.1, Qwen3.8-Flash-Next, DeepSeek-V4-Flash variants, and GLM-5.3-Flash.
- **Single-Program & Single-Repository Unified Architecture**:
  - Unified Rust APU backend (`zero-copy_model_runner`) and C++ front-end into a single repository and integrated build system.
  - CMake now automatically compiles the Rust acceleration engine via Cargo during standard `cmake --build` invocations as a first-class dependency.
  - Retained and enhanced the core `llama` multiplexer (with `llama-apu` symlink/alias), ensuring 100% uninterrupted backward compatibility for programmatic callers (including Lemonade).
  - Integrated direct flag routing (`llama -m ... -p ...`) that auto-detects `.q4nx` zero-copy APU containers and `.gguf` models.
  - Embedded APU management commands directly into `llama` / `llama-apu`: `doctor` (hardware/driver diagnostics), `convert` (model conversion and stamping), `synth` (XCLBIN synthesis), and `run` (zero-copy inference).
- **Quantization Reference & Evaluation Matrix**:
  - Added comprehensive technical analysis in `docs/quantization_alternatives.md` comparing BiLLM, SpinQuant, BitNet b1.58, TQ1_0, TQ2_0, T-ACE, FLUTE, NanoQuant, and mitigations (QuaRot, ReSpinQuant, KronQ, OffQ, AYOT).
  - Documented projected decode throughputs (up to 262 tok/s on MoE models).

---

## [0.3.1] - 2026-09-19

### Added
- **Upstream Alignment & GGML Safety Guards**:
  - GGUF start-relative alignment and memory map validation.
  - GGML buffer allocation failure guards preventing unexpected crashes during heavy memory pressure.
  - Added F16 input support to CPU Fast Walsh-Hadamard Transform (FWHT).

---

## [0.3.0] - 2026-09-18

### Added
- **Quest Sparsity & Chunked KV Allocation**:
  - Hierarchical attention speculation and dynamic sliding-window KV cache management.
  - TriForce hierarchical speculative decoding support across heterogeneous accelerators.

---

## [0.2.3] - 2026-09-17

### Added
- **RDNA 3.5 MoE Tile Heuristic Optimization**:
  - Broadened MoE `ncols_opt` tile sizing in `ggml-cuda/mmq.cu` to include RDNA 3.5 architecture (`GGML_CUDA_CC_IS_RDNA3`), yielding +11% to +16% prefill speedup on Ryzen AI 9 HX 470 (Radeon 890M / gfx1150/1151).
- **CPU Heap Corruption & Cache Line Sizing Fix**:
  - Removed `std::hardware_destructive_interference_size` ambiguity in `ggml-cpu/ops.h` and disabled problematic PCH include ordering, preventing RoPE work-buffer undersizing and heap corruption on AVX-512 Zen 5 cores.
- **Enhanced Tensor Parallelism & Reasoning Parsers**:
  - Fixed split state and granularity for fused QKV attention layers on Gemma 4 and Qwen 3.5 architectures (`src/llama-model.cpp`).
  - Added forced `\n</think>` token injection upon reasoning budget expiration for Qwen3-Coder models (`common/parsers/qwen3-coder.cpp`).
  - Added `--version` build metadata reporting in `llama-bench`.
  - Added support for `HrmTextForCausalLM` (DFM Mimir 1B) and `Maple 20B-A1B` ternary MoE architecture.
  - Improved `im2col` strided memory access patterns for HIP/ROCm vision encoders.
  - Fixed MIMO-2 SWA sliding-window attention pattern loading and Nemotron-H `layer_norm_epsilon` handling.

---

## [0.2.2] - 2026-09-16

### Added
- **K2 Horizon Architecture Support**:
  - Native compute graph implementation (`src/models/k2-horizon.cpp`) supporting dense and MoE configurations (K2-Horizon-7B and K2-Horizon-32B).
  - Architecture registration in `src/llama-arch.cpp` and `src/llama-arch.h`.
  - Tokenizer and vocabulary support (`src/llama-vocab.cpp`, `src/llama-vocab.h`, `models/templates/k2-horizon.jinja`).
  - Hugging Face to GGUF conversion pipeline (`conversion/k2_horizon.py`).
  - Zero-copy `.q4nx` container packaging with embedded XDNA 2 hardware binary bindings.
- **Upstream ROCm & AMD APU Performance Improvements**:
  - `llama`: Disabled lazy tensor loading by default on iGPUs for unified memory stability (`#28326`).
  - `HIP`: Enabled FP32 accumulation in `fattn-mma` on MFMA devices for numerical accuracy (`#28576`).
  - `HIP`: Enabled AllReduce for ROCm backends (`#27825`).
  - `CUDA/HIP`: Flash Attention kernel tuning for `gfx1201` (`#28102`).
  - `memory`: Avoided allocating V cache for indexer when unused, reducing memory footprint (`#28330`).
  - `model`: Fixed MTP context KV cache allocation for DeepSeek-V2 and GLM4-MoE (`#28630`).
  - `server`: Fixed LRU cache hang on concurrent multiple requests for the same model (`#28539`).
  - `server`: Allowed model downloads at model limit (`#28530`).
  - `webui`: Stopped re-probing disabled `/tools` endpoint on every message (`#28646`).
  - `jinja`: Treated null left operand of `in` as plain lookup (`#28620`).
  - `vendor`: Updated `cpp-httplib` to `0.56.0` (`#28787`).
  - `llama`: Used `int32_t` for `llama_sampler_chain_n` return type (`#28631`).

---

## [0.2.1] - 2026-09-15

### Added
- **Native `llama` Multiplexer Binary Integration**: Bundled and installed the native compiled upstream `app/llama.cpp` multiplexer binary to seamlessly dispatch `llama cli`, `llama serve`, `llama bench`, `llama quantize`, `llama download`, and `llama completion` without requiring symlinks.
- **Repository Cleanliness & Privacy**: Configured strict `.gitignore` patterns ensuring no internal metadata or transient files are tracked.

---

## [0.2.0] - 2026-09-15

### Added
- **Production XDNA 2 Hardware Binary Bank**: Bundled 37 pre-compiled production and experimental `.xclbin` profiles in `xclbins/` for Strix Point, Gorgon Point, Krackan Point, and Strix Halo.
- **System-Wide XCLBIN Auto-Discovery**: Runtime container resolver and installer now automatically look up and register hardware profiles in `/usr/local/share/llama-apu/xclbins`, `~/.local/share/llama-apu/xclbins`, and `LLAMA_APU_XCLBINS_DIR`.

---

## [0.1.0] - 2026-09-15

### Added
- **AMD Ryzen AI Zero-Copy APU Backend (`apu-backend`)**:
  - Direct Linux Prime `dma-buf` cross-accelerator memory handoff between RDNA 3.5 iGPU and AMD XDNA 2 NPU (AIE2P).
  - Explicit Linux DRM synchronization objects (`drm_syncobj`) and timeline fence tracking without CPU spinloops.
  - Unified `TransformerContext` binding shared KV cache buffers in physical LPDDR5X DRAM with 64-byte cacheline alignment.
- **Heterogeneous Pipeline Routing**:
  - Batched GEMM prompt prefill offloaded to RDNA 3.5 iGPU (`gfx1150`).
  - Autoregressive single-token GEMV decode streamed through 32 AIE2P spatial tiles on XDNA 2 NPU (`/dev/accel/accel0`).
  - Native Zen 5 AVX-512 SIMD vectorization for sampling and token operations.
- **Granular Accelerator CLI Flags**:
  - `--tokenize {cpu,gpu,npu}`: Override prompt tokenization target.
  - `--prefill {gpu,cpu,npu}`: Override compute-heavy prefill forward pass target.
  - `--decode {npu,gpu,cpu}`: Override autoregressive decode loop target.
  - `--gpu-based`, `--cpu-based`, `--npu-based`: Macro accelerator presets.
  - `--apu-xclbin <PATH>`: Override path to compiled XCLBIN hardware graph microcode.
  - `--apu-verbose`: Enable detailed DMA-BUF allocations and DRM timeline fence telemetry.
- **Hardware Diagnostics & Tooling**:
  - `apu-doctor`: Probes CPU AVX-512, `/dev/dri/renderD128`, `/dev/kfd`, `/dev/accel/accel0`, ROCm runtime, and user group permissions.
  - `apu-model`: Subcommands `info`, `convert`, and `stamp` for GGUF metadata inspection, Q4–Q16 format conversion, and turnkey `.q4nx` container packaging with embedded XCLBINs.
  - `apu-run`: Standalone C++ heterogeneous CLI runner linking `libzero_copy_model_runner.a`.
- **Quantization Support Matrix (Q4 to Q16)**:
  - Full ingestion of `Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `IQ4_NL` (non-linear codebook mapping), `IQ4_XS`, `Q5_0`, `Q5_1`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, `F16`, `BF16`, `F32`.
  - Descriptive rejection of non-recommended sub-4-bit formats (`IQ1_*`, `IQ2_*`, `Q2_K`, `IQ3_*`, `Q3_K_*`) that break AIE2P tile alignment.
- **Upstream `llama.cpp` Integration**:
  - Seamless drop-in compatibility with `llama-cli` and `llama-server` (OpenAI REST API with SSE streaming).
- **Silicon Architecture Support**:
  - AMD Strix Point (Ryzen AI 9 HX 370 / 365).
  - AMD Gorgon Point (Ryzen AI 9 HX 470).
  - AMD Krackan Point (Ryzen AI 7).
  - AMD Strix Halo (Ryzen AI Max+ 395) with 256-bit UMA hugepage memory bus tuning.
- **Packaging & Deployment Automation**:
  - `scripts/install.sh`: Unified single-command installer.
  - `scripts/package_release.sh`: Self-contained tarball generator with SHA-256 verification.
  - `scripts/99-amdxdna-apu.rules`: Udev access rules for non-root hardware nodes.
  - `scripts/llama-server.service`: Hardened systemd daemon configuration.
  - `.github/workflows/ci.yml` and `release.yml`: GitHub Actions continuous integration and automated release pipelines.

### Preliminary Benchmarks
- Included "Small-scale preliminary xclbin comparison test results" across 10 neural model families comparing Vendor Built-in, Custom Enhanced (64MB SRAM), and Custom Mimic XCLBINs on physical AMD Ryzen AI 9 HX 470 APU hardware.
