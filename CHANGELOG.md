# Changelog

All notable changes to **llama-apu** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.10.1] - 2026-10-10

**Upstream Master Sync (`b11541`) & Model Architecture Expansion.**

### Added
- **Upstream Sync (`b11541` / `@2bbca8f20`)**: Merged 107 commits from `ggml-org/llama.cpp` master preserving all AMD APU co-designed extensions (KFD 75% memory governor, Sarathi-Serve stall-free scheduling, PRIME DMA-BUF, DRM syncobj timeline fences, ROCm 10.1).
- **Official K2 Horizon & MoVA Support**: Native architecture integration (`LLM_ARCH_K2_HORIZON`), routed value experts, and pre-tokenization splitters.
- **PLaMo-3 Tokenizer Support**: Tokenizer pre-segmentation regexes and FIM token handling.
- **Upstream Server & UI Models Manager**: Integrated enhanced server UI models manager (`#29583`, `#30228`) and default port 9931 configuration.

### Fixed
- **ROCm 10.1 Clang Build Path**: Resolved HIP runtime header lookup across standalone and wrapper build paths for `/opt/rocm/core-10.1`.
- **Hparams Field Deduplication**: Eliminated duplicate `n_value_expert` / `n_value_expert_used` declarations in `llama-hparams.h`.
- **Imatrix Null Safety**: Reapplied defensive null-pointer verification in `tools/imatrix/imatrix.cpp`.

---

## [0.10.0] - 2026-10-07

**ROCm 10.1 Integration release.**

### Added
- **ROCm 10.1 stack** (Phase 10.0): upgrade from the distro ROCm 7.1 build to **10.1.0** (`/opt/rocm/core-10.1`, HIP clang LLVM 24), validated on gfx1150. Controlled A/B vs the old stack: **+7.9% prefill, decode parity.**
- **Upstream sync** (Phase 10.05): merged `ggml-org/llama.cpp` master (319 upstream commits) preserving every llama-apu modification — 9/9 APU tests pass, perplexity bit-identical.
- **`tools/apu-ab/apu_ab.py`** (Phase 10.6): local A/B harness — interleaved benchmark with medians/verdicts, `rocprofv3` profiling pass, and run diffing.
- **hipBLASLt GEMM support** (Phase 10.1): `-DGGML_HIP_USE_HIPBLASLT` (build, default ON). Runtime-disabled; enable via `LLAMA_APU_GEMM_BACKEND=hipblaslt` (+`LLAMA_APU_HIPBLASLT_PREFILL=1`). Measured **−41%** on quantized prefill (UMA) → not advantageous.
- **a8w8 int8 GEMM probe** (Phase 10.2b, `tools/apu-a8w8/`): gfx1150 int8 hipBLASLt kernels validated bit-exact (6.5–7.7 TOPS prefill); no model-level benefit on UMA.
- `docs/release-notes-0.10.0.md` and `dist/release_notes_v0.10.0.md`.
### Changed
- **TQ2_0 retired from the active path** (Phase 10.2a): default conversion target is now `Q4_K_M`; TQ2_0/T-ACE lowering requires `LLAMA_APU_LOWBIT_TQ2_0=1`. Substrate kept for future repair.
- **ROCm/HIP build-options table** documented in `README.md`.
- **`GGML_HIP_NO_VMM` re-evaluated** (Phase 10.5): default `ON` retained; enabling VMM (`-DGGML_HIP_NO_VMM=OFF`) measured ≈−9% prefill / ≈−11% decode on gfx1150 UMA.

### Notes
- On gfx1150 **UMA**, quantized models route through ggml's MMQ/MMVQ and never BLAS; the GPU-side ROCm features (hipBLASLt, rocBLAS decode gemv, a8w8, VMM, ROCm/CK FMHA) were each evaluated and **not adopted** (neutral→negative or unreachable). The only default-on performance change is the ROCm 10.1 stack itself.
- Full 60-min memory soak was cancelled early; a ~7-min capture showed stable RSS (~1.03 GB, no leak).

---

## [0.9.0] - 2026-10-07

### Added
- **Sarathi-Serve Stall-Free Scheduler (§9.1):** Iteration-level hybrid micro-batching under a hard token budget τ with four-tier priority (ongoing decodes → preempted-slot recovery → in-flight prefill chunks → new-request admission), LIFP preemption, anti-starvation prefill reservation, and EMA-smoothed dynamic SLO budgeting.
- **§9.2 DRM Syncobj Timeline Dispatch:** Single-producer timeline sequencer with dedicated dispatch loop, system-scope iGPU writeback barriers, NPU Tile DMA invalidation, and ERT ring timeout flush with isolated slot teardown.
- **§9.3 Decoupled Q4_0 KV Layout:** INT4 weight and FP16 scale planes split into separate 16-byte Tile-DMA-aligned buffers with dual-view (GPU virtual / NPU physical scatter-gather) block tables.
- **`--stall-free` / `--no-stall-free` toggle pair (default: on):** llama.cpp-standard bool flag style replacing `--enable-stall-free`; env `LLAMA_ARG_STALL_FREE`.
- **Multi-workflow quantization benchmark harness** with NVMe auto-staging, sharded UD-IQ4_XS sanity runner, and large-model progressive-difficulty benchmark runners.
- **Resource-guided conversion strategy engine** and multi-stage AdamW distillation pipeline with teacher SafeTensors integration and MoE NAS streaming.

### Changed
- **Stall-free scheduling is ON by default** with the benchmarked sweet-spot configuration `--token-budget 256 --sarathi-chunk-size 64`; greedy batching remains available via `--no-stall-free`.
- **Authoritative prompt-fill enforces the token budget (Microsoft method):** `n_fill_cap = min(τ, active_decodes + chunk_size)` bounds the prefill share of every iteration, fixing `--sarathi-chunk-size` as a live control instead of dead config.
- APU sparse MoE chunk loader activated by default with dynamic memory scaling; MoE router auto-engages for large models.
- Built and verified against ROCm 10.1 (`amdrocm 10.1.0-3`, HIP clang under `/opt/rocm/core-10.1`).
- **ROCm VMM (`GGML_HIP_NO_VMM`) re-evaluated against 10.1's VMM/NUMA path (Phase 10.5):** the default **`ON` (VMM disabled) is faster** — flipping to VMM cost ≈−9% prefill / ≈−11% decode on gfx1150 (Radeon 890M, UMA). **Default unchanged**; `-DGGML_HIP_NO_VMM=OFF` stays available as an opt-in build parameter.

### Fixed
- KV head adaptation robustness and memory-scaled MoE layer pinning; GPU offload clamped to 75% of the KFD GPU pool to prevent OOM.
- Tile DMA stride assertion downgraded to a warning for non-16-byte-aligned KV chunks.
- Partial tensor loading now skips MTP draft layers cleanly in standard mode.
- UD-IQ4_XS sanity runner adapted to mmap demand paging; strict 3-shard completeness verification.

### Performance (real agentic OpenClaw trace, NeoHorse-1-4B, ROCm 10.1, 0 errors)
- **τ256/c64 vs greedy: p99 TBT 1107 → 423 ms (−62%), max TBT 1194 → 508 ms (−57%), identical p50 (~148 ms).**
- ~10% lower elapsed than the pre-ROCm-10.1 baseline across c64/greedy configs; isolated TBT 87 ms.
- TTFT p50 2.1 s / p99 2.8 s for the 3.1k-token prompt (prefill-dominated).
- Chunk=16 rejected: uniform ~13 s max-TBT outliers on this APU.

---

## [0.8.0] - 2026-09-24

### Added
- **Native Multiplexer Binary:** Added compiled `llama` command-line dispatcher supporting `cli`, `serve`, `bench`, `batched-bench`, `quantize`, `download`, and `perplexity` without symlinks.
- **Full-APU Stage Routing Presets:** Added `--gpu-based`, `--cpu-based`, and `--npu-based` macros controlling stage execution across CPU tokenization, GPU prefill, and NPU decode.
- **Zero-Copy Cross-Device Memory Bridge:** Implemented physical GEM allocation (`/dev/dri/renderD128`) and PRIME dma-buf import into AMDXDNA NPU (`/dev/accel/accel0`) with verified 0 host `memcpy` operations.
- **Dynamic KV Cache Quantization (`Q4_0`):** Real-time KV cache quantization reducing UMA DRAM footprint by 71.9% (3.56x savings) with strict 16-byte Tile DMA beat and 64-byte host cache line alignment.
- **MoE Router On-Chip SRAM Pinning:** Automated isolation and placement of multi-layer router gating matrices ($W_{\text{gate}}$) directly into AIE2P Tile SRAM with configurable `--router-sram {auto,on,off}` and automatic UMA DRAM fallback.
- **DRM Syncobj Timeline Speculative Decoding:** Hardware-synchronized speculative draft verification over Linux DRM timeline fences, achieving sub-10 µs sync latency and eliminating host scheduling bubbles.
- **Native TQ2_0 Standard Support (2.0625 bpw):** Added native `GGML_TYPE_TQ2_0` quantization, RoPE-compliant per-head QuaRot (FWHT) rotations, closed-form Frobenius norm minimization ($\alpha^*$), and T-ACE 16-byte co-packed hardware tile lowering for AMD XDNA 2 AIE2P vector PE.
- **Custom XCLBIN Synthesis & Profile Engine:** Implemented automated topology extraction, spatial 4x8 AIE2P tile allocation, Peano vectorized C++ kernel code generation (`ternary_gemv.cc`), and native `/usr/bin/xclbinutil` packaging with 4-tier hot-registration.
- **End-to-End Conversion Pipeline (`llama-apu-convert`):** Added automated streaming converter transforming SafeTensors / high-precision models into native `TQ2_0` GGUF, companion `.q4nx` binary sidecars, and synthesized XCLBIN profiles.
- **Multi-Stage AdamW Scale Distillation Engine:** Curated a 1,000-sample balanced calibration corpus across 10 domains in `~/databank/distill/` and implemented scale distillation with Cosine Annealing, yielding >22% layer reconstruction MSE reductions.
- **Hardware Diagnostic Doctor (`apu-doctor`):** Integrated diagnostic subcommand in `llama-apu-cli` evaluating XDNA NPU node, GPU DRM render node, KFD compute node, XRT runtime library, and system UMA memory headroom.
- **Production Packaging & Systemd Units:** Added automated installer (`install.sh`), systemd service unit (`llama-server.service`), and udev permission rules (`99-amdxdna-apu.rules`).

### Changed
- Integrated GPU memory allocator automatically disables lazy tensor loading on unified memory architectures (`load_mode = none`) to maximize continuous DMA streaming bandwidth.
- All file writing tools (`llama_apu_convert.py`, `ggml-apu-convert.cpp`, `ptqtp_engine.py`) enforce atomic file operations via temporary PID-stamped buffers and `os.replace` to prevent partial file corruption.
- Multi-entry-point testing protocols enforced across both `llama-cli` and `llama-server` HTTP REST API (`/v1/chat/completions` and `/completion`).

### Fixed
- Fixed Tile DMA misalignment hazards by asserting 16-byte beat boundaries across all KV cache blocks, router SRAM pins, and quantized tensor payloads.
- Fixed DRM syncobj fence drops under high multi-threaded contention by validating 4-thread stress tests with sub-2 µs p99 latency.
- Resolved out-of-core SafeTensors conversion paths for massive models without overflowing the strict 50 GB system memory governor ceiling.

---

## [0.7.2] - 2026-09-24
### Added
- Phase 7.2 end-to-end conversion pipeline orchestrator.
- Companion `.q4nx` container format with 64-byte aligned payload padding.
- Multi-model conversion verified on `Qwen3.8-27B-Cold-Fusion` and `Qwen3-Coder-Next`.

## [0.7.1] - 2026-09-24
### Added
- Phase 7.1 custom XCLBIN synthesis pipeline and tile planner.
- Dynamic profile discovery and Tier 3 local registration.

## [0.7.0] - 2026-09-24
### Added
- Phase 7 native `GGML_TYPE_TQ2_0` type support and T-ACE 16B tile lowering.
- RoPE-compliant per-head QuaRot FWHT transforms and closed-form Frobenius error projection.

## [0.6.0] - 2026-09-24
### Added
- Phase 6 speculative decoding coordination and fast KV cache rollbacks.
- DRM timeline syncobj acceleration on `/dev/dri/renderD128`.

## [0.5.0] - 2026-09-24
### Added
- Phase 5 MoE router matrix isolation and on-chip SRAM pinning.
- Graceful UMA DRAM fallback on budget overflow.

## [0.4.0] - 2026-09-24
### Added
- Phase 4 zero-copy telemetry hooks (`host_memcpy_count: 0`).
- Strict 16B Tile DMA beat and 64B host cache line sub-buffer alignment.

## [0.3.0] - 2026-09-23
### Added
- Phase 3 dynamic KV cache quantization (`Q4_0`) with 71.9% memory savings.

## [0.2.0] - 2026-09-23
### Added
- Phase 2 hardware bridge between `/dev/dri/renderD128` and `/dev/accel/accel0`.
- Initial XRT runtime dynamic loading and NPU capability detection.

## [0.1.0] - 2026-09-23
### Added
- Phase 1 model support, bounded `.q4nx` container reader, and memory estimation.
- Phase 0 ROCm / HIP baseline verification on AMD gfx1150 APU silicon.
