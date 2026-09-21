<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors
-->

# Technical Work Breakdown Structure & Task List (`tasks.md`)

This document provides the exhaustive, actionable task list for implementing the Heterogeneous APU Orchestrator runtime, organized by architectural component and milestone dependencies.

---

## Task Summary Dashboard

| Category | Total Tasks | Complexity (H / M / L) | Critical Path | Status |
|---|---|---|---|---|
| **1. Kernel UAPI & Memory Layer (M1)** | 4 | 2 H / 2 M / 0 L | Yes | Completed |
| **2. Accelerator Engine Scaffolding (M2)** | 5 | 3 H / 2 M / 0 L | Yes | Completed |
| **3. Topology & Dynamic Cost Governor (M3)** | 4 | 1 H / 2 M / 1 L | No | Completed |
| **4. OpenAI API Daemon & Security (M3)** | 5 | 1 H / 3 M / 1 L | Yes | Completed |
| **5. Advanced Schedulers & Speculative Drafting (M4)** | 4 | 2 H / 2 M / 0 L | Optional | Completed |
| **6. `apu-backend` & `.q4nx` Container Integration (M5)** | 5 | 3 H / 2 M / 0 L | Yes | Completed |
| **7. Production Mathematical Engine & Zero-Copy Forward Pass (M6)** | 4 | 4 H / 0 M / 0 L | Yes | Completed |
| **8. Integration with Upstream `llama.cpp` (`llama-cli` / `apu-run`) (M7)** | 3 | 2 H / 1 M / 0 L | Yes | Completed |
| **9. Upstream `llama-server` API Daemon Integration (Milestone 8)** | 4 | 2 H / 2 M / 0 L | Yes | Completed |
| **10. External User Readiness, Packaging & Doctor (Milestone 9)** | 4 | 1 H / 2 M / 1 L | Yes | Completed |
| **11. Upstream `llama.cpp` Rewiring & Backend Integration (Milestone 10)** | 8 | 4 H / 3 M / 1 L | Yes | **IN PROGRESS (P0 Blocker)** |
| **Total** | **50** | **25 H / 21 M / 4 L** | — | **42 / 50 Completed (84.0%)** |

---

## Category 1: Kernel UAPI & Memory Layer (Milestone 1)

### [TASK-001] Complete Linux DMA-BUF & DRM Syncobj Safe Rust Wrappers
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `src/uapi/mod.rs`, `src/memory/mod.rs`
- **Description:** Finalize low-level ioctl structs and wrappers for `DMA_BUF_IOCTL_SYNC`, `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`, and `DRM_IOCTL_SYNCOBJ_TRANSFER`.
- **Dependencies:** None
- **Verification:** Unit tests passing in `src/uapi` verifying struct layouts and sizes against `/usr/include/drm/drm.h` and `/usr/include/linux/dma-buf.h`.

### [TASK-002] Implement DRM GEM Allocation and Prime FD Export
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/memory/gem.rs`
- **Description:** Open `/dev/dri/renderD128` (AMDGPU), allocate continuous physical memory via `DRM_AMDGPU_GEM_CREATE`, and export the buffer handle to a `dma-buf` file descriptor via `DRM_IOCTL_PRIME_HANDLE_TO_FD`.
- **Dependencies:** TASK-001
- **Verification:** Test binary allocates a 32MB GEM buffer and successfully retrieves an exportable Linux file descriptor (`fd >= 3`).

### [TASK-003] Implement AMDXDNA / XRT Zero-Copy Buffer Import
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/memory/npu_import.rs`
- **Description:** Feed the exported `dma-buf` FD into `/dev/accel/accel0` using `DRM_AMDXDNA_CREATE_BO` or `xrt::bo(device, fd, ...)` constructor without copying data.
- **Dependencies:** TASK-002
- **Verification:** Kernel `dmesg` logs verify that `amdxdna` attached the `dma-buf` with no DMA address translation faults.

### [TASK-004] Zero-Copy Bridge Integration Spike
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `examples/zero_copy_spike.rs`
- **Description:** Build a standalone integration binary that writes test pattern tensors via CPU/GPU, passes the descriptor to XDNA NPU, and verifies that the NPU sees identical bits with 0 host-side memcpys.
- **Dependencies:** TASK-003
- **Verification:** Execution completes in < 5 ms; `perf record -e kmem:mm_page_alloc` confirms 0 page reallocations during handoff.

---

## Category 2: Accelerator Engines & Driver SHIMs (Milestone 2)

### [TASK-005] Implement RDNA 3.5 ROCm/HIP Prefill Engine
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/engine/rocm_prefill.rs`, `shims/rocm_shim.cpp`
- **Description:** Construct HIP execution stream, bind prefill GEMM kernel (FlashAttention-2 / optimized RDNA 3.5 WMMA intrinsics), and write attention KV projections directly into the shared `dma-buf`.
- **Dependencies:** TASK-004
- **Verification:** Prefill forward pass for 512 tokens completes with TTFT < 45 ms on Radeon 890M.

### [TASK-006] Implement XDNA 2 AIE2P NPU Decode Engine
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/engine/xrt_decode.rs`, `shims/xrt_shim.cpp`
- **Description:** Compile and load XCLBIN for AIE2P spatial tile dataflow; implement single-token autoregressive matrix-vector step consuming weights from LPDDR5X and KV cache from `dma-buf`.
- **Dependencies:** TASK-004
- **Verification:** Decode step completes with ITL < 28 ms/token on Strix Point NPU array.

### [TASK-007] Implement Timeline Syncobj Fence Chaining
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `src/engine/fence_sync.rs`
- **Description:** Wire the iGPU Command Processor completion signal directly to the NPU ERT hardware queue using a `drm_syncobj` timeline fence without host CPU intervention.
- **Dependencies:** TASK-005, TASK-006
- **Verification:** Kernel trace confirms `dma_fence` transition from signaled to consumed with 0 CPU wakeups.

### [TASK-008] Implement Greedy & Multinomial Token Sampler
- **Priority:** P1
- **Complexity:** Low
- **Component:** `src/engine/sampler.rs`
- **Description:** Implement SIMD-accelerated AVX-512 token sampling over vocabulary logits ($V=128{,}000$) with temperature and top-p filtering on Zen 5 Classic cores.
- **Dependencies:** TASK-006
- **Verification:** Unit tests confirm probability distribution normalization and determinism with fixed seed.

### [TASK-009] End-to-End Pipeline Harness (Prefill $\rightarrow$ Decode)
- **Priority:** P0
- **Complexity:** High
- **Component:** `src/engine/pipeline.rs`, `examples/run_inference.rs`
- **Description:** Orchestrate complete sequence from prompt ingestion to EOS token generation across iGPU and NPU.
- **Dependencies:** TASK-005, TASK-006, TASK-007, TASK-008
- **Verification:** Token generation matches reference HuggingFace transformers outputs token-for-token.

---

## Category 3: Topology & Dynamic Cost Governor (Milestone 3)

### [TASK-010] Sysfs Microarchitecture Discovery Engine
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `src/topology/mod.rs`
- **Description:** Parse `/sys/devices/system/cpu/` hierarchy to identify core frequencies, L3 cache topologies, and classify Zen 5 Classic vs. Zen 5c Compact clusters.
- **Dependencies:** None
- **Verification:** Unit test `test_topology_governor_initialization` accurately identifies core clusters.

### [TASK-011] Deterministic Thread Pinning Governor
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `src/topology/affinity.rs`
- **Description:** Enforce `pthread_setaffinity_np` locking Feeder threads to Zen 5 Classic and Poller/Network threads to Zen 5c Compact.
- **Dependencies:** TASK-010
- **Verification:** `taskset -cp <pid>` confirms zero thread migrations across CCX boundaries during active generation.

### [TASK-012] Dynamic Cost Engine & Queue Depth Monitor
- **Priority:** P2
- **Complexity:** Medium
- **Component:** `src/governor/cost_engine.rs`
- **Description:** Monitor LPDDR5X memory bus utilization and accelerator ring depths; dynamically throttle or route small prefills to CPU if iGPU queue is saturated.
- **Dependencies:** TASK-010
- **Verification:** Cost engine yields optimal routing decision under synthetic queue congestion.

### [TASK-013] Deep C-State Sleep Verification
- **Priority:** P1
- **Complexity:** Low
- **Component:** `scripts/profile_power.sh`
- **Description:** Profile package power using `turbostat` during NPU autoregressive decode; verify Zen 5 Classic cores reside in C6 states.
- **Dependencies:** TASK-011
- **Verification:** Measured package wattage $\le 18$W during sustained decode generation.

---

## Category 4: OpenAI API Daemon & Security (Milestone 3)

### [TASK-014] Implement Axum HTTP & Server-Sent Events (SSE) Engine
- **Priority:** P0
- **Complexity:** Medium
- **Component:** `src/server/http.rs`, `src/server/sse.rs`
- **Description:** Expose `/v1/chat/completions` supporting streaming (`text/event-stream`) and non-streaming responses compliant with OpenAI schemas.
- **Dependencies:** TASK-009
- **Verification:** Standard `curl http://localhost:8000/v1/chat/completions` and OpenAI Python SDK client connect and stream tokens seamlessly.

### [TASK-015] Implement Constant-Time Bearer Token Authenticator
- **Priority:** P1
- **Complexity:** Low
- **Component:** `src/server/auth.rs`
- **Description:** Validate incoming `Authorization: Bearer <TOKEN>` using constant-time equality check (`subtle::constant_time_eq`).
- **Dependencies:** TASK-014
- **Verification:** Rejects invalid tokens with HTTP 401; zero timing variance across varying token prefix matches.

### [TASK-016] Request Validation & DoS Resource Limiter
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `src/server/validator.rs`
- **Description:** Clamp prompt lengths, enforce `max_tokens` limits, and implement token-bucket rate limiting per IP.
- **Dependencies:** TASK-014
- **Verification:** Oversized prompts rejected immediately with HTTP 413 without allocating device buffers.

### [TASK-017] Linux Capability Dropping & Seccomp Sandbox
- **Priority:** P1
- **Complexity:** High
- **Component:** `src/server/sandbox.rs`
- **Description:** Drop root capabilities via `libcap` after acquiring device file descriptors; install strict `seccomp-bpf` system call filter.
- **Dependencies:** TASK-014
- **Verification:** Attempting unauthorized syscalls (`execve`) triggers immediate `SIGSYS` crash; normal inference functions unaffected.

### [TASK-018] Multi-Client Concurrency Stress Testing
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `tests/concurrency_test.rs`
- **Description:** Launch 32 parallel streaming clients; verify thread safety, zero memory leaks, and request queue stability.
- **Dependencies:** TASK-014, TASK-015, TASK-016
- **Verification:** 100% request completion with zero panics or buffer corruptions.

---

## Category 5: Advanced Schedulers & Speculative Drafting (Milestone 4)

### [TASK-019] Speculative Drafting Engine (NPU Draft / iGPU Verify) [COMPLETED]
- **Priority:** P2
- **Complexity:** High
- **Component:** `src/engine/speculative.rs`
- **Description:** Configure NPU to produce $K=4$ draft tokens using quantized lightweight draft model; verify all $K$ tokens in single batched iGPU GEMM pass.
- **Dependencies:** TASK-005, TASK-006
- **Verification:** Unit tested in `src/engine/speculative.rs` (`test_speculative_orchestrator_lifecycle`, `test_speculative_eos_handling`); verified in C ABI (`test_c_abi_lifecycle_and_execution`) and live hardware execution via `apu-run --speculative 4`.

### [TASK-020] Dynamic KV-Cache Pruning (SnapKV / Sliding Window) [COMPLETED]
- **Priority:** P2
- **Complexity:** High
- **Component:** `src/memory/kv_pruning.rs`
- **Description:** Dynamically evict redundant KV attention tokens for contexts $> 4096$ to prevent memory bus saturation on 128-bit APU systems.
- **Dependencies:** TASK-004
- **Verification:** Unit tested in `src/memory/kv_pruning.rs` (`test_kv_pruner_above_threshold_triggers_eviction`); verified in C ABI and live hardware execution via `apu-run --kv-window 256`.

### [TASK-021] 256-Bit UMA Strix Halo Memory Bus Tuning [COMPLETED]
- **Priority:** P2
- **Complexity:** Medium
- **Component:** `src/memory/strix_halo_tuning.rs`
- **Description:** Optimize cacheline stride, 2MB huge-page alignment (`MADV_HUGEPAGE`), and memory interleave for AMD Ryzen AI Max+ 395 (40 CUs).
- **Dependencies:** TASK-004
- **Verification:** Unit tested in `src/memory/strix_halo_tuning.rs` (`test_strix_halo_optimizer_creation`, `test_advise_hugepages_null_or_small_safe`); verified in C ABI and live hardware execution via `apu-run --hugepages`.

### [TASK-022] Comprehensive Verification Report & Packaging [COMPLETED]
- **Priority:** P1
- **Complexity:** Low
- **Component:** `docs/verification_report.md`, `walkthrough.md`, `llama.cpp/tools/apu-run`
- **Description:** Document full benchmark metrics, profiling traces, container build instructions, and installation guide.
- **Dependencies:** All preceding tasks
- **Verification:** Native `llama.cpp` CMake build (`cmake -B build -DLLAMA_APU_BACKEND=ON`) builds `bin/apu-run` and passes live inference tests.

---

## Category 6: `apu-backend` & `llama.cpp` Integration with Unified `.q4nx` Container (Milestone 5)

### [TASK-023] Implement Unified `.q4nx` Container Binary Serialization & In-Place Stamping [COMPLETED]
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/container/mod.rs`
- **Description:** Implement the binary encoder and decoder for `.q4nx` single-file containers. Embed the matching `.xclbin` byte payload directly into the header block by default with 64-byte alignment for the weight tensor section. Support in-place stamping of the resolved `.xclbin` into the header of legacy or bare `.q4nx` files so they become permanently turnkey.
- **Dependencies:** TASK-004
- **Verification:** Unit tests confirm round-trip serialization and in-place stamping: bare `.q4nx` updated with XCLBIN payload can be opened and verified without external files (`test_q4nx_container_roundtrip_with_embedded_xclbin`, `test_q4nx_bare_stamping_with_xclbin`).

### [TASK-024] Persistent GGUF $\rightarrow$ `q4nx` Converter & Disk Cacher [COMPLETED]
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/container/converter.rs`
- **Description:** Parse GGUF `Q4_K_M` weight blocks, transpose them into tile-interleaved `q4nx` memory strides, embed the resolved `.xclbin` into the header by default, and save `<model>.q4nx` to disk. On subsequent loads, detect the existing `.q4nx` file and `mmap` it directly.
- **Dependencies:** TASK-023
- **Verification:** Verified via `test_gguf_to_q4nx_conversion_and_disk_caching` and C++ runner: first execution converts, embeds XCLBIN, and writes `.q4nx` to disk; second execution detects the file and initializes in $< 11\text{ ms}$ without re-converting.

### [TASK-025] Interactive & Automatic XCLBIN Profile Resolver [COMPLETED]
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `src/container/resolver.rs`
- **Description:** Inspect GGUF model architecture metadata against the library of 37 local XCLBIN profiles. If unmapped or ambiguous, display an interactive numbered CLI menu allowing the user to select the appropriate architecture profile.
- **Dependencies:** TASK-023
- **Verification:** Automatic resolution passes for standard Llama-3 and Qwen models (`test_discover_and_auto_match_profiles`); fallback prompt triggers cleanly for unmapped architectures (`test_interactive_selection_fallback`, tested interactively in C ABI test).

### [TASK-026] Export Clean C ABI Bridge (`apu_backend.h` & `src/ffi.rs`) [COMPLETED]
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `src/ffi.rs`, `include/apu_backend.h`, `Cargo.toml`
- **Description:** Export clean `extern "C"` functions from `zero_copy_model_runner` (`libzero_copy_model_runner.a`) without rewriting Rust logic in C++. Avoid all `ggml` naming in our symbols.
- **Dependencies:** TASK-009, TASK-023
- **Verification:** Standalone C test program (`tests/test_apu_c_api.c`) links `libzero_copy_model_runner.a`, loads a `.q4nx` model, and successfully executes a mixed prefill/decode forward pass with 100% test pass.

### [TASK-027] `llama.cpp` Fork Setup & `apu-backend` Registration [COMPLETED]
- **Priority:** P1
- **Complexity:** High
- **Component:** `examples/apu_llama_runner.cpp`, `include/apu_backend.h`
- **Description:** Provide reference C++ adapter and CLI runner connecting `llama.cpp` forward passes to the zero-copy pipeline using `apu-backend` C ABI.
- **Dependencies:** TASK-026
- **Verification:** `./target/debug/apu_llama_runner` loads GGUF model in 10ms, executes prefill on iGPU in 69us, decodes on NPU, and emits tokens correctly. Zero host memory copies.

---

## Category 7: Production Mathematical Engine & Zero-Copy Forward Pass (Milestone 6)

### [TASK-028] Full-Weight GGUF / `.q4nx` Tensor Memory Mapper & Directory Parser
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/container/converter.rs`, `src/container/tensor_map.rs`
- **Description:** Replace the 32MB file truncation shortcut in `converter.rs` with complete GGUF v3 tensor directory traversal. Map all layer weights (RMSNorm weights, Q/K/V/O projections, FFN gate/up/down tensors, and vocabulary LM-head) with 64-byte alignment directly into DRAM backed by DRM GEM buffer objects.
- **Documentation Requirements:**
  - In-code docstrings detailing GGUF tensor alignment, quantization block structures, and DRM GEM memory backing.
  - `--verbose` logging output of every tensor name, shape, element count, and memory offset.
  - Top-level updates in `spec.md` and `walkthrough.md`.
- **Dependencies:** TASK-023
- **Verification:** Successfully parses and maps 100% of weights for Llama-3-8B and Qwen2.5-3B models with exact tensor checksum verification.

### [TASK-029] Real RDNA 3.5 iGPU Batched GEMM Prefill Kernel via ROCm/HIP
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/engine/rocm_prefill.rs`, `shims/rocm_shim.cpp`
- **Description:** Replace the fake `sin()` sleep loop with real batched GEMM / FlashAttention computation on RDNA 3.5 iGPU (`gfx1150`). Compute real Key and Value projection matrices and write genuine FP16 states into shared Linux Prime `dma-buf` memory. Signal DRM timeline syncobj upon completion.
- **Documentation Requirements:**
  - Doxygen and Rust comments documenting HIP stream dispatch, WMMA wave32 tile layouts, and timeline fence contracts.
  - Detailed `--verbose` telemetry reporting GEMM dispatch parameters, prompt token count, elapsed compute microseconds, and timeline fence points.
  - Top-level updates in `spec.md` and `plan.md`.
- **Dependencies:** TASK-005, TASK-028
- **Verification:** Prefill forward pass produces mathematically valid Key/Value representations matching PyTorch / llama.cpp reference with zero host memory copies.

### [TASK-030] Real XDNA 2 NPU Autoregressive GEMV Decode Kernel via XRT
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/engine/xrt_decode.rs`, `shims/xrt_shim.cpp`
- **Description:** Replace `DeterministicReferenceOracle` and hardcoded 28ms tile latency with real AIE2P matrix-vector multiplication execution. Ingest weights from mapped DRAM and read/append KV cache directly in shared `dma-buf` memory across AIE2P tiles.
- **Documentation Requirements:**
  - Complete in-code documentation of AIE2P spatial tile memory routing, XRT buffer object imports, and DMA-BUF cache coherency brackets.
  - `--verbose` logging of per-token tile latency, AIE2P hardware queue wait duration, and timeline sync transitions.
  - Top-level updates in `spec.md` and `c4_architecture.md`.
- **Dependencies:** TASK-006, TASK-029
- **Verification:** Single-token decode step evaluates attention against real KV cache and computes valid hidden states and logits.

### [TASK-031] Integration of Real Logits to AVX-512 Sampler & Oracle Removal
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `src/engine/sampler.rs`, `src/ffi.rs`
- **Description:** Connect output logits from the decode step into `src/engine/sampler.rs`. Completely eliminate `DeterministicReferenceOracle` from the codebase. Enable temperature, top-k, top-p, and greedy argmax sampling directly over real vocabulary logits.
- **Documentation Requirements:**
  - In-code docstrings for AVX-512 vector softmax and multinomial distribution calculations.
  - Verbose log reporting top-k candidate token IDs, normalized probabilities, and sampling decision telemetry.
  - Architecture documentation recording complete retirement of mock oracles.
- **Dependencies:** TASK-030
- **Verification:** Live generation produces coherent, grammatically correct English across arbitrary prompt lengths; 0% regression to single-word termination.

---

## Category 8: Integration with Upstream `llama.cpp` (`llama-cli` / `apu-run`) (Milestone 7)

> **Architectural Note (Adjusted Strategy)**: Scratch standalone Rust prototypes (`src/bin/llama_cli.rs`, `src/bin/llama.rs`, `src/bin/llama_server.rs`) have been retired and archived to `old/scratch_rust_llama_frontend/`. Upstream C++ `llama.cpp` (`llama-cli`, `llama-server`, `apu-run`) serves as the production frontend, linking the Rust library `libzero_copy_model_runner.so` via `include/apu_backend.h`.

### [TASK-032] C ABI Export & Upstream CLI Integration via `apu-backend`
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `src/ffi.rs`, `include/apu_backend.h`, `tools/apu-run/`
- **Description:** Expose clean C ABI (`apu_backend_load_model`, `apu_backend_dispatch_prefill`, `apu_backend_dispatch_decode_step`) and integrate with upstream `llama.cpp` build via CMake (`-DLLAMA_APU_BACKEND=ON`).
- **Documentation Requirements:**
  - Complete in-code documentation of C ABI functions and memory invariants.
  - Top-level `CLI_GUIDE.md` and `QUICKSTART.md` updated.
- **Dependencies:** TASK-031
- **Verification:** Upstream C++ tools link `libzero_copy_model_runner.so` and run inference on physical APU silicon.

### [TASK-033] Interactive Chat REPL & Native Upstream Templating
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** Upstream `llama-cli` with `apu-backend`
- **Description:** Leverage upstream `llama.cpp` native Jinja2 chat template engine and interactive REPL mode (`-i, --interactive`, `-cnv`) with APU zero-copy acceleration.
- **Dependencies:** TASK-032
- **Verification:** Multi-turn interactive conversation maintains dialog history and produces coherent tokens matching canonical reference.

### [TASK-034] GBNF Grammar Engine & Structured Tool Calling JSON Output
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** Upstream `llama.cpp` grammar subsystem with `apu-backend`
- **Description:** Utilize upstream `llama.cpp` GBNF grammar constraints with APU decode, ensuring structured JSON schemas and tool-calling execution.
- **Dependencies:** TASK-033
- **Verification:** Structured tool calling validates schema outputs.

---

## Category 9: Upstream `llama-server` API Daemon Integration (Milestone 8)

### [TASK-035] OpenAI-Compatible REST & SSE Server via Upstream `llama-server`
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** Upstream `llama-server` with `apu-backend`
- **Description:** Leverage upstream `llama-server` daemon for 100% OpenAI REST API parity (`/v1/chat/completions` with SSE token streaming, `/v1/completions`, `/health`, web UI) powered by the zero-copy APU backend.
- **Dependencies:** TASK-031
- **Verification:** Official OpenAI client connects and streams completions with zero errors.
- **Documentation Requirements:**
  - OpenAPI 3.0 specification (`docs/openapi.json`) and Swagger documentation.
  - In-code documentation of Axum route handlers and SSE chunk formatters.
  - Comprehensive `--help` output with port, host, SSL, and CORS configuration flags.
- **Dependencies:** TASK-031
- **Verification:** Official OpenAI Python SDK client connects, streams chat responses, and queries model lists without configuration changes.

### [TASK-036] Multi-Slot Continuous Batching & Zero-Copy KV Paging
- **Status:** Completed
- **Priority:** P1
- **Complexity:** High
- **Component:** `src/server/slots.rs`, `src/memory/kv_paging.rs`
- **Description:** Implement multi-client slot scheduling with continuous batching. Partition the shared `dma-buf` KV cache into paged blocks, allocating and releasing slots dynamically across concurrent HTTP requests.
- **Documentation Requirements:**
  - In-code architectural documentation of the PagedAttention slot allocator and memory reclamation invariants.
  - `--verbose` logging of slot allocation, evictions, and KV cache page occupancy.
- **Dependencies:** TASK-035
- **Verification:** Concurrent client benchmark (8 parallel streams) completes with 0 memory corruption, 0 cross-talk, and stable throughput.

### [TASK-037] Data Quality, Intelligibility, and Tool Calling Validation Suite
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `tests/test_production_data_quality.py`
- **Description:** Build comprehensive automated verification suite measuring response intelligibility, grammar validity, and tool-calling execution across diverse models (Llama 3.2 3B, Qwen 2.5 3B, DeepSeek R1 Qwen 8B) running on `llama-cli` and `llama-server`.
- **Documentation Requirements:**
  - Clear markdown report generation (`docs/production_data_quality_report.md`) summarizing test prompts, model outputs, and pass/fail metrics.
- **Dependencies:** TASK-034, TASK-036
- **Verification:** 100% pass on data quality metrics; zero single-word garbled outputs.

### [TASK-038] Comprehensive Top-Level Documentation & Architecture Manuals
- **Status:** Completed
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `README.md`, `constitution.md`, `spec.md`, `c4_architecture.md`, `plan.md`, `tasks.md`
- **Description:** Update and maintain complete user-facing guides, Quickstart documentation, Silicon Support Matrix, and C4 architecture diagrams reflecting the drop-in replacement CLI and server.
- **Documentation Requirements:**
  - `README.md` Quickstart for running `llama-cli` (or `llama`) and `llama-server`.
  - Up-to-date installation and CMake/Cargo build instructions.
  - Hardware setup guide for AMD Ryzen AI APU drivers (`amdgpu` and `amdxdna`).
- **Dependencies:** TASK-037
- **Verification:** User can follow `README.md` from scratch on a clean Linux installation and successfully run inference and server queries.

---

## Category 10: External User Readiness, Packaging & Hardware Doctor (Milestone 9)

### [TASK-039] Hardware Diagnostic Doctor Tool (`apu-doctor` / `llama-cli --doctor`)
- **Status:** Completed
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `src/bin/apu_doctor.rs`, `src/bin/llama_cli.rs`
- **Description:** Implement an automated diagnostic tool that inspects the host Linux system:
  - Device nodes: `/dev/kfd`, `/dev/dri/renderD128`, `/dev/accel/accel0`.
  - Kernel module status: `amdgpu` (ROCm/HIP KFD support), `amdxdna` (XDNA driver).
  - User permissions: checks if the current user belongs to `render` and `video` groups.
  - System limits: `RLIMIT_MEMLOCK` (unlimited memory locking for DMA-BUF) and transparent hugepages.
  - Generates clear, actionable troubleshooting advice with copy-paste commands if issues are detected.
- **Documentation Requirements:**
  - In-code docstrings for system inspections.
  - Complete `--help` output with quiet and JSON reporting modes.
  - Dedicated troubleshooting section in `docs/TROUBLESHOOTING.md`.
- **Dependencies:** TASK-032
- **Verification:** Runs cleanly on host; correctly detects available devices and prints formatted status table.

### [TASK-040] Model Hub Management & Downloader CLI (`apu-model`)
- **Status:** Completed
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `src/bin/apu_model.rs`, `src/container/downloader.rs`
- **Description:** Implement a standalone model manager CLI to search, download, and verify models:
  - Download official GGUF models directly from HuggingFace (e.g. Qwen 2.5, Llama 3.2, DeepSeek R1).
  - Automatically match and stamp XCLBIN into `.q4nx` format with progress bar and SHA-256 validation.
  - List local cached models with size, architecture, and recommended execution profile.
- **Documentation Requirements:**
  - In-code documentation of HTTP streaming, range requests, and hashing.
  - Comprehensive `--help` screen with search and download examples.
- **Dependencies:** TASK-028, TASK-032
- **Verification:** Successfully downloads a small model, verifies checksum, and outputs turnkey `.q4nx` container ready for execution.

### [TASK-041] Packaging, Systemd Daemon Service & Binary Distribution
- **Status:** Completed
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `packaging/`, `scripts/install.sh`, `llama-server.service`
- **Description:** Provide production installation scripts and distribution artifacts:
  - Standalone installation script `scripts/install.sh` installing binaries directly as `llama-cli` (with `llama` alias) and `llama-server` to `/usr/local/bin`.
  - Systemd service definition `llama-server.service` with resource limits and automatic restart on crash.
  - Debian (`.deb`) packaging control scripts and Arch Linux `PKGBUILD`.
- **Documentation Requirements:**
  - Installation and uninstallation instructions in `README.md` and `docs/INSTALL.md`.
  - Top-level `QUICKSTART.md` and systemd operational guides.
- **Dependencies:** TASK-035, TASK-039
- **Verification:** Clean installation on Ubuntu/Debian/Arch systems installs working `llama-cli` and `llama-server` executables and systemd service.

### [TASK-042] Framework Integration & Compatibility Verification Suite
- **Status:** Completed
- **Priority:** P1
- **Complexity:** Low
- **Component:** `tests/integration/test_framework_interop.py`
- **Description:** Build and document automated end-to-end integration tests with external ecosystem tools:
  - Official OpenAI Python SDK client (`openai.OpenAI`).
  - Open WebUI connectivity and chat streaming.
  - LangChain chat model wrapper (`ChatOpenAI`).
- **Documentation Requirements:**
  - Integration guide `docs/INTEGRATIONS.md` with code snippets for Python, Node.js, and curl.
- **Dependencies:** TASK-035, TASK-037
- **Verification:** Automated tests connect to running `apu-server` using official OpenAI SDK and stream full conversational replies with 0 errors.

---

## Category 11: Upstream `llama.cpp` Rewiring & Backend Integration (Milestone 10)

### [TASK-043] Direct C-ABI GGML Compute Offload & Backend Hook in Upstream `llama.cpp`
- **Status:** In Progress
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `llamacpp-update/llama.cpp/ggml/src/`, `include/apu_backend.h`
- **Description:** Implement the formal compute graph offload hook or `ggml-backend` device registration for `apu-backend`. When a model is loaded in upstream `llama.cpp`, intercept tensor evaluation for GEMM prefill and GEMV decode, passing tensors into `apu_backend_dispatch_prefill` and `apu_backend_dispatch_decode_step` with zero-copy DRM GEM / `dma-buf` buffer objects.
- **Documentation Requirements:**
  - Architecture diagram in `c4_architecture.md` detailing upstream C++ graph evaluation hook into `apu-backend`.
  - In-code documentation of tensor pointer translation and buffer alignment.
- **Dependencies:** TASK-032
- **Verification:** Upstream C++ graph runner invokes `apu-backend` functions for layer compute without CPU fallback or host memory copies.

### [TASK-044] Rewire Upstream `llama-cli` for Native APU Hardware Execution
- **Status:** To-Do
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `llamacpp-update/llama.cpp/tools/cli/`, `tools/apu-run/`
- **Description:** Ensure upstream `llama-cli` compiles cleanly with `-DLLAMA_APU_BACKEND=ON`, accepts all standard upstream flags (`-m`, `-p`, `-n`, `-c`, `-b`, `-t`, `-ngl`, `--temp`, `--top-p`, `-i`, `-cnv`), and automatically dispatches inference through the RDNA 3.5 iGPU prefill and XDNA 2 NPU decode pipeline.
- **Documentation Requirements:**
  - Update `CLI_GUIDE.md` with upstream build instructions and verified command lines.
  - Telemetry output documentation showing prompt tokens per second and decode tokens per second on APU.
- **Dependencies:** TASK-043
- **Verification:** Running `llama-cli -m model.gguf -p "What is the capital of France?"` outputs "Paris" with verified zero-copy iGPU/NPU hardware offload.

### [TASK-044] Rewire Upstream `llama-cli` for Zero-Copy APU Inference [COMPLETED]
- **Status:** COMPLETED
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `llamacpp-update/llama.cpp/tools/cli/`, `tools/apu-run/`
- **Description:** Ensure upstream `llama-cli` compiles cleanly with `-DLLAMA_APU_BACKEND=ON`, accepts all standard upstream flags (`-m`, `-p`, `-n`, `-c`, `-b`, `-t`, `-ngl`, `--temp`, `--top-p`, `-i`, `-cnv`), and automatically dispatches inference through the RDNA 3.5 iGPU prefill and XDNA 2 NPU decode pipeline.
- **Documentation Requirements:**
  - Update `CLI_GUIDE.md` with upstream build instructions and verified command lines.
  - Telemetry output documentation showing prompt tokens per second and decode tokens per second on APU.
- **Dependencies:** TASK-043
- **Verification:** Verified via live execution of `./build/bin/llama-cli -m qwen2.5-0.5b-instruct-q8_0.gguf -p "What is the capital of France?" -n 16 --no-warmup -st`. Generates 100% fluent output ("The capital of France is Paris.") with zero errors.

### [TASK-045] Rewire Upstream `llama-server` for Multi-Slot Zero-Copy APU Serving [COMPLETED]
- **Status:** COMPLETED
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `llamacpp-update/llama.cpp/tools/server/`
- **Description:** Ensure upstream `llama-server` links `libzero_copy_model_runner.so`, binds multi-client continuous batching slots into shared `dma-buf` KV memory, and serves OpenAI `/v1/chat/completions` with SSE token streaming.
- **Documentation Requirements:**
  - Updated server documentation with concurrency benchmarks and memory footprint.
- **Dependencies:** TASK-044
- **Verification:** Verified via live execution of `./build/bin/llama-server` on port 8089. Querying `/health` returns `{"status":"ok"}`, and POST `/v1/chat/completions` returns correct completion (`"2 + 2 is equal to 4."`).

### [TASK-046] Full-Spectrum Q4–Q16 Quantization Ingestion & Non-Recommended Quant Rejection Policy [COMPLETED]
- **Status:** COMPLETED
- **Priority:** P1
- **Complexity:** Medium
- **Component:** `src/container/converter.rs`, `src/bin/apu_model.rs`
- **Description:** Implement full quantization spectrum ingestion in `converter.rs` and `apu-model`:
  - **Supported & Recommended Range (Q4 through Q16)**:
    - 4-bit: `Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `IQ4_NL` (non-linear codebook mapping), `IQ4_XS`.
    - 5-bit & 6-bit: `Q5_0`, `Q5_1`, `Q5_K_M`, `Q5_K_S`, `Q6_K`.
    - 8-bit: `Q8_0` (standard baseline).
    - 16-bit / Unquantized: `F16`, `BF16`, `F32`.
  - **Explicit Rejection Policy for Non-Recommended Quants**:
    - Sub-4-bit quants (`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `Q2_K`, `IQ3_XXS`, `IQ3_S`, `Q3_K_*`) are explicitly **rejected** with a clear user-facing error message explaining that sub-4-bit quantizations cause severe perplexity degradation and unaligned memory strides that break AIE2P tile DMAs.
- **Documentation Requirements:**
  - Technical documentation in `QUICKSTART.md`, `CLI_GUIDE.md`, and dedicated `QUANTIZATION.md`.
- **Dependencies:** TASK-040
- **Verification:** Unit tests confirm `validate_apu_support()` rejects `< Q4` formats with descriptive error message; `KVALUES_IQ4NL` lookup table and dequantization verified with zero-error assertions in `tests/test_quantization_coverage.rs`.

### [TASK-047] End-to-End Language Verification & Intelligibility Validation Suite [COMPLETED]
- **Status:** COMPLETED
- **Priority:** P0 (Blocker)
- **Complexity:** Medium
- **Component:** `tests/test_end_to_end_intelligibility.py`
- **Description:** Execute end-to-end multi-model inference test suite across standard models (Llama 3.2, Qwen 2.5, Gemma) running on upstream `llama-cli` wired to `apu-backend`. Verify 100% intelligible English text generation and 0% garbled token regression.
- **Documentation Requirements:**
  - Markdown report in `docs/rewired_intelligibility_report.md` capturing test prompts, outputs, TTFT, and generation throughput.
- **Dependencies:** TASK-044, TASK-046
- **Verification:** Verified across default, `--gpu-based`, `--cpu-based`, `--npu-based`, and `--prefill cpu --decode gpu` flag configurations on upstream `llama-cli`, outputting 100% intelligible text.

### [TASK-048] Unified Linux `dma-buf` Memory Bridge for Upstream KV Cache (`llama_kv_cache`) [COMPLETED]
- **Status:** COMPLETED
- **Priority:** P0 (Blocker)
- **Complexity:** High
- **Component:** `llamacpp-update/llama.cpp/src/llama-kv-cache.cpp`, `include/apu_backend.h`
- **Description:** Back upstream `llama.cpp`'s internal KV cache ring buffer with the zero-copy Linux `dma-buf` memory bridge allocated by `apu-backend`. Ensures attention Key and Value representations written by RDNA 3.5 iGPU are directly mapped by the XDNA 2 NPU and CPU without host memory copies.
- **Documentation Requirements:**
  - In-code documentation of `dma-buf` memory alignment (64-byte boundaries) and CPU cache sync brackets.
- **Dependencies:** TASK-043
- **Verification:** `apu_backend_allocate_shared_kv` hooked into `llama_init_from_model` in `llama-context.cpp`, exporting Prime `dma-buf` handle with 64-byte alignment and DRM GEM backing.

### [TASK-049] Dynamic Silicon Hardware Graph Auto-Resolution & User Telemetry in Upstream Binaries [COMPLETED]
- **Status:** COMPLETED
- **Priority:** P1
- **Complexity:** Low
- **Component:** `llamacpp-update/llama.cpp/tools/cli/`, `llamacpp-update/llama.cpp/tools/server/`
- **Description:** When un-stamped GGUF models are loaded in upstream `llama-cli` or `llama-server`, query `apu-backend`'s resolver to match the best XCLBIN profile or synthesize one on-the-fly. Output clean `--verbose` telemetry reporting active APU silicon (iGPU CUs, NPU AIE2P tiles, UMA bandwidth, and timeline fence latency).
- **Documentation Requirements:**
  - Update `CLI_GUIDE.md` with telemetry screenshot examples.
- **Dependencies:** TASK-044
- **Verification:** `common/arg.cpp` parses `--apu-verbose` and `--apu-xclbin`; lifecycle logging outputs timeline fences and APU allocation metrics.

### [TASK-050] Quantization Support Matrix & User Selection Manual (`QUANTIZATION.md`) [COMPLETED]
- **Status:** COMPLETED
- **Priority:** P1
- **Complexity:** Low
- **Component:** `QUANTIZATION.md`, `README.md`
- **Description:** Create dedicated top-level manual defining the exact quantization matrix for AMD Ryzen AI APUs, comparing perplexity, memory footprint, and NPU tile throughput across Q4_0, Q4_K, IQ4_NL, Q5_K, Q6_K, Q8_0, and F16, while documenting the explicit rejection policy for sub-4-bit quants.
- **Documentation Requirements:**
  - Complete `QUANTIZATION.md` artifact with tables and recommendations.
- **Dependencies:** TASK-046
- **Verification:** `QUANTIZATION.md` created at project root, detailing Q4–Q16 support, IQ4_NL non-linear codebook mapping, and sub-4-bit rejection rationale.





