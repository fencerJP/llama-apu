<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors
-->

# Constitution & Invariant Principles: Heterogeneous APU Orchestrator

This document establishes the binding, non-negotiable architectural principles, operational invariants, and security boundaries for the Heterogeneous APU Orchestrator runtime targeting AMD Ryzen AI processors (Strix Point, Krackan Point, Gorgon Point, and Strix Halo) on Linux.

---

## Principle 1: Absolute Zero-Copy Invariant (`DMA-BUF` Sharing)

1. **Host-Side Buffer Copy Prohibition:**
   - At no point during the execution lifecycle (model loading, prompt prefill, autoregressive decoding, KV-cache updates, or token sampling) shall tensor weights, activation matrices, or Key-Value (KV) cache entries be duplicated in system memory via host `memcpy`, CPU loops, or intermediary bounce buffers.
   - Physical page frames allocated for model weights and KV-cache blocks must reside in shared unified system memory (LPDDR5X) backing Linux DRM Graphics Execution Manager (GEM) buffer objects.
2. **UAPI Cross-Device Export/Import:**
   - Inter-accelerator buffer transfer between AMDGPU (RDNA 3.5 iGPU) and AMDXDNA (XDNA 2 NPU) must exclusively occur through standard Linux `dma-buf` file descriptors (`/dev/dma_buf` and Prime FDs).
   - The orchestrator imports these buffers into the Xilinx Runtime (`xrt::bo(device, fd, xrt::bo::flags::cacheable, 0)`) such that both the GPU Command Processor and the NPU AIE2P tile DMA controllers point directly to identical physical Page Frame Numbers (PFNs).
3. **Memory Bus Conservation:**
   - On Unified Memory Architecture (UMA) systems, memory bus bandwidth is the hard ceiling for autoregressive decoding throughput. Any redundant data movement directly burns shared LPDDR5X bandwidth (136.5 GB/s on 128-bit, 273 GB/s on 256-bit), degrading token generation speed across all cores.

---

## Principle 2: User-Space Boundary & Upstream UAPI Adherence

1. **Kernel Integrity:**
   - The runtime must operate strictly in user space (`libdrm`, `amdgpu` UAPI, `amdxdna` UAPI, XRT user-mode SHIM `xrt_plugin.*-amdxdna.so`).
   - The runtime shall never modify, patch, or bypass the Linux kernel DRM scheduler (`drm_sched`), CPU schedulers (EEVDF/CFS), or IOMMU page table drivers.
2. **Standard Acceleration Drivers:**
   - Direct all GPU operations through ROCm/HIP / Vulkan via `/dev/dri/cardX` and `/dev/dri/renderD128`.
   - Direct all NPU operations through AMDXDNA via `/dev/accel/accel0` (`DRM_AMDXDNA_*` ioctls).
3. **Explicit Hardware Synchronization:**
   - Cross-accelerator dependencies (iGPU prefill completion -> NPU decode dispatch) must be synchronized via Linux DRM synchronization objects (`drm_syncobj`) and hardware `dma_fence` primitives.
   - User-space CPU polling loops on memory flags for cross-device completion are strictly forbidden. The orchestrator must wait on timeline fences via `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` or eventfds without busy-spinning host threads.

---

## Principle 3: Microarchitectural Topology Awareness & Core Affinity

1. **Deterministic Core Specialization:**
   - The runtime must discover and differentiate between Zen 5 Classic cores (large L3 cache, full dual-pipe AVX-512) and Zen 5c Compact cores (shared compact L3, optimized for sustained low-power execution) via `hwloc` or `/sys/devices/system/cpu/`.
2. **Workload Pinning Invariant:**
   - **Feeder & Tokenization Thread:** Must be pinned via `pthread_setaffinity_np` / `sched_setaffinity` exclusively to Zen 5 Classic core(s) (e.g., Core 0-3) to maximize vector performance for BPE/WordPiece tokenization and avoid L3 thrashing.
   - **Event Polling & Worker Threads:** Low-power event wait threads, asynchronous token streaming, and I/O event loops must be pinned to Zen 5c Compact cores.
   - **Cross-CCX Migration Prevention:** OS thread migration across CCX/NUMA domain boundaries is prohibited during an active inference session to prevent L3 cache flushes, cache invalidation storms, and Inter-Token Latency (ITL) jitter.
3. **Deep C-State Concurrency:**
   - During autoregressive decoding on the NPU, compute-heavy Zen 5 Classic cores and the RDNA 3.5 iGPU must enter low-power C-states and clock-gated power planes, driving package power reduction up to 70%.

---

## Principle 4: Layer & Phase Disaggregation Flexibility

1. **Dynamic Phase & Layer Assignment:**
   - The orchestrator must support heterogeneous partitioning:
     - Phase-level disaggregation: Prefill on iGPU $\rightarrow$ Decode on NPU.
     - Layer-level disaggregation: Early layers on iGPU/CPU, attention/MLP layers on NPU, final logit projection on CPU/iGPU.
     - Speculative drafting: NPU drafting $K$ candidate tokens $\rightarrow$ iGPU batched verification in parallel.
2. **Continuous In-Memory Model Residency:**
   - Model weights must be memory-mapped (`mmap`) once from disk using safe read-only private/shared mappings, directly imported into GPU and NPU address spaces. Dynamic reloading or copying across inference requests is prohibited.

---

## Principle 5: Standard Endpoint Security & Host Isolation

1. **OpenAI API Standard Compliance:**
   - Serve inference via standard OpenAI-compatible REST endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) supporting server-sent events (SSE) token streaming.
2. **Defense-in-Depth Security:**
   - **Authentication:** Mandatory Bearer token validation with configurable constant-time verification.
   - **Input Sanitation & Resource Clamping:** Strict validation on prompt token length, max generation tokens, temperature bounds, and stop tokens to prevent memory exhaustion and DoS attacks.
   - **Process Isolation:** Drop root privileges post-initialization; restrict file system access with Linux `landlock` or `seccomp` system call filters.
   - **Memory Sandboxing:** Device node permissions strictly validated (`/dev/dri/*`, `/dev/accel/*`); verify IOMMU PASID isolation to prevent unauthorized cross-context device DMA.

---

## Principle 6: Full Drop-In Replacement for llama.cpp (CLI & Server)

1. **Drop-In Parity Mandate:**
   - The runtime must serve as a 100% seamless, binary- and API-compatible drop-in replacement for `llama.cpp` on AMD Ryzen AI APU systems.
   - **CLI Replacement (`llama-cli` / `llama`):** The primary command-line binary is named `llama-cli` (and optionally symlinked/aliased to `llama`), accepting identical flags (`-m`, `-p`, `-f`, `-n`, `-c`, `-b`, `--temp`, `--top-k`, `--top-p`, `--chat-template`, `--interactive`, `--grammar`, etc.), supporting full multi-turn conversational chat, and executing structured tool calling with valid JSON outputs.
   - **Server Replacement (`llama-server`):** The primary server daemon is named `llama-server`, providing 100% compliant OpenAI REST/SSE endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`), full multi-slot execution, and native function/tool calling.
2. **Software Naming & Attribution Policy:**
   - **New Code Naming Restriction:** New software, libraries, modules, and internal packages created within this project must strictly avoid using the "ggml" phrase in their names to prevent any false impression of representing or speaking for the GGML organization. Components created by this project must be named under `apu-backend` or `zero-copy-model-runner`. The command-line executables maintain the standard `llama-cli`, `llama`, and `llama-server` names.
   - **Attribution of Upstream Software:** Code, interfaces, headers, and types authored by the original GGML project (e.g., `ggml.h`, `ggml_tensor`, upstream `llama.cpp` structures) must retain their legitimate original names and attributions.

---

## Principle 7: Zero-Mock & Production Mathematical Integrity

1. **Absolute Prohibition of Test Mocks in Production Paths:**
   - No mock oracles, hardcoded token arrays (`DeterministicReferenceOracle`), synthetic math accumulators (`sin()` loops), simulated execution sleep intervals (`std::thread::sleep`), or fake KV cache byte fillers (`token % 251`) are permitted in production code paths.
2. **Real Weight Ingestion & Exact Forward Passes:**
   - All multi-gigabyte GGUF / `.q4nx` model weights must be memory-mapped in their entirety without artificial size capping.
   - Forward passes must compute genuine neural network transformations: RMSNorm, Rotary Positional Embeddings (RoPE), Q4_K dequantization, batched GEMM on RDNA 3.5 iGPU, autoregressive GEMV on XDNA 2 NPU, and true logit probability distributions.

---

## Principle 8: XDNA 2 Exclusive Support Matrix, Hardware Routing & Documentation

1. **Exclusively AMD XDNA 2 Silicon Architecture:**
   - This project strictly targets AMD Ryzen AI APUs featuring **XDNA 2 NPU (AIE2P)** silicon. Legacy XDNA 1 devices (Phoenix, Hawk Point) are explicitly deprecated and unsupported.
   - Supported XDNA 2 APU Silicon Families:
     - **Strix Point & Gorgon Point (32-tile AIE2P, 16 CUs RDNA 3.5):** Default is Prefill on RDNA 3.5 iGPU, Autoregressive Decode on XDNA 2 NPU, Tokenize/Sample on Zen 5 CPU.
     - **Strix Halo (32-tile AIE2P, 40 CUs RDNA 3.5, 273 GB/s UMA Bus):** Default is Prefill on iGPU, Decode on iGPU (Maximum Throughput) or NPU (Power Saving), Tokenize/Sample on Zen 5 CPU.
     - **Krackan Point (16-tile AIE2P, 8 CUs RDNA 3.5):** Default is Prefill on iGPU, Decode on XDNA 2 NPU, Tokenize/Sample on Zen 5 CPU.
2. **User Hardware Routing Authority:**
   - The user retains ultimate authority to override accelerator assignments via command-line flags:
     - **Macro Accelerator Presets:** `--gpu-based` (route all possible stages to iGPU), `--cpu-based` (100% Zen 5 AVX-512 CPU execution), and `--npu-based` (maximize XDNA 2 NPU utilization).
     - **Granular Per-Step Overrides:** `--tokenize <cpu|gpu>`, `--prefill <gpu|cpu|npu>`, `--decode <npu|gpu|cpu>`, and `--sample <cpu|gpu>`.
3. **Mandatory Documentation Invariant Across All Steps:**
   - **In-Code Documentation:** Comprehensive module documentation (`//!`), function/struct docstrings (`///`), and Doxygen comments (`/** */`) explaining memory layouts, alignment constraints, and hardware invariants.
   - **CLI UX Documentation:** Exhaustive, user-friendly `--help` descriptions with usage examples; detailed `--verbose` telemetry exposing timeline fence transitions, DMA-BUF file descriptors, buffer addresses, and per-stage latency breakdowns.
   - **Top-Level Documentation:** Continuous synchrony of `README.md`, `QUICKSTART.md`, `constitution.md`, `spec.md`, `c4_architecture.md`, `plan.md`, and `tasks.md` with every deliverable.

---

## Governance & Compliance Verification

| Check | Tool / Mechanism | Pass Condition |
|---|---|---|
| Zero-Copy Integrity | `bpftrace` / kernel tracepoints (`dma_buf_map_attachment`, `kfree`) | Zero host buffer allocations/copies during inference loop |
| UAPI Boundary | `strace -e ioctl` | Only standard DRM, AMDXDNA, and DMA-BUF ioctls invoked |
| Thread Affinity | `taskset -cp <pid>`, `perf sched` | Zero migrations of feeder threads out of designated Zen 5 cores |
| Security Policy | Synthetic fuzzing + unauthorized token injection | 100% rejection with HTTP 401/400; zero segfaults or OOM panics |
| Mathematical Integrity | Logit verification against reference PyTorch / llama.cpp | Perplexity $\le 1.05\times$ reference; 100% intelligible English; valid tool calls |
| Drop-In Parity | `llama-cli` & `llama-server` test harness suites | 100% flag and endpoint compatibility without crashes |

