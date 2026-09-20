<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors
-->

# Technical Specification: Heterogeneous APU Orchestrator (`spec.md`)

This document defines the functional and non-functional specifications, silicon microarchitecture constraints, execution lifecycle, interface protocols, and verification metrics for the Heterogeneous APU Orchestrator runtime.

---

## 1. Executive Summary & Problem Scope

Modern AMD Accelerated Processing Units (APUs)—including Strix Point (Ryzen AI 300), Krackan Point, Gorgon Point, and Strix Halo—combine high-performance Zen 5 and Zen 5c CPU cores, an RDNA 3.5 integrated GPU (iGPU), and an XDNA 2 Neural Processing Unit (NPU) sharing a single unified LPDDR5X memory subsystem.

Traditional LLM inference frameworks (e.g. stock `llama.cpp` or vLLM) suffer from two critical architectural inefficiencies on APUs:
1. **The Prefill-Decode Conflict:** Running compute-heavy prompt prefill and memory-bandwidth-bound autoregressive decoding on the same engine causes severe tail latency spikes or wastes massive electrical power (35W–54W+ package draw for single-token decode).
2. **Buffer Duplication:** Transferring activations and KV cache across engines using host-side memory copies exhausts the shared memory bus (which has a hard bandwidth limit of 136.5 GB/s to 273 GB/s), severely crippling token generation speed.

The **Heterogeneous APU Orchestrator (`apu-backend` / `zero-copy-model-runner`)** is a full, production-grade drop-in replacement for `llama.cpp` on AMD Ryzen AI APUs. It provides:
- A full drop-in replacement for `llama-cli` (`apu-cli` / `llama-cli`) supporting identical CLI flags, interactive chat, grammars, and structured tool calling.
- A full drop-in replacement for `llama-server` (`apu-server` / `llama-server`) supporting identical OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`), token streaming, and multi-slot concurrency.
- A true zero-copy iGPU-CPU-NPU heterogeneous inference pipeline:
  - **RDNA 3.5 iGPU:** High-throughput prompt prefill via ROCm/HIP.
  - **XDNA 2 NPU:** Energy-efficient autoregressive decode (AIE2P spatial array) via XRT.
  - **Zen 5 CPU:** AVX-512 vector tokenization, orchestrator state machine, and zero-copy DMA-BUF memory management.
- Complete zero-copy KV cache and tensor sharing across iGPU, CPU, and NPU via Linux Prime DMA-BUF and DRM GEM without host-side data duplication.
- Absolute prohibition of the term "ggml" in our software, binary, library, or component naming.
- Zero mock oracles or simulated loops: 100% genuine neural network weight ingestion and forward-pass mathematical execution.

---

## 2. Silicon Microarchitecture Constraints Matrix (XDNA 2 Exclusive)

| Feature / Silicon | **Krackan Point** | **Strix Point** | **Gorgon Point** | **Strix Halo** |
|---|---|---|---|---|
| **Target Series** | Ryzen AI 300 Mobile | Ryzen AI 9 HX 370 | Next-Gen APU / Edge | Ryzen AI Max+ 395 |
| **CPU Microarchitecture** | 4x Zen 5 + 4x Zen 5c | 4x Zen 5 + 8x Zen 5c | Hybrid Zen 5 / 5c | 16x Zen 5 (Dual CCD) |
| **Total CPU Cores/Threads** | 8C / 16T | 12C / 24T | 8-12C / 16-24T | 16C / 32T |
| **CPU Vector / AVX** | Dual-pipe 512-bit FADD/FMUL | Dual-pipe 512-bit FADD/FMUL | Dual-pipe 512-bit | Dual-pipe 512-bit |
| **iGPU Architecture** | RDNA 3.5 (Radeon 840M) | RDNA 3.5 (Radeon 890M) | RDNA 3.5 Enhanced | RDNA 3.5 (Radeon 8060S) |
| **iGPU Compute Units** | 8 CUs (512 Stream Proc.) | 16 CUs (1024 Stream Proc.)| 12-16 CUs | 40 CUs (2560 Stream Proc.) |
| **iGPU FP16 Peak** | ~16 TFLOPs | ~32 TFLOPs | ~24-32 TFLOPs | ~80+ TFLOPs |
| **NPU Architecture** | **XDNA 2 (AIE2P)** | **XDNA 2 (AIE2P)** | **XDNA 2 (AIE2P)** | **XDNA 2 (AIE2P)** |
| **NPU Array Topology** | 4x4 Tiles (16 AIE2P) | 4x8 Tiles (32 AIE2P) | 4x8 Tiles (32 AIE2P) | 4x8 Tiles (32 AIE2P) |
| **NPU Tile Local Memory** | 2 MB Tile Data Mem | 4 MB Tile Data Mem | 4 MB Tile Data Mem | 4 MB Tile Data Mem |
| **NPU Peak INT4/BlockFP16**| ~30 TOPS | 50-55 TOPS | 50-55 TOPS | 50-55 TOPS |
| **Memory Bus Width** | 128-bit LPDDR5X-7500 | 128-bit LPDDR5X-8533 | 128-bit LPDDR5X-8533 | 256-bit LPDDR5X-8533 |
| **Theoretical Mem Bandwidth** | 120.0 GB/s | 136.5 GB/s | 136.5 GB/s | 273.1 GB/s |
| **Effective UMA Payload BW** | ~95 GB/s | ~110 GB/s | ~110 GB/s | ~220 GB/s |
| **Package TDP Envelope** | 15W - 28W | 28W - 54W | 20W - 45W | 55W - 120W |

### 2.1. XCLBIN Binary Portability & Cross-APU Compatibility Rules

1. **Target Architecture: Exclusively XDNA 2 Generation (`NPU2`)**:
   - Legacy XDNA 1 silicon (Phoenix / Hawk Point) is explicitly unsupported.
   - A compiled XDNA 2 AIE2P spatial graph (`.xclbin`) targeting the 32-tile array is **100% binary-portable across Strix Point (Ryzen AI 300 / HX 370), Krackan Point, and Gorgon Point**, and executes in 32-tile compatibility mode on **Strix Halo**.
2. **Native Block-FP16 & INT4 Tensor MAC Support**:
   - All XDNA 2 AIE2P tiles feature 512KB distributed memory tiles with native Block-FP16 and INT4 matrix-vector hardware MAC units.
3. **Compilation Latency Profile**:
   - **Template-Based Graph Synthesis & Assembly (Runtime)**: $\mathbf{1 - 3\text{ seconds}}$ on Strix/Gorgon Point Zen 5 cores. Parameterizes spatial memory strides, stream switch routing, and invokes `/usr/bin/xclbinutil`.
   - **Full Vitis Spatial Place-and-Route (Offline)**: $3 - 10\text{ minutes}$ when performing raw C++ AIE intrinsic kernel compilation and NP-complete routing.

---

## 3. Quantifiable Key Performance Indicators (KPIs)

Target Model Baseline: **Llama-3-8B-Instruct (Q4_K_M / Block FP16 Quantization, ~4.5 GB memory footprint)**.

| Metric | Colocated Baseline (CPU/iGPU only) | Target Specification (Heterogeneous Runtime) | Minimum Acceptance Gate |
|---|---|---|---|
| **Time-To-First-Token (TTFT)** (512 prompt tokens) | 280 ms (CPU) / 85 ms (iGPU) | **< 45 ms** (Strix Point 890M), **< 20 ms** (Strix Halo 8060S) | $\le 55$ ms (Strix Point) |
| **Inter-Token Latency (ITL)** (Decode phase, B=1) | 38 ms/tok (~26 tok/s) | **< 28 ms/tok (> 35 tok/s)** (Strix Point), **< 15 ms/tok (> 65 tok/s)** (Strix Halo) | $\ge 30$ tok/s (Strix Point) |
| **Autoregressive Package Power** | 45W - 65W (iGPU/CPU compute active) | **12W - 18W** (NPU decode active, Classic cores in C-state) | $\le 22$W package draw |
| **Energy Per Output Token** | ~1.8 Joules / token | **< 0.45 Joules / token** (75% energy reduction) | $\le 0.65$ Joules / token |
| **Zero-Copy Host Memory Traffic** | > 9 GB/s copy overhead | **0.00 GB/s host-side memcpy** | Verified via `perf c2c` / `bpftrace` |
| **API Request Concurrency** | Single-tenant blocking | **Up to 64 concurrent client connections** (pipelined) | 32 concurrent requests |
| **API Overhead (P99 latency add)**| N/A | **< 1.2 ms** overhead above raw forward pass | $\le 2.0$ ms P99 |

---

## 4. End-to-End Execution Lifecycle & Phase Transition

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Ingestion & Tokenization (Zen 5 Classic Core 0)               │
│ • Client sends HTTP POST /v1/chat/completions (Bearer Auth validated)  │
│ • Request dispatched to Feeder Thread (locked to Core 0)              │
│ • Fast BPE Tokenization executed via AVX-512 SIMD kernels              │
│ • Input token IDs written directly into shared GEM mapped memory       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Compute-Heavy Prompt Prefill (RDNA 3.5 iGPU)                  │
│ • Feeder thread issues prefill command packet to AMDGPU ring buffer    │
│ • ROCm/HIP kernel executes batched GEMM across prompt sequence         │
│ • Attention Key and Value tensors are written directly into            │
│   pre-allocated Linux dma-buf backing the Unified KV Cache             │
│ • GPU Command Processor signals DRM timeline syncobj fence             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼ [Hardware dma_fence signal, NO CPU copy]
                                   │
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: KV-Cache Handshake & NPU Activation (AMDXDNA UAPI)            │
│ • NPU ERT (Embedded Runtime) consumes signaled syncobj fence           │
│ • NPU imports dma-buf file descriptor as xrt::bo                       │
│ • IOMMU binds NPU SVA virtual address to shared physical memory pages  │
│ • Initial sampled logits produce Token T_0                             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Autoregressive Decoding Loop (XDNA 2 AIE2P Array)             │
│ • Zen 5 Classic cores and iGPU enter low-power C-states (C6/C7)        │
│ • Zen 5c Compact Core runs lightweight async event wait on /dev/accel0 │
│ • For each generation step:                                            │
│   1. Tile DMA controller fetches model weight slice + new KV vector    │
│   2. 32 AIE2P tiles execute VLIW 2048-bit matrix-vector math in SRAM   │
│   3. ERT signals token output event; appended to KV dma-buf            │
│   4. Output token pushed directly to SSE HTTP response stream          │
│ • Repeat until EOS token or max_tokens reached                         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Request Completion & Resource Reclamation                     │
│ • KV cache page offsets marked clean / returned to memory pool         │
│ • HTTP streaming connection cleanly closed with final usage stats      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Standard OpenAI API Endpoint & Security Specification

### 5.1 HTTP API Surface
The daemon exposes standard OpenAI-compatible endpoints:
- `POST /v1/chat/completions`: Streaming (`stream: true`) and non-streaming responses.
- `POST /v1/completions`: Legacy prompt completions.
- `GET /v1/models`: Enumerates currently loaded and configured APU models.
- `GET /health` & `GET /metrics`: Health checks and Prometheus metrics (TTFT, ITL, wattage, memory occupancy).

### 5.2 Security Architecture & Controls

1. **Authentication & Authorization:**
   - Pre-shared API Key (Bearer token) verified using constant-time string comparison (`crypto::subtle::constant_time_eq`) to mitigate timing attacks.
   - Optional mutual TLS (mTLS) for inter-service communication over local networks or Unix Domain Sockets (`/run/zero-copy-runner/runner.sock`) with standard POSIX file permissions (`0660`).
2. **Input Validation & DoS Protection:**
   - Maximum prompt size limit (e.g. 8192 tokens max, configurable) rejected with HTTP 413 if exceeded.
   - `max_tokens` clamped to prevent memory oversubscription.
   - Adaptive Token Bucket rate limiter per client IP / API key.
   - Request timeout guards: Prefill deadline (10 seconds) and Decode step timeout (2 seconds/token) before aborting device jobs.
3. **Privilege Separation & Sandboxing:**
   - Daemon initialization opens `/dev/dri/card*`, `/dev/dri/renderD128`, and `/dev/accel/accel0` file descriptors, then immediately drops all ambient Linux capabilities (`CAP_*`) using `libcap`.
   - Execution sandbox restricted via Linux `seccomp-bpf` to allow only essential syscalls (`read`, `write`, `ioctl`, `epoll_wait`, `futex`, `nanosleep`).
   - File system access locked down via `landlock` or read-only root mount.
4. **Device & Memory Isolation:**
   - Strict IOMMU Passthrough / SVA validation to prevent cross-process DMA bleed.
   - Zero-initialization of reused KV-cache buffers to prevent speculative cross-tenant data leakage.

---

## 6. Heterogeneous Phase and Layer Partitioning Specification

The orchestrator dynamically routes operations across CPU, iGPU, and NPU based on model topography and runtime metrics:

1. **Phase Disaggregation (Standard Pipeline):**
   - **Prompt Prefill:** Dispatched to RDNA 3.5 iGPU (high arithmetic intensity, compute-bound GEMM).
   - **Autoregressive Decode:** Dispatched to XDNA 2 NPU (memory-bound, low operational intensity, power-constrained).
2. **Layer-Level Hybrid Partitioning:**
   - **Embedding Layer & Tokenization:** Dispatched to CPU Zen 5 Classic cores (AVX-512).
   - **Early Transformer Layers (0 to $M$):** Executed on iGPU when prompt exceeds NPU tile SRAM batch capacity.
   - **Middle Transformer Layers ($M+1$ to $N$):** Executed on NPU AIE2P array.
   - **Final RMSNorm & Vocabulary LM-Head:** Dispatched to Zen 5 Classic core or iGPU depending on vocabulary size ($V = 128{,}000$).
3. **Speculative Drafting Pipeline (Strix Halo / High-TDP mode):**
   - **Draft Engine (NPU):** Generates $K=4$ candidate tokens using an INT4/BlockFP16 draft model.
   - **Verification Engine (iGPU):** Evaluates all $K$ tokens in a single parallel batched forward pass on the 40 CU RDNA 3.5 compute units.
   - **No Host Copy:** Candidate tokens and KV states share identical `dma-buf` descriptors.

---

## 7. Unified `.q4nx` Binary Container & `apu-backend` Integration

### 7.1 `.q4nx` File Format Specification
The `.q4nx` container is a self-describing, single-file model package that combines compiled AMD XDNA hardware microcode (`.xclbin`) with tile-interleaved quantized weight tensors (`Q4NX` / BlockFP16).

```
+-------------------------------------------------------------------------+
| Header Magic: 0x584E3451 ('Q', '4', 'N', 'X') (4 Bytes)                |
| Format Version: uint32 (4 Bytes)                                        |
| Architecture Name: char[32] (e.g. "llama", "qwen2", "gemma2")          |
| Model Hyperparameters: hidden_dim, heads, kv_heads, layers, vocab (40B) |
| Embedded XCLBIN Offset & Size: uint64 offset, uint64 size (16 Bytes)    |
| Tensor Index Offset & Size: uint64 offset, uint64 count (16 Bytes)      |
| Reserved Header Padding (to 256 bytes)                                  |
+-------------------------------------------------------------------------+
| Embedded XCLBIN Payload: Compiled AIE2P hardware dataflow graph binary  |
+-------------------------------------------------------------------------+
| Tensor Metadata Table: name, dims, type, 64-byte aligned payload offset |
+-------------------------------------------------------------------------+
| Tile-Interleaved Weight Tensors: 64-byte aligned DRAM payload           |
+-------------------------------------------------------------------------+
```

### 7.2 Model Ingestion & Persistent Caching Lifecycle
1. **Input Detection:** User specifies `--model <file>`:
   - **Case A (`.q4nx` file with embedded XCLBIN):** Reads embedded `.xclbin` from header $\rightarrow$ initializes NPU hardware context $\rightarrow$ `mmap`s weight payload into DRM GEM shared memory. Zero conversion, startup $< 100\text{ ms}$.
   - **Case B (Legacy / Bare `.q4nx` file without XCLBIN):** The runtime inspects model metadata, resolves the matching `.xclbin` from the local 37-profile bank (or prompts user if ambiguous), and **stamps the `.xclbin` into the file header by default**, making it permanently turnkey.
   - **Case C (`.gguf` file):**
     1. Architecture inspection: Extracts model architecture, layer count, and dimension metadata.
     2. XCLBIN lookup: Scans local library of 37 production XCLBIN profiles. If unmapped or ambiguous, prompts user interactively with a numbered list.
     3. Persistent Conversion: Repacks GGUF `Q4_K_M` weights into `q4nx` tile strides, embeds the matching `.xclbin` into the header by default, writes `<model>.q4nx` to disk, and executes. Future runs load the `.q4nx` file directly.

### 7.3 `apu-backend` Architecture & Software Naming Rules
- **Naming Rule & Attribution Policy:** New code, binaries, libraries, modules, and packages created in this project must NOT use the phrase "ggml" to avoid appearing as though we represent or speak for the GGML organization. Components created by this project must be named under `apu-backend`, `zero-copy-model-runner`, `apu-cli`, or `apu-server`. Existing code, interfaces, files, and types authored by the upstream GGML project (e.g. `ggml.h`, `ggml_tensor`, upstream headers) must maintain their original names as proper attribution.
- **C-ABI Export Layer:** `include/apu_backend.h` and `src/ffi.rs` export a clean, high-performance C ABI (`apu_backend_*`) connecting user applications to the underlying zero-copy engine.
- **Physical Accelerator Silicon Direct Path:** Connects directly to `/dev/dri/renderD128` (AMDGPU) and `/dev/accel/accel0` (AMDXDNA). Mock fallbacks are strictly disabled in production builds and reserved exclusively for unit test environments when explicitly invoked with `--mock`.

---

## 8. Full Drop-In Replacement Specification (`llama-cli` & `llama-server`)

### 8.1 Drop-In CLI Binary (`llama-cli` / `llama`)
The primary command-line binary is named `llama-cli` (with optional `llama` alias) and provides 100% flag and operational compatibility with upstream `llama-cli`:
- **Model Loading:** `-m, --model <path>` (supports `.q4nx` and `.gguf` with automatic turnkey conversion and persistent disk caching).
- **Prompt Execution:** `-p, --prompt <text>`, `-f, --file <path>`, `-e` (process escapes).
- **Generation Parameters:** `-n, --predict <N>`, `-c, --ctx-size <N>`, `-b, --batch-size <N>`, `--temp <T>`, `--top-k <K>`, `--top-p <P>`, `--min-p <P>`, `--repeat-penalty <N>`, `--seed <S>`.
- **Interactive & Multi-Turn Chat:** `-i, --interactive`, `--in-prefix <str>`, `--in-suffix <str>`, `--chat-template <str>` (Jinja2-compatible chat template formatting for Llama-3, Qwen, DeepSeek, Gemma).
- **Structured Tool Calling & Grammars:** `--grammar <str>`, `--grammar-file <file>`, `--json` for deterministic schema adherence and valid JSON tool call generation.
- **Hardware Diagnostic:** `--doctor` for automated pre-flight system inspection.
- **Output Quality Standard:** 100% intelligible, grammatically valid output matching standard LLM quality; passes all tool-calling evaluation suites.

### 8.2 Drop-In Server Binary (`llama-server`)
The primary server daemon is named `llama-server` and provides 100% drop-in REST/SSE API compatibility:
- **OpenAI Endpoints:**
  - `POST /v1/chat/completions`: Full streaming (`stream: true`) and non-streaming responses, multi-turn message arrays, structured function/tool definitions (`tools` and `tool_choice`).
  - `POST /v1/completions`: Raw prompt completion endpoint.
  - `POST /v1/embeddings`: Batch token embeddings.
  - `GET /v1/models`: Model registry enumeration.
  - `GET /health` & `GET /metrics`: Health status and Prometheus telemetry.
- **Multi-Slot & Parallel Processing:** Concurrent slot allocation with continuous batching and zero-copy KV cache block paging.
- **Drop-In Compatibility Target:** Able to be targeted directly by the official OpenAI Python SDK, LangChain, LlamaIndex, Open WebUI, and Ollama clients without modification.

### 8.3 Silicon Support Matrix & Per-APU Default Execution Profiles (XDNA 2 Only)
Because hardware capabilities and memory architectures vary across AMD XDNA 2 APU generations, the orchestrator automatically detects the host APU silicon and selects the optimal default execution policy:

| APU Family | Silicon / GPU / NPU | Memory Bus | Optimal Default Execution Profile | Rationale |
|---|---|---|---|---|
| **Strix Point** (Ryzen AI 9 HX 370/365) | 16 CUs RDNA 3.5 / 32-tile AIE2P (50 TOPS) | 128-bit LPDDR5X (136.5 GB/s) | **Prefill: iGPU \| Decode: NPU \| Tokenize: CPU** | 16 CUs provide 1,200+ t/s prefill; 32-tile NPU provides 35+ t/s decode at only 12W–18W package draw. |
| **Gorgon Point** | 12-16 CUs RDNA 3.5 / 32-tile AIE2P (50 TOPS) | 128-bit LPDDR5X (136.5 GB/s) | **Prefill: iGPU \| Decode: NPU \| Tokenize: CPU** | Same AIE2P array; optimal throughput/watt profile. |
| **Strix Halo** (Ryzen AI Max+ 395) | 40 CUs RDNA 3.5 / 32-tile AIE2P (50 TOPS) | 256-bit LPDDR5X (273.1 GB/s) | **Prefill: iGPU \| Decode: iGPU (Speed) or NPU (Power) \| Tokenize: CPU** | 40 CUs with 273 GB/s UMA bandwidth delivers 65–80+ tok/s decode on iGPU. Default defaults to max-speed (iGPU decode) with instant `--npu-based` switch for low-power operation. |
| **Krackan Point** (Ryzen AI 300) | 8 CUs RDNA 3.5 / 16-tile AIE2P (30 TOPS) | 128-bit LPDDR5X (120.0 GB/s) | **Prefill: iGPU \| Decode: NPU \| Tokenize: CPU** | Compact APU; NPU decode preserves battery and thermals over 8 CU iGPU. |

---

### 8.4 User Hardware Execution Routing Flags

To give developers total control over how inference runs across the APU, `llama-cli` and `llama-server` expose two tiers of hardware routing arguments:

#### 1. Holistic Accelerator Presets (Macro Flags)
Enables the user to route as much workload as possible to a specific APU accelerator:
- `--gpu-based`: Maximizes RDNA 3.5 iGPU utilization (Prefill: GPU, Decode: GPU, Tokenize: CPU/GPU, Sampling: GPU/CPU).
- `--cpu-based`: 100% CPU execution using Zen 5 AVX-512 SIMD vector pipelines for all stages (Tokenize: CPU, Prefill: CPU, Decode: CPU, Sampling: CPU). Operates completely independently of GPU/NPU drivers.
- `--npu-based`: Maximizes XDNA 2 NPU utilization (Decode: NPU, Prefill: NPU/GPU, Tokenize: CPU).

#### 2. Granular Per-Step Hardware Overrides
Allows surgical assignment of each stage in the LLM inference lifecycle:
- `--tokenize <cpu|gpu>` (Default: `cpu` via Zen 5 AVX-512)
- `--prefill <gpu|cpu|npu>` (Default: per-APU optimal, typically `gpu`)
- `--decode <npu|gpu|cpu>` (Default: per-APU optimal, `npu` on Strix Point, `gpu` on Strix Halo for peak speed)
- `--sample <cpu|gpu>` (Default: `cpu` via AVX-512 vector softmax)

---

### 8.5 External User Readiness & Distribution Architecture

To ensure turnkey adoption for external users and production deployment, the platform includes:

1. **Hardware Diagnostic Doctor Tool (`apu-doctor` / `apu-cli --doctor`):**
   - Probes hardware accessibility: `/dev/kfd`, `/dev/dri/renderD128`, `/dev/accel/accel0`.
   - Validates kernel driver versions (`amdgpu`, `amdxdna`).
   - Checks user group permissions (`video`, `render`).
   - Verifies system `memlock` ulimits and Hugepage availability (`/sys/kernel/mm/transparent_hugepage`).
   - Outputs self-healing instructions (e.g. `sudo usermod -aG render $USER`).
2. **Model Management CLI (`apu-model`):**
   - Direct download and inspection of GGUF and Q4NX models from HuggingFace / FastFlowLM.
   - Automatic SHA-256 validation and turnkey XCLBIN container stamping.
3. **Turnkey Packaging & Service Deployment:**
   - Standalone portable binary distribution (glibc static/dynamic) installed directly as `llama-cli` (with `llama` alias) and `llama-server`.
   - Ready-to-use Systemd unit file: `llama-server.service` for headless daemon deployment.
   - Debian package (`.deb`) and Arch Linux `PKGBUILD`.
4. **First-Class Framework Interoperability:**
   - 100% drop-in compatibility with the official OpenAI Python SDK (`openai.OpenAI(base_url="http://localhost:8080/v1")`).
   - Validated integration with Open WebUI, Ollama client, LangChain, and LlamaIndex.

---

### 8.6 Mandatory Documentation Across All Steps
Documentation is an uncompromising deliverable in every phase of development:
- **In-Code Documentation:** Every function, struct, and module must include detailed comments (`//!`, `///`, `/** */`) detailing data layouts, physical memory alignments (64-byte), and hardware synchronization contracts.
- **CLI UX Documentation (`--help` and `--verbose`):**
  - `--help` must provide clear, categorized flag descriptions with usage examples.
  - `--verbose` must output runtime hardware telemetry: topology details, GEM Prime DMA-BUF file descriptors, DRM timeline syncobj points, memory bus bandwidth metrics, and per-phase latency breakdowns (TTFT, ITL).
- **Top-Level Documentation:** All architectural guides (`README.md`, `QUICKSTART.md`, `constitution.md`, `spec.md`, `c4_architecture.md`, `plan.md`, `tasks.md`, `walkthrough.md`) must be kept continuously synchronized with code changes.





