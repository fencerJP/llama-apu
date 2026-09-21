<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors
-->

# Implementation Plan & Milestone Roadmap (`plan.md`)

This plan defines the engineering milestones, verification criteria, profiling commands, and deliverable artifacts for building the Heterogeneous APU Orchestrator runtime on AMD Ryzen AI processors.

---

## 1. Architectural Strategy & Phased Delivery

The delivery follows a strict progressive-validation roadmap:
1. Validate low-level Linux kernel UAPI zero-copy sharing on physical hardware (M1).
2. Integrate the sequential inference pipeline across iGPU and NPU (M2).
3. Enforce deterministic core pinning and dynamic cost scheduling (M3).
4. Deploy speculative drafting and advanced KV compression for high-tier APUs (M4).

---

## 2. Milestone Breakdown & Roadmaps

### Milestone 1 (M1): Zero-Copy Bridge Spike (Kernel UAPI Verification)
- **Objective:** Minimal standalone proof-of-concept verifying that physical DRAM pages allocated by AMDGPU can be exported via `dma-buf` and attached into AMDXDNA / XRT without CPU memcpy.
- **Key Deliverables:**
  - `src/uapi`: Low-level DRM, DMA-BUF, and AMDXDNA ioctl bindings.
  - `src/memory`: `DmaBufHandle` and `SharedBuffer` RAII abstractions.
  - Spike verification binary: `examples/zero_copy_spike.rs` exporting a 64MB buffer from `/dev/dri/renderD128` (AMDGPU GEM Prime FD) and importing it into `/dev/accel/accel0` (`DRM_AMDXDNA_CREATE_BO`).
- **Acceptance Gate:**
  - Zero memory copies verified via `perf trace -e dma_buf*`.
  - Kernel `dmesg` confirms successful IOMMU page table binding with zero fault flags.
  - CPU-side `DMA_BUF_IOCTL_SYNC` correctly manages cacheline flush/invalidate without page corruption.

### Milestone 2 (M2): Static Pipeline Integration (iGPU Prefill $\rightarrow$ XDNA Decode)
- **Objective:** End-to-end execution of a single inference query across accelerators for fixed-length prompts.
- **Key Deliverables:**
  - `PrefillEngine`: Dispatching GEMM kernels on RDNA 3.5 via ROCm/HIP.
  - `DecodeEngine`: Dispatched autoregressive token generation on XDNA 2 AIE2P tiles via XRT native API.
  - `SyncPort`: DRM timeline syncobj signaling completion fence from iGPU directly into NPU mailbox.
  - Static KV-cache buffer backing the sequence.
- **Acceptance Gate:**
  - Continuous token generation matching greedy reference output from stock Llama-3-8B.
  - Inter-token latency (ITL) $\le 35$ ms/token on AMD Strix Point.
  - Zero host-side data copies during token loop transitions.

### Milestone 3 (M3): Topology Awareness & Dynamic Cost Governor
- **Objective:** Eliminating core migration jitter, enabling deep CPU C-states, and monitoring accelerator queue depths.
- **Key Deliverables:**
  - `ApuTopologyGovernor`: Sysfs probe differentiating Zen 5 Classic cores (5.1 GHz boost, 16MB L3) and Zen 5c Compact cores (3.3 GHz, shared compact L3).
  - Thread Affinity Pinning: Feeder locked to Zen 5 Classic; EventPoller locked to Zen 5c Compact.
  - Dynamic Cost Engine: Monitoring memory bandwidth saturation and thermal throttling.
  - OpenAI API Daemon: Axum-based HTTP/SSE server supporting `/v1/chat/completions` with constant-time Bearer authentication and seccomp sandboxing.
- **Acceptance Gate:**
  - Package power during autoregressive decode drops to $\le 18$W (measured via RAPL / `turbostat`).
  - ITL standard deviation drops by $> 60\%$ compared to unpinned OS scheduling.
  - P99 API overhead $\le 1.2$ ms.

### Milestone 4 (M4): Advanced Schedulers & Speculative Drafting (Strix Halo)
- **Objective:** Maximize throughput and long-context efficiency using speculative decoding and KV pruning.
- **Key Deliverables:**
  - `SpeculativeDraftingEngine`: NPU drafts $K=4$ candidate tokens; RDNA 3.5 iGPU verifies all $K$ candidates in a single batched GEMM pass.
  - SnapKV / Dynamic Window KV Pruning: Pruning low-attention heads on the fly to prevent UMA bus saturation on contexts $> 4096$ tokens.
  - 256-bit memory bus optimizations for Strix Halo (Radeon 8060S with 40 CUs).
- **Acceptance Gate:**
  - Effective generation speed $> 65$ tokens/second on Strix Halo.
  - Dynamic memory footprint reduced by $> 40\%$ on 8k context sequences.

### Milestone 5 (M5): `apu-backend` for `llama.cpp` Fork & Unified `.q4nx` Container
- **Objective:** Enable seamless, single-file model distribution and drop-in `llama.cpp` compatibility for all AMD Ryzen AI APUs.
- **Key Deliverables:**
  - **Unified `.q4nx` Binary Container:** Serialization format embedding the exact `.xclbin` spatial graph directly into the file header alongside 64-byte aligned `q4nx` weights.
  - **Auto-Matching & Interactive Selector:** Automatic matching against the 37 local XCLBIN profiles with interactive CLI fallback selection.
  - **Persistent Disk Cache:** One-time conversion from standard GGUF to `.q4nx` saved to disk for instant subsequent loads.
  - **Rust C-ABI Export Layer:** `include/apu_backend.h` exposing clean C symbols from `libzero_copy_model_runner.a` without rewriting Rust logic in C++.
  - **`llama.cpp` Fork Integration:** Out-of-tree fork of `llama.cpp` registering `apu-backend` in CMake.
- **Acceptance Gate:**
  - `apu-run` compiles and initializes `.q4nx` container with embedded XCLBIN submitted to `/dev/accel/accel0`.
  - Zero usage of the phrase "ggml" across any components, headers, or build artifacts.

### Milestone 6 (M6): Production Mathematical Engine (Zero-Mock Forward Pass)
- **Objective:** Eliminate all test oracles, simulated math loops, and weight truncation shortcuts; implement genuine neural network execution across iGPU, CPU, and NPU.
- **Key Deliverables:**
  - **Full GGUF Tensor Memory Mapping:** Replace 32MB payload cap in `src/container/converter.rs` with complete tensor directory parsing, weight unpadding, and zero-copy mmap of all model layers (4.5GB+ for Llama-3-8B).
  - **RDNA 3.5 iGPU GEMM Kernel:** Replace fake `sin()` sleep loop in `src/engine/rocm_prefill.rs` with real batched GEMM / FlashAttention forward pass via ROCm/HIP writing genuine FP16 Key-Value states into shared `dma-buf`.
  - **XDNA 2 NPU GEMV Kernel:** Replace `DeterministicReferenceOracle` in `src/engine/xrt_decode.rs` with real AIE2P matrix-vector multiplication forward step consuming weights from DRAM and KV cache from `dma-buf`.
  - **AVX-512 Logit Sampling Pipeline:** Feed real logits into `src/engine/sampler.rs` to generate authentic vocabulary tokens with temperature, top-k, and top-p filtering.
- **Acceptance Gate:**
  - 100% elimination of `DeterministicReferenceOracle` from production paths.
  - Generates fully coherent, grammatically correct English across arbitrary prompt lengths.
  - Perplexity on Wikitext-2 within $1.05\times$ of stock llama.cpp reference.

### Milestone 7 (M7): Full Drop-In Replacement for `llama-cli` (`llama-cli` / `llama`)
- **Objective:** Deliver `llama-cli` (with `llama` alias) as a 100% binary-compatible drop-in replacement.
- **Key Deliverables:**
  - **Complete Flag Parsing:** Support `-m`, `-p`, `-f`, `-n`, `-c`, `-b`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed`, etc.
  - **Hardware Routing Flags:** Support `--gpu-based`, `--cpu-based`, `--npu-based`, and per-step flags (`--tokenize`, `--prefill`, `--decode`, `--sample`).
  - **Interactive Mode & Chat Templates:** Full Jinja2 chat templating for Llama-3, Qwen 2.5, DeepSeek R1, Gemma 2, and interactive REPL session management.
  - **Grammar & Structured Output:** GBNF grammar parser integration for guaranteed valid JSON schema outputs and structured tool calling.
- **Acceptance Gate:**
  - Passes 100% of standard `llama-cli` functional tests.
  - Passes automated tool-calling benchmark with valid JSON function arguments.

### Milestone 8 (M8): Full Drop-In Replacement for `llama-server` (`llama-server`)
- **Objective:** Deliver `llama-server` as a 100% OpenAI API-compatible drop-in server daemon.
- **Key Deliverables:**
  - **OpenAI Endpoints:** `POST /v1/chat/completions` (SSE streaming + non-streaming), `POST /v1/completions`, `POST /v1/embeddings`, `GET /v1/models`, `GET /health`, `GET /metrics`.
  - **Hardware Routing Flags:** Support `--gpu-based`, `--cpu-based`, `--npu-based`, and per-step flags in server configuration.
  - **Function & Tool Calling:** Support for OpenAI `tools` and `tool_choice` schema execution with zero-copy APU acceleration.
  - **Multi-Slot Continuous Batching:** Parallel slot manager multiplexing concurrent client requests with shared zero-copy KV cache blocks.
- **Acceptance Gate:**
  - Passes official OpenAI Python SDK client suite without code modifications.
  - Fully interoperable with Open WebUI, Ollama client, LangChain, and LlamaIndex.

### Milestone 9 (M9): External User Readiness, Packaging & Hardware Doctor
- **Objective:** Provide a seamless out-of-the-box onboarding experience for external users with automatic diagnostic checks, one-command model downloads, and system packaging.
- **Key Deliverables:**
  - **Hardware Doctor Tool (`apu-doctor` / `llama-cli --doctor`):** Comprehensive pre-flight system inspection validating `/dev/kfd`, `/dev/dri/renderD128`, `/dev/accel/accel0`, kernel driver versions, `render`/`video` group memberships, and `memlock` ulimits with self-healing advice.
  - **Model Hub Downloader (`apu-model`):** Automated retrieval, checksum verification, and turnkey XCLBIN stamping for models from HuggingFace / FastFlowLM.
  - **Deployment Packaging:** Production install script (`scripts/install.sh`) installing binaries directly as `llama-cli` (with `llama` alias) and `llama-server`, Systemd service unit (`llama-server.service`), and Debian/Arch packaging.
  - **Comprehensive Documentation Suite:** `QUICKSTART.md`, `HARDWARE_SUPPORT.md`, `CLI_GUIDE.md`, and `TROUBLESHOOTING.md`.
- **Acceptance Gate:**
  - `apu-doctor` runs on clean system and outputs actionable diagnostics.
  - New user can clone repository, run `install.sh`, and execute `llama-cli` on a downloaded model in under 2 minutes.

### Milestone 10 (M10): Upstream `llama.cpp` Rewiring & Backend Integration
- **Objective:** Connect the Rust `apu-backend` acceleration library (`libzero_copy_model_runner.so`) directly into upstream C++ `llama.cpp`, retiring scratch Rust front-end prototypes and enabling native upstream `llama-cli` and `llama-server` execution on AMD APUs.
- **Key Deliverables:**
  - **C-ABI Compute Offload Hook:** Intercept upstream `llama.cpp` tensor graph evaluation, routing prompt prefill to the RDNA 3.5 iGPU and decode steps to the XDNA 2 NPU via `include/apu_backend.h`.
  - **Rewired Upstream `llama-cli`:** Ensure `llama-cli` dispatches compute across the APU, using upstream's battle-tested tokenizers, Jinja chat templates, and GBNF grammars.
  - **Rewired Upstream `llama-server`:** Multi-client OpenAI API daemon with zero-copy `dma-buf` KV slot caching.
  - **Multi-Quantization Ingestion:** Converter support in `apu-model` for unusual non-linear importance-matrix quants (`IQ4_NL`, `IQ4_XS`, `IQ3_XXS`) into canonical `.q4nx`.
  - **End-to-End Intelligibility Suite:** Automated multi-model validation verifying 100% intelligible English output matching upstream reference.
- **Acceptance Gate:**
  - Upstream `llama-cli -m model.gguf -p "What is the capital of France?"` outputs "The capital of France is Paris." with APU hardware acceleration.
  - Passes 100% of end-to-end intelligibility tests with zero garbled tokens.

---

## 3. Profiling Commands & Verification Test Harness

### Automated Hardware Profiling Commands
```bash
# 1. Zero-Copy Invariant Verification (Kernel Tracepoints)
sudo perf record -e dma_buf:dma_buf_sync_start,dma_buf:dma_buf_sync_end,kmem:mm_page_alloc -p $(pgrep zero-copy-runner) -- sleep 10
sudo perf script | grep -E "dma_buf|memcpy"

# 2. Package Wattage & Deep C-State Verification
sudo turbostat --quiet --Summary --interval 1 --show PkgWatt,CorWatt,GFXWatt,Pkg_%pc6,CPU%c6

# 3. CPU Core Migration Audit
perf stat -e migrations,context-switches,cache-misses -p $(pgrep zero-copy-runner) -- sleep 10

# 4. Memory Bus Contention & Roofline Saturation
sudo perf stat -e amd_l3/total_cache_accesses/,amd_l3/total_cache_misses/ -a -- sleep 10

# 5. End-to-End Latency Benchmark (OpenAI Streaming API)
python3 -m tests.benchmark_client --endpoint http://localhost:8000/v1/chat/completions --prompt-tokens 512 --max-tokens 128 --concurrency 8
```

---

## 4. Verification Gates Matrix

| Milestone | Gate Criteria | Metric Target | Verification Tool |
|---|---|---|---|
| **M1** | DMA-BUF Export/Import | 0 host memcpys, 0 IOMMU faults | `dmesg`, `bpftrace`, `perf` |
| **M2** | Pipeline Latency & Equivalence | TTFT < 45 ms, ITL < 28 ms | Test harness vs HuggingFace logits |
| **M3** | Package Energy Reduction | Package wattage $\le 18$W | `turbostat`, Linux RAPL energy-pkg |
| **M3** | Core Pinning Jitter | Zero CCX cross-migrations | `perf stat -e migrations` |
| **M4** | Speculative Decoding Speedup | $\ge 1.8\times$ baseline decode tok/s | Benchmark client streaming test |
| **M5** | Naming Compliance | 0 instances of 'ggml' in runtime names | `grep -rI "ggml"` |
| **M6** | Mathematical Engine Integrity | Real logit generation; zero test oracles | Perplexity $\le 1.05\times$ reference |
| **M7** | Drop-In CLI Parity | 100% flag parity; intelligible chat & tool calling | `apu-cli` functional test suite |
| **M8** | Drop-In Server Parity | OpenAI API client compatibility | OpenAI SDK integration test suite |
| **M9** | External User Readiness | One-line installer, `apu-doctor`, clean setup < 2m | `apu-doctor`, fresh OS install test |
| **M10** | Upstream Rewiring & Intelligibility | 100% intelligible English; 0 garbled tokens; zero-copy offload | Upstream `llama-cli` & `llama-server` test suite |


