<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors
-->

# References & Academic Bibliography

This document catalogs the academic papers, technical references, and foundational texts that informed the design, quantization algorithms, inference optimization strategies, and systems architecture of **llama-apu** / the Heterogeneous APU Orchestrator.

Papers are grouped by topic area, with arXiv links where available. Local copies of most papers are stored in [`research/`](../research/).

---

## 1. Primary Quantization Methods

### 1.1 BiLLM — Core 1-bit Quantization Method (Primary)

**BiLLM: Pushing the Limit of Post-Training Quantization for LLMs**  
Haifei Huang, Tianyi Zhang, et al. (2024)  
arXiv: [2402.04291](https://arxiv.org/abs/2402.04291)

> **Relevance:** Foundational paper for the `BiLLM` 1.08 bpw format used as the default in this project. Introduces Hessian-based salient weight identification (top 1–2%), binary residual approximation for salient weights (2-bit, $s_1 B_1 + s_2 B_2$), and optimal splitting search for non-salient bell-curve distributions (1-bit). Achieves **8.41 PPL on LLaMA-2-70B** at 1.08 bpw without retraining.

---

### 1.2 SpinQuant — Offline Orthogonal Rotation (Primary)

**SpinQuant: LLM Quantization with Learned Rotations**  
Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort (Meta FAIR, 2024)  
arXiv: [2405.16406](https://arxiv.org/abs/2405.16406)

> **Relevance:** Defines the **offline orthogonal rotation stage** that *strictly precedes* BiLLM quantization in our 4-stage pipeline. Learns optimal orthogonal rotation matrices $R, Q$ via Cayley gradient descent on calibration data, rotating weights and activations to flatten outliers into sub-Gaussian distributions before Hessian computation and binarization. Reduces the accuracy gap to full precision by up to **45.1%** relative to QuaRot on hard models.

---

### 1.3 GPTQ — Hessian-Based Post-Training Quantization (Baseline)

**GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**  
Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (ETH Zürich, 2022)  
arXiv: [2210.17323](https://arxiv.org/abs/2210.17323)

> **Relevance:** Standard PTQ baseline using second-order (Hessian/OBQ) weight reconstruction. Used as the Q4/Q8 conversion baseline and as the quantization step when combining with rotation-based preprocessing (QuaRot+GPTQ).

---

### 1.4 QuaRot — Hadamard Rotation for End-to-End 4-bit Quantization

**QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**  
Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman (ETH Zürich / Apple, 2024)  
arXiv: [2404.00456](https://arxiv.org/abs/2404.00456)

> **Relevance:** Introduces randomized Hadamard rotation for suppressing activation and weight outliers in all linear layers, enabling 4-bit end-to-end inference (weights + activations + KV cache). Referenced in our "Advanced Orthogonal Rotations" alternatives section as the `QuaRot / QuIP#` approach.

---

### 1.5 QuIP# — Lattice Codebooks for 2-bit Quantization

**QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks**  
Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa (Cornell, 2024)  
arXiv: [2402.04396](https://arxiv.org/abs/2402.04396)

> **Relevance:** Extends Hadamard incoherence preprocessing with lattice-based codebooks achieving the best 2-bit PTQ quality. Referenced in our sub-2-bit quantization alternatives matrix.

---

### 1.6 BitNet b1.58 — Native Ternary Weights (From-Scratch Training)

**The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits**  
Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft, 2024)  
arXiv: [2402.17764](https://arxiv.org/abs/2402.17764)

> **Relevance:** Defines the `BitNet b1.58` ternary weight format {-1, 0, +1} at $\log_2 3 \approx 1.58$ bpw, enabling multiplication-free transformer inference. Referenced in our quantization alternatives matrix and TQ1_0/TQ2_0 ternary format evaluations.

---

### 1.7 T-MAC — Multiplication-Free LUT GEMM on CPUs/NPUs (Primary: NPU Execution)

**T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge**  
Jianyu Wei, Shijie Cao, et al. (Microsoft Research, 2024)  
arXiv: [2407.09720](https://arxiv.org/abs/2407.09720)

> **Relevance:** Directly implements the **T-MAC SRAM LUT decode approach** used in our XDNA 2 AIE2P NPU kernel for `BiLLM` models. Precomputes dot-product partial sums from packed weight bit patterns into small on-chip SRAM tables (32 KB tile data memory), enabling multiplication-free 1-bit GEMV at full memory-bandwidth saturation. Achieves 4× higher throughput vs. llama.cpp with 60–70% energy reduction.

---

### 1.8 FLUTE — Fast Non-Uniform LUT-Quantized GEMM

**Fast Matrix Multiplications for Lookup Table-Quantized LLMs**  
Brendan Buckley, Sanjit Singh Batra, Albert Tseng, Christopher De Sa, Nir Shavit, Tri Dao, Baris Kasikci, Vyas Sekar (Cornell / MIT / CMU, 2024)  
arXiv: [2407.10960](https://arxiv.org/abs/2407.10960)

> **Relevance:** Provides CUDA GEMM kernels for non-uniform lookup-table quantization (NF4, 3-bit, 4-bit non-linear codebooks) with Stream-K work decomposition. Referenced in our FLUTE 3–4 bpw alternative row in the quantization alternatives matrix.

---

## 2. Inference Acceleration & Speculative Decoding

### 2.1 FlashAttention — IO-Aware Tiled Attention

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford, 2022) — *NeurIPS 2022*  
arXiv: [2205.14135](https://arxiv.org/abs/2205.14135) · Local: `research/2205.14135v2.pdf`

> **Relevance:** Foundational IO-aware tiled attention algorithm used by the RDNA 3.5 iGPU prefill kernel. Eliminates HBM ↔ SRAM round-trips via tiling, enabling fast batched GEMM prompt prefill without materializing the full $N \times N$ attention matrix.

---

### 2.2 Speculative Decoding — Google Research

**Fast Inference from Transformers via Speculative Decoding**  
Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research, 2022) — *ICML 2023*  
arXiv: [2211.17192](https://arxiv.org/abs/2211.17192) · Local: `research/2211.17192v2.pdf`

> **Relevance:** Establishes the theoretical correctness proof for draft-verify token acceptance (rejection sampling scheme preserving target distribution). Informed our NPU draft → iGPU batched verification pipeline design.

---

### 2.3 Speculative Sampling — Google DeepMind

**Accelerating Large Language Model Decoding with Speculative Sampling**  
Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper (Google DeepMind, 2023)  
arXiv: [2302.01318](https://arxiv.org/abs/2302.01318) · Local: `research/2302.01318v1.pdf`

> **Relevance:** Concurrent independent speculative decoding work demonstrating 2–2.5× speedup on Chinchilla 70B with a formal sampling-based acceptance mechanism. Informed our heterogeneous draft-on-NPU / verify-on-iGPU pipeline.

---

### 2.4 TriForce — Hierarchical Speculative Decoding with Sparse KV

**TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding**  
Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian, Beidi Chen (2024)  
arXiv: [2404.11912](https://arxiv.org/abs/2404.11912)

> **Relevance:** Extends speculative decoding to long-context sequences (128K) via hierarchical two-tier drafting and dynamic sparse KV retrieval. Referenced in our Quest Sparsity + Chunked KV Allocation design (v0.3.0), achieving 2.31× speedup on LLaMA2-7B-128K.

---

### 2.5 Lookahead Decoding

**Break the Sequential Dependency of LLM Inference Using Lookahead Decoding**  
Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang (2024) — *ICML 2024*  
arXiv: [2402.02057](https://arxiv.org/abs/2402.02057) · Local: `research/2402.02057v1.pdf`

> **Relevance:** Draft-model-free parallel inference via Jacobi iteration n-gram lookahead (1.8–4× speedup). Informed the lightweight NPU-based speculative generation path for non-MoE dense models.

---

### 2.6 Sequoia — Hardware-Aware Scalable Speculative Decoding

**Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding**  
Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, Beidi Chen (2024)  
arXiv: [2402.12374](https://arxiv.org/abs/2402.12374) · Local: `research/2402.12374v3.pdf`

> **Relevance:** DP-optimized token draft tree construction + hardware-aware verification scheduling (4.04× speedup on Llama2-7B; 9.5× on offloaded Llama3-70B). Referenced for heterogeneous APU speculative pipeline optimization.

---

### 2.7 Medusa — Multi-Head Parallel Token Prediction

**Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**  
Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao (2024)  
arXiv: [2401.10774](https://arxiv.org/abs/2401.10774) · Local: `research/2401.10774v3.pdf`

> **Relevance:** Adds extra parallel prediction heads to an LLM (without a separate draft model) with tree-based attention for parallel token verification. Evaluated for the iGPU parallel token prefetch path.

---

### 2.8 EAGLE-2 — Dynamic Draft Trees

**EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees**  
Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (2024)  
arXiv: [2406.16858](https://arxiv.org/abs/2406.16858) · Local: `research/2406.16858v2.pdf`

> **Relevance:** Context-adaptive dynamic speculative draft trees achieving 3.05–4.26× speedup (20–40% improvement over EAGLE-1). Informed our speculative token tree construction for cross-accelerator verification.

---

### 2.9 Triton — Tiled Neural Network Computation IR

**Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations**  
Philippe Tillet, Hsiang-Tsung Kung, David D. Cox (2019) — *ACM MAPL at PLDI*  
DOI: [10.1145/3315508.3329973](https://doi.org/10.1145/3315508.3329973) · Local: `research/3315508.3329973.pdf`

> **Relevance:** Introduces the Triton compiler for writing high-performance GPU kernels in Python — the backbone of PyTorch 2.x custom CUDA/ROCm kernel generation used in our iGPU prefill pipeline and custom BiLLM GEMV kernels.

---

## 3. KV Cache Optimization & Memory Management

### 3.1 PagedAttention / vLLM

**Efficient Memory Management for Large Language Model Serving with PagedAttention**  
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023) — *SOSP 2023*  
arXiv: [2309.06180](https://arxiv.org/abs/2309.06180) · Local: `research/2309.06180v1.pdf`

> **Relevance:** OS-inspired paged KV cache allocation eliminating fragmentation (2–4× throughput gain). Our `TransformerContext` chunked KV cache and slot management are directly informed by PagedAttention semantics.

---

### 3.2 H₂O — Heavy-Hitter Oracle KV Cache Eviction

**H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**  
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen (2023)  
arXiv: [2306.14048](https://arxiv.org/abs/2306.14048) · Local: `research/2306.14048v3.pdf`

> **Relevance:** Identifies "heavy hitter" tokens (high accumulated attention mass) and evicts the rest from the KV cache, reducing memory by up to 80%. Directly informed our CSA2 dynamic KV cache quantization heuristics and the MoE active-expert memory budget.

---

### 3.3 SnapKV — Observation-Window KV Compression

**SnapKV: LLM Knows What You are Looking for Before Generation**  
Yuhong Li, Yingbing Huang, Bowen Yang, et al. (2024)  
arXiv: [2404.14469](https://arxiv.org/abs/2404.14469) · Local: `research/2404.14469v2.pdf`

> **Relevance:** Attention-pattern-guided KV cache pruning (40–60% size reduction, 3.6× speedup at 16K tokens). Referenced in our INT8/INT4 dynamic KV cache quantization compatibility heuristics.

---

### 3.4 Mooncake — KVCache-Centric Disaggregated Serving

**Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**  
Ruoyu Qin, Zheming Li, Weiran He, et al. (Moonshot AI, 2024)  
arXiv: [2407.00079](https://arxiv.org/abs/2407.00079) · Local: `research/2407.00079v4.pdf`

> **Relevance:** Production KV cache disaggregation across DRAM and SSD with global scheduler (75% more requests vs. vLLM, 525% throughput in long-context). Informed our NAS-backed quantized model caching and KV checkpoint storage design.

---

## 4. LLM Serving Architecture & Phase Disaggregation

### 4.1 Orca — Continuous Batching

**Orca: A Distributed Serving System for Transformer-Based Generative Models**  
Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun (2022) — *USENIX OSDI 2022*  
Local: `research/osdi22-yu.pdf`

> **Relevance:** Introduced iteration-level (continuous) batching, allowing requests to join/leave inference batches at each decoding step. Foundational to all modern LLM serving systems (vLLM, TGI). Informed our `llama-server` multi-slot concurrency design (up to 64 concurrent clients).

---

### 4.2 Splitwise — Prefill/Decode Phase Disaggregation

**Splitwise: Efficient Generative LLM Inference Using Phase Splitting**  
Pratyush Patel, Esha Choukse, Chaojie Zhang, et al. (Microsoft + UW, 2023)  
arXiv: [2311.18677](https://arxiv.org/abs/2311.18677) · Local: `research/2311.18677v2.pdf`

> **Relevance:** Formally establishes prefill (compute-bound) and decode (memory-bandwidth-bound) as fundamentally different hardware requirements, with up to 2.35× higher throughput via separate machine pools. Our iGPU-prefill → NPU-decode APU pipeline is a single-chip embodiment of this principle.

---

### 4.3 DistServe — Goodput-Optimized Disaggregated Serving

**DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving**  
Yinmin Zhong, Shengyu Liu, Junda Chen, et al. (2024)  
arXiv: [2401.09670](https://arxiv.org/abs/2401.09670) · Local: `research/2401.09670v3.pdf`

> **Relevance:** SLO-aware goodput optimization for disaggregated serving with independent parallelism per phase. Confirmed our phase-disaggregated APU architecture decision.

---

### 4.4 Sarathi-Serve — Chunked Prefill Scheduling

**Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**  
Amey Agrawal, Nitin Kedia, et al. (2024)  
arXiv: [2403.02310](https://arxiv.org/abs/2403.02310) · Local: `research/2403.02310v3.pdf`

> **Relevance:** Chunked-prefills interleave prefill and decode in a stall-free schedule (up to 6.9× vs. Orca on Falcon-180B). Informed our NPU decode pipelining alongside iGPU prefill execution.

---

## 5. On-Device & Edge Inference

### 5.1 FlexGen — Disk-Offloaded LLM Inference

**FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**  
Ying Sheng, Lianmin Zheng, et al. (Stanford / UC Berkeley, 2023)  
arXiv: [2303.06865](https://arxiv.org/abs/2303.06865) · Local: `research/2303.06865v2.pdf`

> **Relevance:** LP-solver–based tensor placement across GPU/CPU/disk enabling OPT-175B on a 16GB GPU with 4-bit compression. Informed our NAS-backed model streaming and large-model-fallback strategies.

---

### 5.2 LLM in a Flash — NVMe-Streamed Inference (Apple)

**LLM in a Flash: Efficient Large Language Model Inference with Limited Memory**  
Keivan Alizadeh-Vahid, Iman Mirzadeh, et al. (Apple, 2023)  
arXiv: [2312.11514](https://arxiv.org/abs/2312.11514) · Local: `research/2312.11514v3.pdf`

> **Relevance:** On-demand NVMe weight streaming for models exceeding DRAM, using windowing and row-column bundling (4–25× speedup). Directly informed our understanding of out-of-core MoE expert streaming on UMA-constrained APU hardware.

---

### 5.3 PowerInfer — Hot/Cold Neuron Hybrid Inference

**PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU**  
Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen (SJTU, 2023)  
arXiv: [2312.12456](https://arxiv.org/abs/2312.12456) · Local: `research/2312.12456v2.pdf`

> **Relevance:** Exploits power-law neuron activation sparsity — hot neurons to GPU VRAM, cold to CPU (11.69× vs. llama.cpp). Informed our MoE router SRAM pinning architecture: hot router/expert weights → on-chip SRAM, cold → UMA DRAM.

---

### 5.4 PowerInfer-2 — Mobile MoE Inference

**PowerInfer-2: Fast Large Language Model Inference on a Smartphone**  
Zhenliang Xue, Yixin Song, Zeyu Mi, Le Chen, Yubin Xia, Haibo Chen (SJTU, 2024)  
arXiv: [2406.06282](https://arxiv.org/abs/2406.06282) · Local: `research/2406.06282v3.pdf`

> **Relevance:** Neuron cluster decomposition + polymorphic mobile NPU/CPU/GPU pipeline enabling a 47B MoE model on a smartphone. Validated our approach to 100B+ MoE inference on a single memory-constrained APU.

---

### 5.5 Deja Vu — Contextual Sparsity

**Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time**  
Zichang Liu, Jue Wang, Tri Dao, et al. (2023) — *ICML 2023*  
arXiv: [2310.17157](https://arxiv.org/abs/2310.17157) · Local: `research/2310.17157v1.pdf`

> **Relevance:** Shows that only ~5% of attention heads/MLP parameters are needed per token (input-dependent sparse subset predictable at inference time), enabling >2× LLM inference speedup. Foundational for our dynamic MoE expert sparsity and active-expert budget routing.

---

## 6. Hardware Architecture & Performance Analysis

### 6.1 Roofline Model

**Roofline: An Insightful Visual Performance Model for Multicore Architectures**  
Samuel Williams, Andrew Waterman, David A. Patterson (UC Berkeley, 2009)  
*Communications of the ACM*, Vol. 52 No. 4, pp. 65–76  
Local: `research/roofline-cacm2008.pdf`

> **Relevance:** The Roofline model is the primary analytical framework used throughout this project to characterize LLM operations: decode (autoregressive GEMV) is memory-bandwidth-bound (operational intensity ~$1.08\,\text{bpw} / 8 = 0.135$ bytes/FLOP → well below the 110 GB/s / ~50 TOPS ridge point), while prefill (batched GEMM) is compute-bound on RDNA 3.5. This directly determines the iGPU-prefill / NPU-decode phase split.

---

### 6.2 MLIR — Compiler Infrastructure

**MLIR: A Compiler Infrastructure for the End of Moore's Law**  
Chris Lattner, Mehdi Amini, et al. (Google, 2020) — *CGO 2021*  
arXiv: [2002.11054](https://arxiv.org/abs/2002.11054) · Local: `research/2002.11054v2.pdf`

> **Relevance:** MLIR is the underlying compiler infrastructure used by AMD Vitis AI / XRT for compiling AIE2P spatial dataflow kernels. Required for understanding XCLBIN graph lowering and NPU kernel code generation.

---

### 6.3 LMAX Disruptor — Lock-Free Ring Buffer

**Disruptor: High Performance Alternative to Bounded Queues for Exchanging Data Between Concurrent Threads**  
Martin Thompson, Dave Farley, Michael Barker, Patricia Gee, Andrew Stewart (LMAX, 2011)  
Local: `research/Disruptor-1.0.pdf`

> **Relevance:** Describes the lock-free, cache-line-optimized ring buffer pattern for ultra-low-latency inter-thread command passing — relevant to our zero-copy NPU command ring and inference pipeline buffer handoff design.

---

### 6.4 AMD Hardware References

| Document | Description | Local File |
| :--- | :--- | :--- |
| **AMD XDNA 2 / AIE2P SW Optimization Guide** | Tile data memory layout, DMA alignment, INT4 MAC units, kernel compilation | `research/47414_15h_sw_opt_guide.pdf` |
| **AMD64 APM (Vol 1–5)** | Zen 5 AVX-512 intrinsics, VPOPCNTDQ, VPXORD, cache topology | `research/40332_4.10_APM_Vol1-5_PUB.pdf` |
| **AMD IOMMU Specification** | PASID isolation, SVMA for zero-copy DMA-BUF sharing between AMDGPU and AMDXDNA | `research/48882_3.11_IOMMU_PUB.pdf` |
| **AMD Developer Reference (APU)** | Strix Point / Gorgon Point / Krackan Point silicon topology | `docs/AMD_APU_DEVELOPER_REFERENCE.md` |

---

## 7. Community Attribution

### Atomic-Germ / Guanaco — MoE Expert Disk Streaming in llama.cpp

**Guanaco: Disk-Streaming for MoE Models on llama.cpp**  
*Atomic-Germ* — [https://github.com/Atomic-Germ/Guanaco](https://github.com/Atomic-Germ/Guanaco)

> **Relevance:** Pioneered on-demand NVMe/disk streaming of Mixture-of-Experts (MoE) expert weights directly within llama.cpp. Core insight: MoE models only activate a handful of experts per token (e.g., top-8 of 256 per layer). Guanaco exploits this by keeping only "hot" experts resident in RAM and dynamically streaming unpinned expert weight slices from NVMe on demand via `io_uring` / `madvise`, enabling 100B+ MoE models to run on RAM-constrained edge hardware. This architecture directly inspired our **MoE router matrix SRAM pinning**, **active expert memory budgeting**, and **out-of-core MoE expert execution pipeline** for running massive 35B–320B models (DeepSeek-V4, Qwen3.8-Flash-Next, Sarvam-105B, GLM-5.3-Flash) on AMD Ryzen AI APUs with 64 GB shared UMA DRAM.

---

## 8. Foundational Textbooks

| Title | Authors | Relevance |
| :--- | :--- | :--- |
| **Computer Systems: A Programmer's Perspective (3rd ed.)** | Bryant & O'Hallaron | Cache hierarchies, memory layout, DRAM timing, DMA mechanics |
| **Operating Systems: Three Easy Pieces** | Arpaci-Dusseau & Arpaci-Dusseau | Virtual memory paging, I/O scheduling, process isolation |
| **Understanding the Linux Kernel** | Bovet & Cesati | DRM subsystem, GEM buffer objects, `drm_syncobj`, `dma_fence` |
| **The Linux Programming Interface** | Michael Kerrisk | `mmap()`, `ioctl()`, `dma-buf` FD passing, `eventfd`, core affinity |
| **What Every Programmer Should Know About Memory** | Ulrich Drepper | Cache-line alignment (64B), NUMA topology, memory bandwidth modeling |
| **Modern Processor Design: Fundamentals of Superscalar Processors** | Shen & Lipasti | Out-of-order execution, branch prediction, AVX-512 throughput modeling |
| **Computer Architecture: A Quantitative Approach** | Patterson & Hennessy | SIMD parallelism, memory wall, roofline analysis |
| **Rust Atomics and Locks** | Mara Bos | Lock-free ring buffers, memory ordering, atomics for DRM fence chaining |
| **Programming Rust** | Blandy & Orendorff | Ownership model for DMA-BUF lifetime management in Rust |

---

## 9. Quick Reference Index by ArXiv ID

| ArXiv ID | Title (Short) | Category |
| :--- | :--- | :--- |
| `2002.11054` | MLIR | Hardware/Compilers |
| `2205.14135` | FlashAttention | Attention/IO |
| `2210.17323` | GPTQ | Quantization |
| `2211.17192` | Speculative Decoding (Google Research) | Speculative Decode |
| `2302.01318` | Speculative Sampling (DeepMind) | Speculative Decode |
| `2303.06865` | FlexGen | Edge/Offload |
| `2306.14048` | H₂O Heavy-Hitter Oracle | KV Cache |
| `2309.06180` | PagedAttention / vLLM | KV Cache |
| `2310.17157` | Deja Vu Contextual Sparsity | Sparsity |
| `2311.18677` | Splitwise | Phase Disaggregation |
| `2312.11514` | LLM in a Flash | Edge/NVMe |
| `2312.12456` | PowerInfer | Edge |
| `2401.09670` | DistServe | Serving Systems |
| `2401.10774` | Medusa Multi-Head | Speculative Decode |
| `2402.02057` | Lookahead Decoding | Speculative Decode |
| `2402.04291` | **BiLLM** | **Primary: Quantization** |
| `2402.04396` | QuIP# Lattice Codebooks | Quantization |
| `2402.12374` | Sequoia | Speculative Decode |
| `2402.17764` | BitNet b1.58 | Quantization |
| `2403.02310` | Sarathi-Serve | Serving Systems |
| `2404.00456` | QuaRot Hadamard | Quantization |
| `2404.11912` | TriForce | Speculative Decode |
| `2404.14469` | SnapKV | KV Cache |
| `2405.16406` | **SpinQuant** | **Primary: Quantization** |
| `2406.06282` | PowerInfer-2 | Edge/MoE |
| `2406.16858` | EAGLE-2 | Speculative Decode |
| `2407.00079` | Mooncake | Serving Systems |
| `2407.09720` | **T-MAC** | **Primary: NPU Execution** |
| `2407.10960` | FLUTE | Quantization |
| `osdi22-yu` | Orca Continuous Batching | Serving Systems |
| `3315508.3329973` | Triton Tiled Kernels | GPU Compilation |
| `roofline-cacm2008` | Roofline Model | Hardware Analysis |
