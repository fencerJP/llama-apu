<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors
-->

# Quantization Alternatives, Binarization Mitigations & APU Silicon Trade-offs

This document serves as the comprehensive technical reference for extreme low-bit quantization formats, mathematical mitigations for binarization collapse, hardware execution trade-offs on the AMD Ryzen AI APU (Zen 5 CPU, RDNA 3.5 iGPU, XDNA 2 NPU), and evaluations of large-scale open-weight models for edge deployment.

---

## 1. Primary Architecture: SpinQuant + BiLLM (1.08 bpw)

### 1.1 The Mathematical Synergy
The primary quantization architecture selected is **BiLLM (Post-Training Binarization with Salient Weight Protection)** combined with **SpinQuant (Offline Learned Orthogonal Rotation)**.

In unrotated models, activation and weight outliers are heavily concentrated in a small subset of channels. In standard BiLLM, non-salient weights still span a broad magnitude range, which degrades accuracy during optimal splitting binarization.

By applying SpinQuant rotations prior to BiLLM:
1. **Computational Invariance:** Orthogonal rotation matrices $R \in \mathbb{R}^{d_{in} \times d_{in}}$ and $Q \in \mathbb{R}^{d_{out} \times d_{out}}$ ($R R^\top = I, Q Q^\top = I$) are optimized offline via gradient descent on calibration data. Rotated weights $\tilde{W} = Q W R^\top$ and rotated activations $\tilde{X} = R X$ maintain exact full-precision outputs:
   $$\tilde{Y} = \tilde{X} \tilde{W}^\top = (R X) (Q W R^\top)^\top = R X R W^\top Q^\top = X W^\top Q^\top$$
   (with $Q^\top$ and $R$ absorbed offline into adjacent projection layers or RMSNorm scales).
2. **Outlier Flattening:** Rotations transform the weight distribution into a smooth, sub-Gaussian bell curve, eliminating heavy-tailed outlier spikes.
3. **Well-Conditioned Saliency:** Computing the Hessian in rotated space ($\tilde{H} = 2 \tilde{X} \tilde{X}^\top$) provides a stable, well-conditioned sensitivity metric.
4. **Minimal Residual Loss:** When BiLLM isolates salient entries in $\tilde{W}$, the non-salient entries have a tightly bounded, uniform distribution, enabling optimal splitting search to achieve near-zero residual binarization loss.

### 1.2 The Strict 4-Stage Execution Workflow
```
[Hugging Face FP16/BF16 Model]
               │
               ▼
[Phase 1: SpinQuant Offline Rotation]
  • Learn orthogonal matrices R, Q via Cayley gradient descent
  • Compute rotated weights: W_tilde = Q * W * R^T
  • Fuse R, Q^T into preceding RMSNorm / down-proj layers
               │
               ▼
[Phase 2: Rotated Hessian Calibration]
  • Pass calibration tokens through rotated network
  • Compute rotated activation Hessian: H_tilde = 2 * X_tilde * X_tilde^T
               │
               ▼
[Phase 3: BiLLM Salient Weight Identification]
  • Evaluate salient coordinates via H_tilde and |W_tilde| (~1-2% of weights)
  • Structurally segment salient weights
               │
               ▼
[Phase 4: Residual Binarization & Optimal Splitting]
  • Salient weights: W_salient ≈ α_1 * B_1 + α_2 * B_2 (2-bit binary residual)
  • Non-salient weights: W_non_salient ≈ s_g * (2 * B - 1) (1-bit optimal split)
  • Pack into 1.08 bpw container
```

> [!CRITICAL]
> **Pipeline Invariant (Strict Precedence):**  
> **Offline rotation MUST strictly precede Hessian computation and salient weight selection in the codebase.**  
> Rotating a matrix forms linear combinations of entries across hidden dimensions. If salient weights were selected before rotation, applying $R$ afterwards would mix salient and non-salient entries together, destroying the structural isolation required for binary residual encoding.

---

## 2. Quantization Alternatives Reference Matrix

| Format / Strategy | Effective Bitrate | Block Size & Layout | Unpacking & Arithmetic Mechanism | Accuracy & Perplexity Impact | Primary Use Case |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`BiLLM`** *(with SpinQuant)* | **1.08 bpw** | Salient weights in 2-bit binary residual ($s_1 B_1 + s_2 B_2$); non-salient weights in 1-bit. | Binary residual lookup + structural grouping. | **SOTA 1-Bit PTQ**: Achieves **8.41 perplexity** on LLaMA-2-70B without full retraining. | **Primary runtime target for 27B–320B models.** |
| **`Q1_0` / `Q1_0_g128`** *(llama.cpp / Bonsai)* | **1.125 bpw** | Group size $G=128$. 1 sign bit ($b_i \in \{0, 1\}$) + 1 FP16 scale ($s_g$). | Pure 1-bit binarization. Inline bit-shift and sign selection. | High when distilled (Bonsai 8B reaches 70.5 benchmark avg); naive PTQ collapses. | Simple baseline; native llama.cpp compatibility. |
| **`BitNet b1.58`** *(Native Ternary)* | **1.58 bpw** ($\log_2 3$) | Native ternary weights $\{-1, 0, 1\}$ trained from scratch + per-tensor/channel scale. | Multiplication-free additions/subtractions ($X \cdot W$). | Matches full-precision Transformer baseline across all scales. | From-scratch pre-trained models. |
| **`TQ1_0`** *(llama.cpp Ternary)* | **1.69 bpw** | Block size 256. 5 trits packed per byte ($3^5 = 243 \le 256$) + 1 FP16 scale per 256 weights. | 5-trit byte packing. Requires multi-stage or LUT-based 5-trit unpacking at runtime. | Smallest file size among ternary formats, but higher unpacking latency reduces decode speed. | Maximum disk/RAM compression for ternary. |
| **`TQ2_0` & `I2_S`** *(llama.cpp / BitNet.cpp)* | **2.00 – 2.06 bpw** | Block size 256. Aligned 2-bit (INT2) layout per ternary weight + 1 FP16 scale. | Direct 2-bit bitmasking/unpacking; avoids non-power-of-two trit decoding. | **Optimal Edge Throughput**: Trades slightly larger file size for significantly higher prefill/decode tok/s. | High-speed edge decode when 2.0 bpw fits in RAM. |
| **`T-ACE` Co-Packed 16B** | **2.00 bpw** | 16-byte co-packed block containing 64 ternary weights + 2-level power-of-two scale metadata. | Two-stage 5-trit hardware unpacker (60 packed weights + 4 unencoded) + shift-based scaling. | **Zero Scale-Fetch Penalty**: Eliminates separate DRAM scale requests, maximizing bus bandwidth. | Specialized accelerator pipelines. |
| **`FLUTE`** *(Non-Uniform LUT)* | **3.13 – 4.13 bpw** | Group size 64/128. Stores non-uniform quantization indices mapped via on-chip shared-memory vectorized LUTs. | Vectorized lookup table (2 elements per lookup) in shared memory + Stream-K work decomposition. | **High Quality on Hard Models**: Mitigates accuracy collapse on difficult-to-quantize models like LLaMA-3. | 3-bit and 4-bit non-uniform weights. |
| **`NanoQuant`** *(Sub-1-Bit PTQ)* | **< 1.0 bpw** | Low-rank binary factorization ($W \approx A \cdot B$) solved via ADMM. | Low-rank binary matrix multiplication. | Extreme compression: Enables LLaMA-2-70B ($25.8\times$ compression) to fit within 8 GB VRAM. | Extreme memory-constrained edge devices. |

---

## 3. Alternative Mitigations for Binarization Collapse

### 3.1 Advanced Orthogonal Rotations
- **Randomized Hadamard Transform (`QuaRot` / `QuIP#`):** Applies randomized orthogonal Hadamard matrices $H$ to hidden states and weights. Absorbed offline into adjacent layers at zero runtime FLOP cost.
- **`ReSpinQuant` (Layer-Wise Subspace Residual Rotations):** Overcomes the constraint of a single global rotation matrix across layers by applying localized layer-wise adaptation with residual subspace rotations, fused offline into weights.
- **`KronQ` (Bidirectional Incoherence & Gradient Covariance):** Uses Kronecker-factored Hessian approximations combining activation and gradient covariances. Rotates both input and output dimensions, achieving **7.93 perplexity** on LLaMA-3-70B at 2 bits (where GPTQ diverges > 2000 PPL).

### 3.2 Outlier Preservation & Structuring
- **`OffQ` (Structured Channel Offsetting):** Rotates activation matrices to consolidate scattered outliers into a single designated channel, which is then subtracted as a shared offset.
- **Salient Residual Splitting (`BiLLM`):** Preserves the top 1–2% sensitive parameters in higher-precision or residual binary representations.

### 3.3 Reasoning-Aware Calibration
- **`ScaleQ-1.58` & AYOT (Attend to Your Own Thoughts):** Standard calibration sets (e.g., WikiText-2) ignore multi-step reasoning, causing quantized models to collapse on math and coding. AYOT feeds reasoning traces generated by the full-precision teacher model back as context during calibration, preserving complex reasoning with $1,000,000\times$ fewer calibration tokens than pre-training.

### 3.4 Differentiable Softening
- **`CAT-Q`:** Uses learnable modulation and softened step functions during post-training optimization to smoothly transition continuous weights toward discrete ternary/binary targets.

---

## 4. APU Silicon Execution Architecture & Stage Routing

On AMD Ryzen AI APUs (AMD Ryzen AI 9 HX 470, Strix Point silicon), execution is mapped across the Zen 5 CPU, RDNA 3.5 iGPU, and XDNA 2 AIE2P NPU.

```
Method A: SRAM Lookup Tables (T-MAC / LUT-GEMM) ──► Selected for NPU Decode
[Packed Sub-4-Bit Indices] ──► [SRAM LUT Gather (32KB Tile SRAM)] ──► [Vector Addition] ──► Output
                                 (Multiplication-Free)

Method B: Runtime Unpacking & Vector MAC (T-ACE / BitNet.cpp)
[Packed Sub-4-Bit Weights] ──► [Shift-Mask / Inline Unpacker] ──► [INT4 Registers] ──► [AIE2P INT4 MAC/MMUL]
```

### 4.1 T-MAC on XDNA 2 NPU for q1(BiLLM)
For `q1(BiLLM)` models running on the XDNA 2 NPU, **T-MAC SRAM Lookup Tables** are used:
- **Multiplication-Free:** Precomputes activation dot products into local 32 KB tile SRAM. 1-bit weight indices act as table gathers (`TBL`/`PSHUF`) followed by vector additions.
- **Overcomes Native INT4 Limitation:** Eliminates the need to expand 1-bit weights into INT4 registers, preventing register pressure bloat and pipeline stalls caused by the $4\times$ bandwidth mismatch between 1-bit weights and FP16/INT8 activations.

### 4.2 Full Support for APU Backend Overrides
The custom runtime multiplexer guarantees complete support for all stage routing CLI flags with `q1(BiLLM)` models:
- `--tokenize {cpu,gpu,npu}`: Routes tokenization (default: `cpu` via AVX-512 Classic core).
- `--prefill {gpu,cpu,npu}`: Routes prompt evaluation forward pass (default: `gpu` via RDNA 3.5 batched GEMM).
- `--decode {npu,gpu,cpu}`: Routes autoregressive token generation loop (default: `npu` via T-MAC on AIE2P tiles).
- `--gpu-based`: Macro preset forcing all stages to RDNA 3.5 iGPU (`--tokenize gpu --prefill gpu --decode gpu`).
- `--cpu-based`: Macro preset forcing all stages to Zen 5 CPU (`--tokenize cpu --prefill cpu --decode cpu`).
- `--npu-based`: Macro preset forcing all stages to XDNA 2 NPU (`--tokenize npu --prefill npu --decode npu`).

When overridden to CPU, `q1(BiLLM)` executes via AVX-512 VPOPCNTDQ (`_mm512_popcnt_epi64`) and VPXORD (`_mm512_xor_si512`). When overridden to iGPU, it executes via the SIMD32 bitfield extraction (`bfe`) GEMV kernel.

---

## 5. Storage Invariants & Testing Protocol

> [!IMPORTANT]
> ### Mandatory Testing & Storage Rules
> 1. **In-Place Quantization Only (Zero Full FP16 Disk Footprint):**  
>    When testing, **never download the full FP16 / safetensors model to local disk**. Models must be quantized in-place on-the-fly via streaming chunked ingestion or processed directly from the network stream into `BiLLM` / `.q4nx` format.
> 2. **Single Quantized Model Disk Invariant:**  
>    When running inference tests on quantized models, **always download/hold exactly one quantized model on local disk at a time**. Before downloading or generating the next model, the previous model must either be deleted or moved to the network storage repository.
> 3. **Designated NAS Repository:**  
>    - **SMB Share URI:** `smb://truenas/media/Downloads/model_testing`
>    - **Local Mount Point:** `/mnt/Media/Downloads/model_testing` (verified writable, live).
>    - All quantized weights and evaluation artifacts are stored on this share.

---

## 6. Complete Target Models Evaluation Matrix (with Projected Output tok/s)

The evaluation suite includes 10 leading open-weight models: the two original baseline models (`Gemma-4-31B` and `Qwen3.8-27B Cold-Fusion`) and the eight newly added frontier MoE architectures.

Under **BiLLM 1.08 bpw**, all 10 models fit within the 64 GB UMA memory space of the Ryzen AI 9 HX 470 (59 GiB physical, ~39 GiB available):

| Model Identifier | Architecture | Total Params | Active Params | Context Window | FP16 Size | BiLLM 1-Bit Size (~1.08 bpw) | Fits in 64 GB APU RAM? | Active Weight Traffic / tok | Projected Decode tok/s (110 GB/s UMA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`google/gemma-4-31B`](https://huggingface.co/google/gemma-4-31B)** | Dense | 31.0B | 31.0B | 128K | ~62.0 GB | **~4.18 GB** | **✅ Fits (< 7% RAM)** | **~4.18 GB** | **~26.3 tok/s** |
| **[`DavidAU/Qwen3.8-27B-Cold-Fusion`](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU)** | Dense | 27.2B | 27.2B | 128K (1M YaRN) | ~54.4 GB | **~3.67 GB** | **✅ Fits (< 6% RAM)** | **~3.67 GB** | **~30.0 tok/s** |
| **[`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next)** | MoE (Agentic Coding) | 80B | 3B | 262K | ~160 GB | **~11.2 GB** | **✅ Fits (< 20% RAM)** | **~420 MB** | **~262 tok/s** |
| **[`sarvamai/sarvam-105b`](https://huggingface.co/sarvamai/sarvam-105b)** | MoE (Reasoning / Math) | 105B | 10.3B | 128K | ~210 GB | **~14.7 GB** | **✅ Fits (< 25% RAM)** | **~1.44 GB** | **~76 tok/s** |
| **[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1)** | MoE (256 Experts) | 118B | 8B | 128K | ~236 GB | **~16.5 GB** | **✅ Fits (< 28% RAM)** | **~1.12 GB** | **~98 tok/s** |
| **[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** | MoE (GDN + QSA) | 125B | 6B | 262K (1M YaRN) | ~250 GB | **~17.5 GB** | **✅ Fits (< 30% RAM)** | **~840 MB** | **~131 tok/s** |
| **[`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)** | MoE (CSA) | 284B | 13B | 1M | ~568 GB | **~39.8 GB** | **✅ Fits (< 68% RAM)** | **~1.82 GB** | **~60 tok/s** |
| **[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** | MoE (CED) | 284B | 13B | 1M | ~568 GB | **~39.8 GB** | **✅ Fits (< 68% RAM)** | **~1.82 GB** | **~60 tok/s** |
| **[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)** | MoE (CSA Checkpoint) | 284B | 13B | 1M | ~568 GB | **~39.8 GB** | **✅ Fits (< 68% RAM)** | **~1.82 GB** | **~60 tok/s** |
| **[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)** | MoE (Sparse/Linear + mHC)| 320B | 18B | 128K | ~640 GB | **~45.0 GB** | **✅ Fits (< 76% RAM)** | **~2.52 GB** | **~44 tok/s** |

### 6.1 Decode Throughput Derivation Formula
$$\text{Projected Decode (tok/s)} = \frac{\text{Effective UMA Memory Bandwidth (110 GB/s)}}{\text{Active Parameters per Token} \times \left(\frac{1.08 \text{ bits}}{8 \text{ bits/byte}}\right) + \text{KV Cache Traffic / tok}}$$
*(Assuming batch size = 1, KV cache quantized to Q8_0/Q4_0, and zero host-side `memcpy` via Linux DMA-BUF).*

---

## 7. Acknowledgments & Community Attribution

Special thanks to **[Atomic-Germ / Guanaco](https://github.com/Atomic-Germ/Guanaco)** for pioneering **on-demand NVMe/disk streaming of Mixture-of-Experts (MoE) expert weights in llama.cpp**. Guanaco's core insight — that MoE models only activate a handful of experts per token (e.g., top-8 of 256 per layer), and by keeping only "hot" experts resident in RAM and dynamically streaming unpinned expert weight slices from NVMe on demand via `io_uring` / `madvise`, 100B+ MoE models can run on RAM-constrained edge hardware — directly inspired our **MoE router matrix SRAM pinning**, **active expert memory budgeting**, and **out-of-core MoE expert execution pipeline** that enable running 35B–320B models on AMD Ryzen AI APUs.

---

## 8. Academic References

For the full list of papers that informed this project's quantization algorithms, inference architecture, and hardware design, see **[docs/REFERENCES.md](REFERENCES.md)**.

Key papers referenced in this document:

| Method | Paper | ArXiv |
| :--- | :--- | :--- |
| **BiLLM** | Huang et al., "BiLLM: Pushing the Limit of Post-Training Quantization for LLMs" | [2402.04291](https://arxiv.org/abs/2402.04291) |
| **SpinQuant** | Liu et al., "SpinQuant: LLM Quantization with Learned Rotations" (Meta FAIR) | [2405.16406](https://arxiv.org/abs/2405.16406) |
| **T-MAC** | Wei et al., "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment" (Microsoft) | [2407.09720](https://arxiv.org/abs/2407.09720) |
| **BitNet b1.58** | Ma et al., "The Era of 1-bit LLMs" (Microsoft) | [2402.17764](https://arxiv.org/abs/2402.17764) |
| **QuaRot** | Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" | [2404.00456](https://arxiv.org/abs/2404.00456) |
| **QuIP#** | Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence" (Cornell) | [2402.04396](https://arxiv.org/abs/2402.04396) |
| **GPTQ** | Frantar et al., "GPTQ: Accurate Post-Training Quantization" (ETH Zürich) | [2210.17323](https://arxiv.org/abs/2210.17323) |
| **FLUTE** | Buckley et al., "Fast Matrix Multiplications for Lookup Table-Quantized LLMs" | [2407.10960](https://arxiv.org/abs/2407.10960) |
| **H₂O** | Zhang et al., "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference" | [2306.14048](https://arxiv.org/abs/2306.14048) |
| **Roofline** | Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model" | CACM 2009 |
