# AMD Ryzen AI APU Quantization Support Matrix & Guidelines

This document defines the supported quantization spectrum, conversion pipelines, and architectural constraints for running quantized models on **AMD Ryzen AI APUs** (AMD XDNA 2 AIE2P silicon).

---

## 1. Supported Quantization Spectrum (1-Bit BiLLM & Q4 through Q16)

The runtime supports ingesting GGUF models across the **1-bit (BiLLM) and Q4 through Q16** quantization spectrum. The target hardware execution format (`.q4nx`) packages weights in a 4-bit tile-interleaved memory layout tailored for the 32 AIE2P spatial tiles on AMD XDNA 2 silicon, while 1-bit models utilize **T-MAC SRAM lookup tables** directly inside the 32 KB tile data memory.

| Quantization Format | Category | Ingestion Support | Conversion to `.q4nx` | Hardware Acceleration Path | Recommendation Level |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`BiLLM` / `Q1_BILLM`** | 1-bit Residual + Salient | **YES** | SpinQuant Rotation + Saliency Pack | XDNA 2 NPU (T-MAC SRAM LUT) | **Highly Recommended** (1.08 bpw, coherent perplexity via salient protection) |
| **`Q1_0` / `Q1_0_G128`** | 1-bit Block Sign | **YES** | T-MAC Bit Expansion | XDNA 2 NPU (T-MAC SRAM LUT) / CPU | Supported (Requires SpinQuant rotation to prevent collapse) |
| **`IQ4_NL`** | 4-bit Non-Linear | **YES** | Native Codebook Remap | XDNA 2 NPU (AIE2P) | **Highly Recommended** (Best perplexity at 4-bit) |
| **`Q4_K_M` / `Q4_K_S`**| 4-bit K-Quant | **YES** | Direct Tile Packing | XDNA 2 NPU (AIE2P) | **Highly Recommended** (Standard upstream baseline) |
| **`Q4_0`** | 4-bit Linear | **YES** | Zero-Conversion Fastpath | XDNA 2 NPU (AIE2P) | **Recommended** (Fastest conversion, lowest RAM) |
| **`Q4_1`** | 4-bit Linear + Offset | **YES** | Direct Tile Packing | XDNA 2 NPU (AIE2P) | Supported |
| **`IQ4_XS`** | 4-bit Extra Small | **YES** | Codebook Remap | XDNA 2 NPU (AIE2P) | Recommended |
| **`Q5_K_M` / `Q5_K_S`**| 5-bit K-Quant | **YES** | Re-quantized / High-P Scale | XDNA 2 NPU / iGPU | Supported (Higher precision) |
| **`Q5_0` / `Q5_1`** | 5-bit Linear | **YES** | Re-quantized / High-P Scale | XDNA 2 NPU / iGPU | Supported |
| **`Q6_K`** | 6-bit K-Quant | **YES** | High-P Tile Mapping | RDNA 3.5 iGPU / CPU AVX-512 | Supported (High fidelity) |
| **`Q8_0`** | 8-bit Linear | **YES** | Direct Int8 / Tile Repack | RDNA 3.5 iGPU / CPU AVX-512 | **Recommended** (Reference quality baseline) |
| **`F16` / `BF16`** | 16-bit Float | **YES** | Direct FP16 / BF16 | RDNA 3.5 iGPU (WMMA) / Zen 5 | Supported (Lossless unquantized) |
| **`F32`** | 32-bit Float | **YES** | Converted to FP16 / BF16 | RDNA 3.5 iGPU / Zen 5 AVX-512 | Supported (Training / Master weights) |

---

## 2. Non-Recommended & Explicitly Rejected Naive Quantizations (< Q4)

Naive sub-4-bit quantizations (unrotated 1-bit, 2-bit, and 3-bit formats without outlier or salient protection) are **explicitly rejected** because naive binarization collapses perplexity:

| Quantization Format | Category | Status | Technical Reason for Rejection |
| :--- | :--- | :---: | :--- |
| **`IQ1_S` / `IQ1_M`** (naive) | 1-bit (naive) | **REJECTED** | Catastrophic perplexity collapse; output text degrades into repetitive loops. Use **`BiLLM`** instead. |
| **`IQ2_XXS` / `IQ2_XS` / `IQ2_S`** | 2-bit | **REJECTED** | Severe accuracy degradation; 2-bit packing requires irregular bit-slicing that wastes AIE2P memory bandwidth. |
| **`Q2_K`** | 2-bit | **REJECTED** | Severe quality collapse; non-standard block sizes break 64-byte DMA-BUF alignment invariants. |
| **`IQ3_XXS` / `IQ3_S`** | 3-bit | **REJECTED** | 3-bit non-power-of-two packing cannot align to 64-byte AIE2P memory transactions without heavy software bit-shifting that degrades performance below 4-bit throughput. |
| **`Q3_K_S` / `Q3_K_M` / `Q3_K_L`**| 3-bit | **REJECTED** | Unaligned stride stalls tile DMAs; perplexity is strictly inferior to `IQ4_NL` with negligible memory savings. |

### User Error Messaging
When an unsupported sub-4-bit model is provided, `apu-model` and the runtime fail fast with a clear, helpful message:

```text
[UnsupportedQuantizationError]
Model '/path/to/model-iq2_xxs.gguf' uses quantization format 'IQ2_XXS' (2-bit).
Sub-4-bit quantizations (IQ1, IQ2, Q3) are not supported on AMD Ryzen AI APUs
due to severe perplexity loss and non-aligned AIE2P hardware memory strides.

Supported Quantization Range: Q4 through Q16
Recommended Formats:
  - High Efficiency: IQ4_NL (Non-Linear) or Q4_K_M
  - High Precision : Q6_K, Q8_0, or F16
```

---

## 3. Deep-Dive: Why `IQ4_NL` is the Ideal 4-Bit Format

While standard `Q4_0` divides the weight range into 16 linear intervals of equal width:
$$\text{weight} = \text{scale} \times (\text{quant\_index} - 8)$$
Real neural network weights follow a bell-curve (Gaussian/Laplacian) distribution where 95% of weights cluster near zero, and only 5% reside in the tails.

**`IQ4_NL` (Importance-matrix Non-Linear)** uses a 16-element non-linear lookup grid optimized via an importance matrix (`imatrix`):
```text
kvalues_iq4nl = {
  -3.56, -2.57, -1.90, -1.38, -0.95, -0.57, -0.22, 0.12,
   0.46,  0.82,  1.22,  1.69,  2.28,  3.09,  4.15, 5.82
}
```

### Ingestion into `.q4nx`
1. `apu-model convert` reads the 16 non-linear codebook values and maps each tensor block into high-precision intermediate representations.
2. The weights are then packed into the **AMD XDNA 2 AIE2P 32-tile spatial interleave** format with 64-byte alignment.
3. The resulting `.q4nx` file executes at **100% native hardware speed** on the 32 AIE2P tiles, while retaining the high perplexity and output coherence of the non-linear quantization.

---

## 4. Deep-Dive: BiLLM (1.08 bpw) with SpinQuant & T-MAC NPU Execution

Standard naive 1-bit quantization (such as `IQ1_S`) collapses model perplexity because outlier activations distort the binarization threshold, causing loss of critical attention patterns.

### The 4-Stage Rotation & Saliency Pipeline
`llama-apu` addresses this via **BiLLM** combined with **SpinQuant**:
1. **Stage 1 (SpinQuant Rotation)**: Applies learned orthogonal rotation matrices ($W' = Q W R^\top$) to rotate weight and activation channels. This diffuses cross-channel kurtosis and suppresses outliers without requiring runtime de-quantization.
   > [!IMPORTANT]
   > **Strict Precedence Invariant**: SpinQuant rotation **strictly precedes** Hessian computation and saliency selection. Rotating after selecting salient weights destroys the coordinate isolation needed for binary residual encoding.
2. **Stage 2 (Hessian Saliency Identification)**: Computes the empirical Hessian $\tilde{H} = 2 \tilde{X}\tilde{X}^\top$ over rotated calibration activations to identify the top 0.5% - 1.0% most sensitive weights.
3. **Stage 3 (Salient Weight Protection)**: Isolates these critical weights into higher-precision storage (INT4 or FP16), shielding them from binarization noise.
4. **Stage 4 (Binary Residual Encoding)**: Decomposes the remaining 99% weights into a 2-stage binary residual:
   $$W_{\text{bin}} = \alpha_1 \text{sign}(W) + \alpha_2 \text{sign}(W - \alpha_1 \text{sign}(W))$$
   yielding an effective density of **1.08 bits per weight (bpw)**.

### T-MAC SRAM Lookup Tables on XDNA 2 AIE2P
Because XDNA 2 AIE2P lacks native 1-bit MACs, naive unpacking to INT4 incurs heavy memory bandwidth and register inflation penalties.
`llama-apu` executes 1-bit models using **T-MAC lookup tables (LUTs)**:
- Pre-computes activation sums for small bit groups ($k=4$ or $k=8$) directly in the **32 KB tile data SRAM**.
- During autoregressive decode, the NPU performs **multiplication-free table lookups** directly using weight bit patterns as indices.
- Achieves full memory-bandwidth saturation (~110–136 GB/s UMA) at 5–10W active NPU power.

For comprehensive architectural comparisons with other 1-bit / 2-bit formats (BitNet b1.58, TQ1_0, T-ACE, FLUTE, NanoQuant), see [docs/quantization_alternatives.md](docs/quantization_alternatives.md).

---

## 5. Summary Guidance for External Users

| Goal | Recommended Ingestion Format | Resulting Container | Expected Performance |
| :--- | :--- | :--- | :--- |
| **Lowest RAM Footprint (1.08 bpw)** | `BiLLM` (with SpinQuant) | `.q4nx` / `.gguf` (1-bit) | 60–260 tok/s on MoE, 3.6–4.2 GB for 27B–31B |
| **Best Balance of Quality & Speed** | `IQ4_NL` or `Q4_K_M` | `.q4nx` (4-bit) | 25–60 tok/s on NPU, minimal perplexity loss |
| **Fastest Turnkey Conversion** | `Q4_0` | `.q4nx` (4-bit) | Instant conversion, baseline 4-bit memory |
| **Maximum Language Fidelity** | `Q8_0` or `F16` | `.q4nx` (or direct GGUF) | Reference PyTorch accuracy |
| **Battery / Low Power Mode** | `BiLLM` or `IQ4_NL` with `--npu-based` | `.q4nx` (1-bit / 4-bit) | Lowest package power (~5–15W TDP) |
