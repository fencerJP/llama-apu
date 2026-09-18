# AMD Ryzen AI APU Quantization Support Matrix & Guidelines

This document defines the supported quantization spectrum, conversion pipelines, and architectural constraints for running quantized models on **AMD Ryzen AI APUs** (AMD XDNA 2 AIE2P silicon).

---

## 1. Supported Quantization Spectrum (Q4 through Q16)

The runtime supports ingesting GGUF models across the **Q4 through Q16** quantization spectrum. The target hardware execution format (`.q4nx`) packages weights in a 4-bit tile-interleaved memory layout tailored for the 32 AIE2P spatial tiles on AMD XDNA 2 silicon.

| Quantization Format | Category | Ingestion Support | Conversion to `.q4nx` | Hardware Acceleration Path | Recommendation Level |
| :--- | :--- | :---: | :---: | :--- | :--- |
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

## 2. Non-Recommended & Explicitly Rejected Quantizations (< Q4)

Sub-4-bit quantizations (1-bit, 2-bit, and 3-bit formats) are **explicitly not recommended and rejected** by the ingestion pipeline.

| Quantization Format | Category | Status | Technical Reason for Rejection |
| :--- | :--- | :---: | :--- |
| **`IQ1_S` / `IQ1_M`** | 1-bit | **REJECTED** | Catastrophic perplexity loss; output text degrades into repetitive loops or incoherent tokens. |
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

## 4. Summary Guidance for External Users

| Goal | Recommended Ingestion Format | Resulting Container | Expected Performance |
| :--- | :--- | :--- | :--- |
| **Best Balance of Quality & Speed** | `IQ4_NL` or `Q4_K_M` | `.q4nx` (4-bit) | 25–60 tok/s on NPU, minimal perplexity loss |
| **Fastest Turnkey Conversion** | `Q4_0` | `.q4nx` (4-bit) | Instant conversion, baseline 4-bit memory |
| **Maximum Language Fidelity** | `Q8_0` or `F16` | `.q4nx` (or direct GGUF) | Reference PyTorch accuracy |
| **Battery / Low Power Mode** | `IQ4_NL` with `--npu-based` | `.q4nx` (4-bit) | Lowest package power (~15W TDP) |
