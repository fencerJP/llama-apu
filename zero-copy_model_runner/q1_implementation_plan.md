<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Heterogeneous APU Orchestrator Contributors
-->

# Complete Implementation Plan: PrismML 1-Bit (`Q1_0_g128`) Quantization Engine & Zero-Copy APU Runtime

**Target Hardware:** AMD Ryzen AI 9 HX 470 (Radeon 890M RDNA 3.5 iGPU + XDNA 2 AIE2P NPU + Zen 5 AVX-512 CPU)  
**Primary Repositories:**  
- Engine / Backend: [`zero-copy_model_runner`](file:///home/fencer/.openclaw/workspace/projects/zero-copy_model_runner)  
- Frontend / Multiplexer: [`llamacpp-update/llama.cpp`](file:///home/fencer/.openclaw/workspace/projects/llamacpp-update/llama.cpp)  
**Evaluation Targets:**  
1. [`google/gemma-4-31B`](https://huggingface.co/google/gemma-4-31B)  
2. [`DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU`](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU)

---

## 1. Executive Summary & Design Principles

This document specifies the end-to-end mathematical, software, and hardware integration plan to introduce the **PrismML Bonsai 1-bit quantization format (`Q1_0_g128`)** into the custom `llama-apu` zero-copy runtime.

### Core Principles
1. **Coexistence with Existing Quantizations:**  
   `Q1_0_g128` is introduced strictly as an **additional quantization option** alongside existing supported formats (`IQ4_NL`, `Q4_K_M`, `Q4_0`, `Q4_1`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, `BF16`). It **does not replace, disable, or modify** the ingestion, conversion, or execution pathways of any other quantization type.
2. **Sub-4-Bit Strategic Exception:**  
   `QUANTIZATION.md` historically rejected `< Q4` formats (`IQ1_S`, `Q2_K`, `Q3_K`) due to non-power-of-two packing and severe perplexity collapse. `Q1_0_g128` introduces a principled exception:
   - **Outlier Mitigation:** Offline orthogonal pre-rotation (Hadamard / SpinQuant) neutralizes heavy-tailed outliers prior to binarization.
   - **Hardware Alignment:** 18-byte blocks pack in multiples of 4 (72 bytes) and 16 (288 bytes), integrating cleanly into RDNA 3.5 SIMD32 wave loads and Zen 5 AVX-512 cache lines.
3. **Memory Feasibility for 27B–31B Models on 64 GB UMA:**  
   On the local AMD Ryzen AI 9 HX 470 (59 GiB physical RAM, ~39 GiB available):
   - **FP16 / BF16:** Gemma-4-31B is ~62 GB and Qwen3.8-27B is ~54 GB. Neither fits in available RAM.
   - **PrismML `Q1_0_g128` (1.125 bpw):** Gemma-4-31B compresses to **~4.36 GB** and Qwen3.8-27B compresses to **~3.80 GB**, running fully in unified LPDDR5X DRAM with massive headroom for 256K context KV caches.

---

## 2. Mathematical Formulations & Algorithms

### 2.1 Format Definition & Bitrate Derivation
In `Q1_0_g128`, weights are partitioned into contiguous groups of $g = 128$ parameters. Each weight $w_i \in \mathbb{R}$ is represented by a single sign bit $b_i \in \{0, 1\}$, sharing one FP16 scale factor $s_g$.

$$\text{Bitrate} = 1 \text{ bit/weight} + \frac{16 \text{ scale bits}}{128 \text{ weights}} = 1 + 0.125 = 1.125 \text{ bits/weight (bpw)}$$

This provides an idealized **14.22× memory compression** relative to FP16.

### 2.2 Optimal Scale Factor Derivation
To reconstruct $w_i$ from $b_i$, the dequantization function is:

$$\hat{w}_i = s_g \cdot (2b_i - 1) = \begin{cases} +s_g & \text{if } b_i = 1 \\ -s_g & \text{if } b_i = 0 \end{cases}$$

Given a group of FP16 weights $W_g = [w_1, w_2, \dots, w_{128}]^\top$, we choose $s_g$ and $b_i$ to minimize the Mean Squared Error (L2 reconstruction loss):

$$\mathcal{L}(s_g, b) = \sum_{i=1}^{128} (w_i - \hat{w}_i)^2 = \sum_{i=1}^{128} \left(w_i - s_g \cdot (2b_i - 1)\right)^2$$

For any non-negative scale $s_g > 0$, the optimal bit assignment minimizing each term is:

$$b_i = \begin{cases} 1 & \text{if } w_i \ge 0 \\ 0 & \text{if } w_i < 0 \end{cases}$$

Substituting $2b_i - 1 = \text{sign}(w_i)$:

$$\mathcal{L}(s_g) = \sum_{i=1}^{128} \left(w_i - s_g \cdot \text{sign}(w_i)\right)^2 = \sum_{i=1}^{128} \left(|w_i| - s_g\right)^2$$

Taking the partial derivative with respect to $s_g$ and setting it to zero:

$$\frac{\partial \mathcal{L}}{\partial s_g} = -2 \sum_{i=1}^{128} (|w_i| - s_g) = 0 \implies 128 \cdot s_g = \sum_{i=1}^{128} |w_i|$$

$$s_g = \frac{1}{128} \sum_{i=1}^{128} |w_i| = \frac{1}{128} \|W_g\|_1$$

The optimal scale factor $s_g$ is exactly the **L1-mean of the group weights**.

### 2.3 Apple MLX Scale & Bias Re-encoding
Apple MLX requires affine parameters $(s_{\text{mlx}}, b_{\text{mlx}})$ per group:

$$\hat{w}_i = s_{\text{mlx}} \cdot b_i + b_{\text{mlx}}$$

Matching PrismML's scale-only formula:

$$s_g \cdot (2b_i - 1) = (2s_g) \cdot b_i + (-s_g)$$

Therefore:
$$s_{\text{mlx}} = 2s_g, \quad b_{\text{mlx}} = -s_g$$

### 2.4 Pre-Quantization Incoherence & Orthogonal Rotation
Naive binarization fails when heavy-tailed activation outliers concentrate large magnitudes in small channels. To eliminate outliers without adding inference FLOPs, an orthogonal rotation matrix $R \in \mathbb{R}^{d \times d}$ ($R R^\top = I$) is applied:

$$W' = W \cdot R, \quad X' = R^\top \cdot X$$

Because $R$ is orthogonal, the matrix product is invariant:

$$Y = X' W'^\top = (R^\top X) (W R)^\top = X R^\top R W^\top = X W^\top$$

- **Offline Fusion:** $R^\top$ is absorbed offline into the preceding layer's output weights or RMSNorm scale vectors.
- **Inference Cost:** Exactly **0 additional FLOPs**.
- **Rotation Operator:** Randomized Hadamard Transform (RHT) or learned SpinQuant rotation matrices.

---

## 3. Data Structures & Binary Layout

### 3.1 C/C++ GGML Block Struct (`block_q1_0_g128`)

```c
#define QK1_0 128  // Group size: 128 weights per scale

typedef struct {
    ggml_fp16_t d;          // FP16 group scale factor s_g (2 bytes)
    uint8_t qs[QK1_0 / 8];  // 128 packed sign bits (16 bytes)
} block_q1_0_g128;         // Total size: 18 bytes (1.125 bpw)
```

### 3.2 Bit-Packing Convention (LSB-First)
For 128 weights indexed $i \in [0, 127]$:
- Byte $k = \lfloor i / 8 \rfloor$ ($k \in [0, 15]$)
- Bit offset $j = i \pmod 8$ ($j \in [0, 7]$)

$$\text{qs}[k] = \sum_{j=0}^{7} b_{8k + j} \cdot 2^j$$

To unpack bit $i$:
$$b_i = (\text{qs}[i \gg 3] \gg (i \& 7)) \& 1$$

---

## 4. Converter Implementation Specification (`convert_to_prismml.py`)

A standalone, modular Python tool located at `converter/convert_to_prismml.py`.

```
[Hugging Face Safetensors (FP16/BF16)]
                   │
                   ▼
[Stage 1: Incoherence Rotation Pass (RHT / SpinQuant)]
                   │  (W' = W * R, R fused into preceding RMSNorm/Projections)
                   ▼
[Stage 2: Layer & Tensor Partitioning (128-weight chunks)]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[s_g = mean(|w_i|)]     [b_i = (w_i >= 0) ? 1 : 0]
         │                   │
         │          [LSB-first Bit Packing into 16B]
         └─────────┬─────────┘
                   ▼
[Stage 3: Assemble block_q1_0_g128 (18 bytes)]
                   │
         ┌─────────┴───────────────────────┐
         ▼                                 ▼
[GGUF Serializer (gguf-py)]       [MLX Serializer]
(GGML_TYPE_Q1_0_G128)             (s_mlx = 2*s_g, b_mlx = -s_g)
```

### Python Conversion Algorithm (Core Inner Loop)

```python
import numpy as np
import torch
import struct

def quantize_row_q1_0_g128(weights_fp16: np.ndarray) -> bytes:
    """
    Quantizes a 1D float16 numpy array whose length is a multiple of 128.
    Returns packed byte payload: 18 bytes per 128 weights.
    """
    assert weights_fp16.size % 128 == 0
    n_blocks = weights_fp16.size // 128
    w = weights_fp16.reshape(n_blocks, 128)

    # 1. Compute L1 mean scale factor per group
    scales = np.mean(np.abs(w), axis=1).astype(np.float16)

    # 2. Extract sign bits: 1 if w >= 0 else 0
    bits = (w >= 0).astype(np.uint8)

    # 3. Pack 128 bits into 16 bytes LSB-first
    bits_reshaped = bits.reshape(n_blocks, 16, 8)
    multipliers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    packed_bytes = np.sum(bits_reshaped * multipliers, axis=2, dtype=np.uint8)

    # 4. Interleave FP16 scale (2 bytes) + packed sign bits (16 bytes) = 18 bytes
    out_buffer = bytearray()
    for block_idx in range(n_blocks):
        scale_bytes = struct.pack("<e", scales[block_idx]) # FP16 little-endian
        out_buffer.extend(scale_bytes)
        out_buffer.extend(packed_bytes[block_idx].tobytes())

    return bytes(out_buffer)
```

---

## 5. Modifications to `zero-copy_model_runner` (APU Engine)

### 5.1 Update Ingestion & Quantization Matrix (`QUANTIZATION.md` & `src/container/converter.rs`)
- **Preserve Existing Formats:** Keep `IQ4_NL`, `Q4_K_M`, `Q4_0`, `Q4_1`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, `BF16` fully active.
- **Add `Q1_0_g128` Ingestion:** Remove the blanket rejection of `< Q4` formats specifically for `Q1_0_g128`.

```rust
// src/container/converter.rs
pub fn is_quantization_supported(quant_name: &str) -> bool {
    match quant_name.to_uppercase().as_str() {
        // Supported 4-bit to 16-bit spectrum
        "IQ4_NL" | "Q4_K_M" | "Q4_K_S" | "Q4_0" | "Q4_1" | "IQ4_XS" |
        "Q5_K_M" | "Q5_K_S" | "Q5_0" | "Q5_1" | "Q6_K" | "Q8_0" |
        "F16" | "BF16" | "F32" => true,

        // High-fidelity 1-bit PrismML Bonsai support
        "Q1_0_G128" | "Q1_0" => true,

        // Legacy sub-4-bit quants remain rejected due to unaligned strides / collapse
        "IQ1_S" | "IQ1_M" | "IQ2_XXS" | "IQ2_XS" | "IQ2_S" | "Q2_K" |
        "IQ3_XXS" | "IQ3_S" | "Q3_K_S" | "Q3_K_M" | "Q3_K_L" => false,

        _ => false,
    }
}
```

### 5.2 Expand Model Topology Parser for 27B–31B Architectures (`src/container/xclbin_builder.rs`)
Modern models like `Gemma-4-31B` and `Qwen3.8-27B` feature large parameter counts, deeper metadata dictionaries, and large vocabulary dimensions (262,144 tokens).

1. **Increase Scan Limit:** Raise the metadata scan limit in `ModelGraphTopology::from_gguf` from 256 keys to 2048 keys.
2. **Add 27B/31B Topology Profiles:**
   - **Gemma-4-31B Profile:**
     - `hidden_dim: 5120`, `num_heads: 40`, `num_kv_heads: 16`, `num_layers: 56`, `ffn_dim: 24576`, `vocab_size: 262144`.
     - Handle Gemma's RMSNorm offset: $\text{norm}(x) = x \cdot (\text{weight} + 1.0)$.
   - **Qwen3.8-27B Profile:**
     - `hidden_dim: 5120`, `num_heads: 40`, `num_kv_heads: 8`, `num_layers: 64`, `ffn_dim: 17920`, `vocab_size: 262144`.
     - Standard SwiGLU projection structure.

### 5.3 RDNA 3.5 iGPU HIP/ROCm Prefill & Decode Kernel (`src/engine/rocm_prefill.rs`)
Targeting `/dev/dri/renderD128` (GFX1150 / Radeon 890M) using direct register bit manipulation:

```cpp
// Kernel: Dequantize and compute GEMV dot product on RDNA 3.5 iGPU
__global__ void gemv_q1_0_g128_f16(
    const block_q1_0_g128* __restrict__ weights, // Packed 1-bit weights
    const __half*          __restrict__ x,       // Input activation vector
    __half*                __restrict__ y,       // Output vector
    int k_dim,
    int n_dim
) {
    int row = blockIdx.x;
    if (row >= n_dim) return;

    int tid = threadIdx.x;
    int num_blocks = k_dim / QK1_0; // 128 weights per block
    float accum = 0.0f;

    // Process blocks across warp threads
    for (int b = tid; b < num_blocks; b += blockDim.x) {
        const block_q1_0_g128& blk = weights[row * num_blocks + b];
        float scale = __half2float(blk.d);

        // Load 16 bytes (128 bits) as 4 uint32 values
        const uint32_t* qs32 = reinterpret_cast<const uint32_t*>(blk.qs);

        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t bits = qs32[w];
            int base_idx = b * 128 + w * 32;

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                // Branchless sign extraction using bit shift and condition
                float sign_w = ((bits >> i) & 1u) ? scale : -scale;
                float act = __half2float(x[base_idx + i]);
                accum += sign_w * act;
            }
        }
    }

    // Warp-level shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        accum += __shfl_down_sync(0xffffffff, accum, offset);
    }

    if (tid == 0) {
        y[row] = __float2half(accum);
    }
}
```

### 5.4 Zen 5 CPU AVX-512 Worker (`src/engine/cpu_worker.rs`)
On Zen 5 cores, 1-bit vector dot-products are performed using bit-parallel XOR and population count (`VPOPCNTDQ` + `VPXORD`):

For activations quantized to 1-bit signs $a_i \in \{-1, +1\}$:
$$w_i \cdot a_i = \begin{cases} +1 & \text{if } w_i = a_i \\ -1 & \text{if } w_i \ne a_i \end{cases}$$

$$\sum_{i=1}^{128} w_i a_i = 128 - 2 \cdot \text{popcount}(b_w \oplus b_a)$$

For FP32/FP16 continuous activations, AVX-512 unpacks 32 sign bits into vector registers and uses masked fused-multiply-add:

```rust
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
pub unsafe fn vec_dot_q1_0_g128_avx512(
    weights: &[u8],
    activations: &[f32],
    scale: f32,
) -> f32 {
    // 512-bit vector bitfield expansion and FMA loop
    // Computes dot-product at full L1/L2 cache streaming speeds
    // ...
}
```

---

## 6. Frontend `llama.cpp` Integration (`llamacpp-update/llama.cpp`)

### 6.1 `ggml/include/ggml.h`
Add `GGML_TYPE_Q1_0_G128` to `enum ggml_type`:

```c
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // ...
    GGML_TYPE_IQ4_NL  = 20,
    // ...
    GGML_TYPE_Q1_0_G128 = 42, // Unique ID for PrismML 1-bit format
    GGML_TYPE_COUNT,
};
```

### 6.2 `ggml/src/ggml.c`
Register type traits:

```c
[GGML_TYPE_Q1_0_G128] = {
    .type_name          = "q1_0_g128",
    .blck_size          = 128,
    .type_size          = sizeof(block_q1_0_g128),
    .is_quantized       = true,
    .to_float           = (ggml_to_float_t) dequantize_row_q1_0_g128,
    .from_float         = (ggml_from_float_t) quantize_row_q1_0_g128,
    .vec_dot            = (ggml_vec_dot_t) ggml_vec_dot_q1_0_g128,
    .vec_dot_type       = GGML_TYPE_F16,
},
```

---

## 7. Testing & Verification Suite

### 7.1 Stage 1: Partial / Unit Tests

1. **Numerical Reconstruction Unit Test (`tests/test_q1_quant.rs`):**
   ```bash
   cargo test --test test_q1_quant
   ```
   - Verifies L1-mean scale calculation across Gaussian-distributed weight tensors.
   - Confirms that dequantization reconstruction error matches theoretical expectation $\mathbb{E}[|w - \hat{w}|] \approx \sqrt{2/\pi}\sigma (1 - 2/\pi)$.
   - Verifies bit-packing byte consistency against known reference vectors.

2. **Parametric XCLBIN Synthesis Test (`apu_synth`):**
   ```bash
   # Test synthetic graph generation for Gemma-4-31B
   cargo run --bin apu-synth -- tests/fixtures/gemma-4-31b-dummy.gguf target/Gemma4-31B-NPU2.xclbin

   # Test synthetic graph generation for Qwen3.8-27B
   cargo run --bin apu-synth -- tests/fixtures/qwen3.8-27b-dummy.gguf target/Qwen3.8-27B-NPU2.xclbin
   ```
   - Confirms generated XCLBIN contains valid AIE2P 32-tile spatial instruction streams.

3. **Zero-Copy DMA-BUF Cross-Device Attachment Test:**
   ```bash
   cargo test --test test_physical_backend -- --nocapture
   ```
   - Allocates 4 GB shared memory buffer via `/dev/dri/renderD128`.
   - Exports Linux Prime FD and attaches to `/dev/accel/accel0`.
   - Verifies `drm_syncobj` hardware timeline fence signaling.

---

### 7.2 Stage 2: End-to-End Model Execution Tests

#### Test 1: `google/gemma-4-31B`
- **Model Ingestion & Quantization:**
  ```bash
  python3 converter/convert_to_prismml.py \
      --model-id google/gemma-4-31B \
      --output-gguf /models/gemma-4-31b-q1_0_g128.gguf \
      --apply-rotation
  ```
- **Inference Invocations:**
  ```bash
  # 1. Verification of model topology and hardware profile
  apu-model inspect /models/gemma-4-31b-q1_0_g128.gguf

  # 2. Run single-prompt inference on AMD AI 470 APU
  llama cli -m /models/gemma-4-31b-q1_0_g128.gguf \
      -p "Explain the physics of gravitational lensing in detail." \
      -n 256 --prefill gpu --decode gpu --apu-verbose

  # 3. Interactive conversational test
  llama cli -m /models/gemma-4-31b-q1_0_g128.gguf -cnv
  ```
- **Validation Criteria:**
  - Memory: Resident Set Size (RSS) $\le 7.5\text{ GB}$ (weights ~4.36 GB + KV cache).
  - Accuracy: Output text remains coherent without loops or garbled tokens.
  - Zero-Copy: Confirm zero host-side `memcpy` logs via `--apu-verbose`.

---

#### Test 2: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU`
- **Model Ingestion & Quantization:**
  ```bash
  python3 converter/convert_to_prismml.py \
      --model-id DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
      --output-gguf /models/qwen3.8-27b-turbo-q1_0_g128.gguf \
      --apply-rotation
  ```
- **Inference Invocations:**
  ```bash
  # 1. Inspect container
  apu-model inspect /models/qwen3.8-27b-turbo-q1_0_g128.gguf

  # 2. Run complex code generation prompt on AMD AI 470 APU
  llama cli -m /models/qwen3.8-27b-turbo-q1_0_g128.gguf \
      -p "Write a high-performance concurrent lock-free queue in Rust with full explanation." \
      -n 384 --prefill gpu --decode gpu --apu-verbose
  ```
- **Validation Criteria:**
  - Memory: Resident Set Size (RSS) $\le 6.8\text{ GB}$ (weights ~3.80 GB + KV cache).
  - Throughput: High generation tok/s across LPDDR5X UMA bus.
  - Accuracy: Syntactically valid Rust code with full reasoning traces.

---

## 8. Summary Table: File Changes Matrix

| Path | Component | Action | Description |
| :--- | :--- | :---: | :--- |
| `converter/convert_to_prismml.py` | Standalone Python Tool | **NEW** | Offline FP16/BF16 $\to$ `Q1_0_g128` GGUF converter with RHT outlier rotation. |
| `src/container/converter.rs` | Rust APU Engine | **MODIFY** | Permits `Q1_0_g128` while preserving all other quants (`IQ4_NL`, `Q4_K_M`, etc.). |
| `src/container/xclbin_builder.rs` | Rust APU Engine | **MODIFY** | Expands GGUF metadata scan limit; adds Gemma-4-31B & Qwen3.8-27B topology profiles. |
| `src/engine/rocm_prefill.rs` | Rust APU Engine | **MODIFY** | Implements RDNA 3.5 iGPU HIP bit-extraction GEMM/GEMV kernel (`/dev/dri/renderD128`). |
| `src/engine/cpu_worker.rs` | Rust APU Engine | **MODIFY** | Implements Zen 5 AVX-512 `_mm512_popcnt_epi64` / `_mm512_xor_si512` dot product. |
| `QUANTIZATION.md` | Documentation | **MODIFY** | Documents `Q1_0_g128` format specs and coexistence with Q4–Q16 formats. |
| `llamacpp-update/llama.cpp/ggml/include/ggml.h` | Frontend / GGML | **MODIFY** | Registers `GGML_TYPE_Q1_0_G128` and `block_q1_0_g128`. |
| `llamacpp-update/llama.cpp/ggml/src/ggml.c` | Frontend / GGML | **MODIFY** | Adds reference quantize/dequantize and AVX-512 vector dot functions. |
| `llamacpp-update/llama.cpp/src/llama-context.cpp`| Frontend / Multiplexer | **MODIFY** | Ensures `Q1_0_g128` tensors pass cleanly into `apu_backend_load_model`. |
