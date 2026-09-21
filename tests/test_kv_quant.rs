// SPDX-License-Identifier: Apache-2.0
//! Integration and Precision Tests for CSA2 / Quantized Dynamic KV Cache (INT8 / INT4).

use zero_copy_model_runner::memory::chunked_kv::ChunkedKvConfig;
use zero_copy_model_runner::memory::kv_quant::{
    dequantize_block_int4, dequantize_block_int8, evaluate_kv_quant_compatibility,
    quantize_block_int4, quantize_block_int8, KvCacheQuantType,
};
use zero_copy_model_runner::memory::sys_mem::{MoeMemoryPlan, SystemMemoryInfo};

#[test]
fn test_int8_quant_dequant_precision() {
    // Generate test signal with varying amplitudes in [-2.5, 2.5]
    let mut input = [0.0f32; 32];
    for i in 0..32 {
        let phase = (i as f32) * std::f32::consts::PI / 8.0;
        input[i] = 2.5 * phase.sin();
    }

    let (quantized, scale) = quantize_block_int8(&input);
    let reconstructed = dequantize_block_int8(&quantized, scale);

    let mut mse = 0.0f32;
    let mut max_err = 0.0f32;
    for i in 0..32 {
        let err = (input[i] - reconstructed[i]).abs();
        mse += err * err;
        if err > max_err {
            max_err = err;
        }
    }
    mse /= 32.0;

    // INT8 quantization with scale across [-2.5, 2.5] has step size 2.5 / 127 ~= 0.0197
    // Expected MSE < 1e-4, max error < 0.02
    assert!(
        mse < 1e-4,
        "INT8 MSE too high: {} (expected < 1e-4)",
        mse
    );
    assert!(
        max_err < 0.025,
        "INT8 max error too high: {} (expected < 0.025)",
        max_err
    );
}

#[test]
fn test_int4_quant_dequant_precision() {
    // Generate test signal in [-1.5, 1.5]
    let mut input = [0.0f32; 32];
    for i in 0..32 {
        let t = (i as f32) / 31.0;
        input[i] = -1.5 + 3.0 * t; // Linear ramp from -1.5 to 1.5
    }

    let (packed, scale) = quantize_block_int4(&input);
    assert_eq!(packed.len(), 16, "INT4 block must pack 32 elements into 16 bytes");

    let reconstructed = dequantize_block_int4(&packed, scale);

    let mut mse = 0.0f32;
    let mut max_err = 0.0f32;
    for i in 0..32 {
        let err = (input[i] - reconstructed[i]).abs();
        mse += err * err;
        if err > max_err {
            max_err = err;
        }
    }
    mse /= 32.0;

    // INT4 step size is ~1.5 / 7 ~= 0.214
    // Expected MSE < 0.02, max error < 0.25
    assert!(
        mse < 0.02,
        "INT4 MSE too high: {} (expected < 0.02)",
        mse
    );
    assert!(
        max_err < 0.25,
        "INT4 max error too high: {} (expected < 0.25)",
        max_err
    );
}

#[test]
fn test_kv_quant_compatibility_matrix() {
    // 1. Sensitive non-linear codebook model (IQ4_NL) -> Auto-fallback to FP16
    let res_iq4 = evaluate_kv_quant_compatibility(
        "qwen2",
        128,
        8192,
        "IQ4_NL",
        KvCacheQuantType::Auto,
    );
    assert_eq!(res_iq4.recommended_type, KvCacheQuantType::Fp16);
    assert!(res_iq4.reason.contains("IQ4_NL"));

    // 2. Small head dimension (head_dim = 32 < 64) -> Auto-fallback to FP16
    let res_small_head = evaluate_kv_quant_compatibility(
        "custom_small",
        32,
        8192,
        "Q4_K_M",
        KvCacheQuantType::Auto,
    );
    assert_eq!(res_small_head.recommended_type, KvCacheQuantType::Fp16);
    assert!(res_small_head.reason.contains("< 64"));

    // 3. Multi-Head Latent Attention (MLA / DeepSeek V4) -> Auto-selects INT8, strictly protects from INT4
    let res_deepseek = evaluate_kv_quant_compatibility(
        "deepseek_v4",
        128,
        32768,
        "Q4_K_M",
        KvCacheQuantType::Auto,
    );
    assert_eq!(res_deepseek.recommended_type, KvCacheQuantType::Int8);
    assert!(res_deepseek.reason.contains("MLA") || res_deepseek.reason.contains("Latent Attention"));

    // 4. Short context (2048) -> Keeps FP16 to avoid quantization overhead when memory pressure is low
    let res_short_ctx = evaluate_kv_quant_compatibility(
        "llama3",
        128,
        2048,
        "Q4_K_M",
        KvCacheQuantType::Auto,
    );
    assert_eq!(res_short_ctx.recommended_type, KvCacheQuantType::Fp16);

    // 5. Long context (16384) on standard GQA -> Auto-selects INT8
    let res_long_ctx = evaluate_kv_quant_compatibility(
        "llama3",
        128,
        16384,
        "Q4_K_M",
        KvCacheQuantType::Auto,
    );
    assert_eq!(res_long_ctx.recommended_type, KvCacheQuantType::Int8);

    // 6. Explicit user override -> Forced regardless of model type
    let res_forced = evaluate_kv_quant_compatibility(
        "deepseek_v4",
        32,
        2048,
        "IQ4_NL",
        KvCacheQuantType::Int4,
    );
    assert_eq!(res_forced.recommended_type, KvCacheQuantType::Int4);
}

#[test]
fn test_dynamic_dram_reclamation_for_moe_experts() {
    let mem_info = SystemMemoryInfo {
        total_dram_bytes: 48 * 1024 * 1024 * 1024,       // 48 GB APU (e.g. Strix Point / Gorgon)
        available_dram_bytes: 40 * 1024 * 1024 * 1024,   // 40 GB available
        gtt_limit_bytes: 44 * 1024 * 1024 * 1024,
        os_headroom_bytes: 4 * 1024 * 1024 * 1024,       // 4 GB OS headroom
    };

    let dense_weights_bytes = 10 * 1024 * 1024 * 1024;    // 10 GB dense weights
    let total_experts_per_tensor = 64;
    let per_expert_slice_bytes = 128 * 1024 * 1024;       // 128 MB per expert slice
    let num_layers = 32;
    let num_kv_heads = 8;
    let head_dim = 128;
    let max_context_length = 65536;                       // 64K context window

    // 1. Calculate baseline plan with FP16 KV Cache
    let plan_fp16 = MoeMemoryPlan::calculate_with_quant(
        &mem_info,
        dense_weights_bytes,
        total_experts_per_tensor,
        per_expert_slice_bytes,
        num_layers,
        num_kv_heads,
        head_dim,
        max_context_length,
        None,
        true,
        KvCacheQuantType::Fp16,
    );

    // 2. Calculate plan with INT8 KV Cache
    let plan_int8 = MoeMemoryPlan::calculate_with_quant(
        &mem_info,
        dense_weights_bytes,
        total_experts_per_tensor,
        per_expert_slice_bytes,
        num_layers,
        num_kv_heads,
        head_dim,
        max_context_length,
        None,
        true,
        KvCacheQuantType::Int8,
    );

    // 3. Calculate plan with INT4 KV Cache
    let plan_int4 = MoeMemoryPlan::calculate_with_quant(
        &mem_info,
        dense_weights_bytes,
        total_experts_per_tensor,
        per_expert_slice_bytes,
        num_layers,
        num_kv_heads,
        head_dim,
        max_context_length,
        None,
        true,
        KvCacheQuantType::Int4,
    );

    println!("FP16 KV Reserved: {} MB, Hot Experts: {}", plan_fp16.kv_cache_reserved_bytes / (1024 * 1024), plan_fp16.resident_experts_per_tensor);
    println!("INT8 KV Reserved: {} MB, Hot Experts: {}", plan_int8.kv_cache_reserved_bytes / (1024 * 1024), plan_int8.resident_experts_per_tensor);
    println!("INT4 KV Reserved: {} MB, Hot Experts: {}", plan_int4.kv_cache_reserved_bytes / (1024 * 1024), plan_int4.resident_experts_per_tensor);

    // Total elements = 2 * 32 * 8 * 128 * 65536 = 4,294,967,296 elements
    // FP16 reservation: 4.29B * 2.0 B = 8,589,934,592 B (8 GB)
    // INT8 reservation: 4.29B * 1.0625 B ~= 4.56 GB (~3.44 GB saved)
    // INT4 reservation: 4.29B * 0.5625 B ~= 2.41 GB (~6.17 GB saved)
    assert!(
        plan_int8.kv_cache_reserved_bytes < plan_fp16.kv_cache_reserved_bytes,
        "INT8 KV cache reservation must be lower than FP16"
    );
    assert!(
        plan_int4.kv_cache_reserved_bytes < plan_int8.kv_cache_reserved_bytes,
        "INT4 KV cache reservation must be lower than INT8"
    );

    // Verify memory savings unlocked more resident hot experts
    assert!(
        plan_int8.resident_experts_per_tensor >= plan_fp16.resident_experts_per_tensor,
        "INT8 must unlock >= FP16 resident hot experts"
    );
    assert!(
        plan_int4.resident_experts_per_tensor >= plan_int8.resident_experts_per_tensor,
        "INT4 must unlock >= INT8 resident hot experts"
    );
    assert!(
        plan_int4.hot_expert_pool_bytes >= plan_fp16.hot_expert_pool_bytes,
        "Reclaimed KV memory must expand the resident hot expert pool"
    );
}

#[test]
fn test_chunked_kv_stride_quantization() {
    let base_config = ChunkedKvConfig {
        chunk_size: 32,
        total_blocks: 512,
        token_byte_stride: 512, // Baseline FP16 stride (e.g. 256 elements * 2 bytes)
    };

    let int8_config = base_config.with_quant_type(KvCacheQuantType::Int8, 512);
    let int4_config = base_config.with_quant_type(KvCacheQuantType::Int4, 512);

    // 512 * (1.0625 / 2.0) = 272 bytes. 272 is 16-byte aligned (17 * 16)
    assert_eq!(int8_config.token_byte_stride, 272);
    assert_eq!(int8_config.token_byte_stride % 16, 0, "Stride must be 16-byte aligned");

    // 512 * (0.5625 / 2.0) = 144 bytes. 144 is 16-byte aligned (9 * 16)
    assert_eq!(int4_config.token_byte_stride, 144);
    assert_eq!(int4_config.token_byte_stride % 16, 0, "Stride must be 16-byte aligned");

    assert!(int8_config.token_byte_stride < base_config.token_byte_stride);
    assert!(int4_config.token_byte_stride < int8_config.token_byte_stride);
}
