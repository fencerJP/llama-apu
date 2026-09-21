// SPDX-License-Identifier: Apache-2.0
use zero_copy_model_runner::memory::{MoeMemoryPlan, SystemMemoryInfo};

#[test]
fn test_system_memory_probing() {
    let mem_info = SystemMemoryInfo::probe();
    assert!(mem_info.total_dram_bytes > 0);
    assert!(mem_info.available_dram_bytes > 0);
    assert!(mem_info.os_headroom_bytes >= 1024 * 1024 * 1024);
    assert!(mem_info.os_headroom_bytes <= 3500 * 1024 * 1024);

    let slab_count = mem_info.dynamic_slab_count();
    assert!(slab_count == 4 || slab_count == 8);
}

#[test]
fn test_moe_memory_plan_saturation() {
    let mem_info = SystemMemoryInfo {
        total_dram_bytes: 48 * 1024 * 1024 * 1024,
        available_dram_bytes: 36 * 1024 * 1024 * 1024,
        gtt_limit_bytes: 40 * 1024 * 1024 * 1024,
        os_headroom_bytes: 2500 * 1024 * 1024,
    };

    let dense_bytes = 4 * 1024 * 1024 * 1024; // 4 GB dense
    let total_experts = 64;
    let slice_bytes = 100 * 1024 * 1024; // 100 MB per expert
    let num_layers = 32;
    let num_kv_heads = 4;
    let head_dim = 128;
    let max_context = 4096;

    let plan = MoeMemoryPlan::calculate(
        &mem_info,
        dense_bytes,
        total_experts,
        slice_bytes,
        num_layers,
        num_kv_heads,
        head_dim,
        max_context,
        None,
        true,
    );

    assert!(plan.resident_experts_per_tensor >= 1);
    assert!(plan.resident_experts_per_tensor <= total_experts);
    assert_eq!(plan.streaming_slab_count, 4);
    assert!(plan.hot_expert_pool_bytes > 0);
    assert!(plan.kv_cache_reserved_bytes > 0);
    assert_eq!(plan.preserved_headroom_bytes, mem_info.os_headroom_bytes);
}

#[test]
fn test_moe_memory_plan_manual_override() {
    let mem_info = SystemMemoryInfo {
        total_dram_bytes: 64 * 1024 * 1024 * 1024,
        available_dram_bytes: 50 * 1024 * 1024 * 1024,
        gtt_limit_bytes: 60 * 1024 * 1024 * 1024,
        os_headroom_bytes: 3000 * 1024 * 1024,
    };

    let plan = MoeMemoryPlan::calculate(
        &mem_info,
        4 * 1024 * 1024 * 1024,
        64,
        100 * 1024 * 1024,
        32,
        4,
        128,
        4096,
        Some(16), // manual override
        true,
    );

    assert_eq!(plan.resident_experts_per_tensor, 16);
    assert_eq!(plan.streaming_slab_count, 8);
}

#[test]
fn test_moe_memory_plan_buffering_disabled() {
    let mem_info = SystemMemoryInfo::probe();
    let total_experts = 32;
    let slice_bytes = 50 * 1024 * 1024;

    let plan = MoeMemoryPlan::calculate(
        &mem_info,
        2 * 1024 * 1024 * 1024,
        total_experts,
        slice_bytes,
        16,
        4,
        64,
        2048,
        None,
        false, // disabled
    );

    assert_eq!(plan.resident_experts_per_tensor, total_experts);
    assert_eq!(plan.streaming_slab_count, 0);
    assert_eq!(plan.streaming_ring_bytes, 0);
}
