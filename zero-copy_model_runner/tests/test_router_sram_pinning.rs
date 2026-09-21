// SPDX-License-Identifier: Apache-2.0
//! Integration Tests for Multi-Layer MoE Router Matrix (W_gate) SRAM Pinning & Safety Ceiling.

use zero_copy_model_runner::container::xclbin_builder::{
    ModelGraphTopology, TargetHardware, XclbinBuilder,
};

#[test]
fn test_dense_model_sram_allocation() {
    let dense_topo = ModelGraphTopology {
        arch_name: "llama".to_string(),
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_layers: 32,
        vocab_size: 32000,
        context_length: 4096,
        head_dim: 128,
        ffn_dim: 11008,
        num_experts: 0, // Dense model: no MoE router matrices
    };

    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, dense_topo);
    let plan = builder.plan_router_sram();

    assert_eq!(plan.total_router_bytes, 0, "Dense model has no router matrices");
    assert_eq!(plan.pinned_sram_bytes, 0, "Dense model must allocate 0 bytes in SRAM");
    assert_eq!(plan.pinned_layer_count, 0);
    assert!(!plan.sram_exhaustion_prevented);
}

#[test]
fn test_standard_moe_sram_allocation() {
    // Mixtral-style MoE: 8 experts, hidden_dim 4096, 32 layers
    let mixtral_topo = ModelGraphTopology {
        arch_name: "mixtral".to_string(),
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_layers: 32,
        vocab_size: 32000,
        context_length: 4096,
        head_dim: 128,
        ffn_dim: 14336,
        num_experts: 8,
    };

    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, mixtral_topo);
    let plan = builder.plan_router_sram();

    // Bytes per layer: 4096 * 8 * 2 = 65,536 bytes (64 KB)
    // Total router bytes: 32 * 65,536 = 2,097,152 bytes (2 MB)
    assert_eq!(plan.total_router_bytes, 2 * 1024 * 1024);
    assert_eq!(plan.pinned_sram_bytes, 2 * 1024 * 1024);
    assert_eq!(plan.pinned_layer_count, 32, "All 32 layers fit comfortably within SRAM");
    assert!(!plan.sram_exhaustion_prevented, "No ceiling breach for 2MB router matrix");
}

#[test]
fn test_massive_moe_sram_ceiling_clamping() {
    // DeepSeek-style MoE: 256 experts, hidden_dim 7168, 61 layers
    let deepseek_topo = ModelGraphTopology {
        arch_name: "deepseek_v3".to_string(),
        hidden_dim: 7168,
        num_heads: 128,
        num_kv_heads: 128,
        num_layers: 61,
        vocab_size: 129280,
        context_length: 4096,
        head_dim: 128,
        ffn_dim: 18432,
        num_experts: 256,
    };

    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, deepseek_topo);
    let plan = builder.plan_router_sram();

    // Bytes per layer: 7168 * 256 * 2 = 3,670,016 bytes (~3.5 MB)
    // Total unconstrained router bytes: 61 * 3,670,016 = 223,870,976 bytes (~213.5 MB)
    let expected_bytes_per_layer = 7168 * 256 * 2;
    let expected_total_bytes = 61 * expected_bytes_per_layer;
    assert_eq!(plan.total_router_bytes, expected_total_bytes);

    // 32MB safety ceiling = 32 * 1024 * 1024 = 33,554,432 bytes
    // Pinned layers: floor(33,554,432 / 3,670,016) = 9 layers
    let expected_pinned_layers = (32 * 1024 * 1024) / expected_bytes_per_layer;
    let expected_pinned_bytes = expected_pinned_layers * expected_bytes_per_layer;

    assert_eq!(plan.pinned_layer_count, expected_pinned_layers);
    assert_eq!(plan.pinned_sram_bytes, expected_pinned_bytes);
    assert!(
        plan.pinned_sram_bytes <= 32 * 1024 * 1024,
        "Pinned SRAM bytes ({}) must not exceed 32 MB ceiling",
        plan.pinned_sram_bytes
    );
    assert!(
        plan.sram_exhaustion_prevented,
        "Safety ceiling flag must be true to indicate hardware exhaustion was avoided"
    );
}

#[test]
fn test_custom_sram_limit_override() {
    let mixtral_topo = ModelGraphTopology {
        arch_name: "mixtral".to_string(),
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_layers: 32,
        vocab_size: 32000,
        context_length: 4096,
        head_dim: 128,
        ffn_dim: 14336,
        num_experts: 8,
    };

    // Restrict ceiling to 512 KB (leaves space for only 8 layers of 64 KB each)
    let limit_bytes = 512 * 1024;
    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, mixtral_topo)
        .with_router_sram(true, Some(limit_bytes));
    let plan = builder.plan_router_sram();

    assert_eq!(plan.pinned_layer_count, 8);
    assert_eq!(plan.pinned_sram_bytes, 512 * 1024);
    assert!(plan.sram_exhaustion_prevented);
}

#[test]
fn test_router_sram_explicitly_disabled() {
    let mixtral_topo = ModelGraphTopology {
        arch_name: "mixtral".to_string(),
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_layers: 32,
        vocab_size: 32000,
        context_length: 4096,
        head_dim: 128,
        ffn_dim: 14336,
        num_experts: 8,
    };

    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, mixtral_topo)
        .with_router_sram(false, None);
    let plan = builder.plan_router_sram();

    assert_eq!(plan.total_router_bytes, 0);
    assert_eq!(plan.pinned_sram_bytes, 0);
    assert_eq!(plan.pinned_layer_count, 0);
    assert!(!plan.sram_exhaustion_prevented);
}

#[test]
fn test_xclbin_metadata_contains_router_sram() {
    let mixtral_topo = ModelGraphTopology {
        arch_name: "mixtral".to_string(),
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        num_layers: 32,
        vocab_size: 32000,
        context_length: 4096,
        head_dim: 128,
        ffn_dim: 14336,
        num_experts: 8,
    };

    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, mixtral_topo);
    let xclbin_bytes = builder.generate_xclbin_bytes().expect("Synthesize XCLBIN");

    // Check that the synthesized binary contains extended-data JSON with router_sram tags
    let content = String::from_utf8_lossy(&xclbin_bytes);
    assert!(
        content.contains("router_sram_pinned_bytes"),
        "Generated XCLBIN extended metadata must contain 'router_sram_pinned_bytes'"
    );
    assert!(
        content.contains("router_sram_layers"),
        "Generated XCLBIN extended metadata must contain 'router_sram_layers'"
    );
}
