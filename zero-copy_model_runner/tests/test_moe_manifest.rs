use zero_copy_model_runner::memory::MoeManifest;

#[test]
fn test_moe_manifest_slicing_and_alignment() {
    let mut manifest = MoeManifest::default();

    // Register a 64-expert fused tensor aligned to 4096 bytes
    let file_offset = 4096u64;
    let total_bytes = 64 * 1024 * 1024; // 64 MB total -> 1 MB per expert
    let num_experts = 64;

    manifest.register_fused_expert_tensor(
        "blk.0.ffn_up_exps.weight",
        0,
        file_offset,
        total_bytes,
        num_experts,
        "BILLM",
    );

    assert!(manifest.is_moe());
    assert_eq!(manifest.num_experts_per_layer, 64);
    assert_eq!(manifest.expert_slice_size_bytes, 1024 * 1024);

    let slice_0 = manifest.get_slice(0, 0).expect("Slice 0 should exist");
    assert_eq!(slice_0.layer_idx, 0);
    assert_eq!(slice_0.expert_idx, 0);
    assert_eq!(slice_0.file_offset, 4096);
    assert_eq!(slice_0.slice_byte_size, 1024 * 1024);
    assert!(slice_0.is_64byte_aligned());
    assert!(slice_0.is_page_aligned());

    let slice_1 = manifest.get_slice(0, 1).expect("Slice 1 should exist");
    assert_eq!(slice_1.expert_idx, 1);
    assert_eq!(slice_1.file_offset, 4096 + 1024 * 1024);
    assert!(slice_1.is_64byte_aligned());
}

#[test]
fn test_dense_manifest_not_moe() {
    let manifest = MoeManifest::default();
    assert!(!manifest.is_moe());
}
