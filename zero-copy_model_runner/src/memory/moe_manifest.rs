// SPDX-License-Identifier: Apache-2.0
//! MoE Manifest Parser & 64-Byte / 4KB Aligned Tensor Slicing.
//!
//! Extracts fused expert tensor layout geometries (e.g. `blk.N.ffn_up_exps.weight`)
//! to support in-place zero-copy virtual memory slicing and DMA-BUF streaming.

use std::collections::HashMap;

/// Metadata descriptor for a single expert tensor slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSliceDescriptor {
    /// Full tensor identifier (e.g. `blk.5.ffn_up_exps.weight`).
    pub tensor_name: String,
    /// Layer index within the transformer stack.
    pub layer_idx: usize,
    /// Expert index within this layer (0 .. num_experts - 1).
    pub expert_idx: usize,
    /// Absolute byte offset in the underlying model file / payload.
    pub file_offset: u64,
    /// Byte size of this individual expert slice.
    pub slice_byte_size: usize,
    /// Quantization format name (e.g. "Q4_K_M", "BILLM", "Q8_0").
    pub quant_type: String,
    /// Total experts in the parent fused tensor.
    pub total_experts_in_tensor: usize,
}

impl ExpertSliceDescriptor {
    /// Validate that the slice adheres to 64-byte cacheline alignment.
    #[inline]
    pub fn is_64byte_aligned(&self) -> bool {
        self.file_offset % 64 == 0 && self.slice_byte_size % 64 == 0
    }

    /// Validate that the slice adheres to 4KB IOMMU page alignment.
    #[inline]
    pub fn is_page_aligned(&self) -> bool {
        self.file_offset % 4096 == 0
    }
}

/// Manifest catalog mapping layer indices to expert tensor slices.
#[derive(Debug, Clone, Default)]
pub struct MoeManifest {
    /// Slices grouped by layer index.
    pub layer_experts: HashMap<usize, Vec<ExpertSliceDescriptor>>,
    /// Total expert tensors discovered.
    pub total_tensors: usize,
    /// Number of experts per layer.
    pub num_experts_per_layer: usize,
    /// Byte size per individual expert slice (average or uniform).
    pub expert_slice_size_bytes: usize,
    /// Total dense (non-expert) weights byte size.
    pub dense_weights_bytes: usize,
}

impl MoeManifest {
    /// Register a discovered expert tensor and decompose into aligned slice descriptors.
    pub fn register_fused_expert_tensor(
        &mut self,
        tensor_name: &str,
        layer_idx: usize,
        file_offset: u64,
        total_byte_size: usize,
        num_experts: usize,
        quant_type: &str,
    ) {
        if num_experts == 0 {
            return;
        }

        let slice_size = total_byte_size / num_experts;
        self.num_experts_per_layer = self.num_experts_per_layer.max(num_experts);
        self.expert_slice_size_bytes = slice_size;
        self.total_tensors += 1;

        let entries = self.layer_experts.entry(layer_idx).or_default();

        for e_idx in 0..num_experts {
            let slice_offset = file_offset + (e_idx * slice_size) as u64;
            entries.push(ExpertSliceDescriptor {
                tensor_name: tensor_name.to_string(),
                layer_idx,
                expert_idx: e_idx,
                file_offset: slice_offset,
                slice_byte_size: slice_size,
                quant_type: quant_type.to_string(),
                total_experts_in_tensor: num_experts,
            });
        }
    }

    /// Check if the model is an MoE architecture.
    pub fn is_moe(&self) -> bool {
        self.num_experts_per_layer > 1 && !self.layer_experts.is_empty()
    }

    /// Query slice descriptor for a given layer and expert index.
    pub fn get_slice(
        &self,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Option<&ExpertSliceDescriptor> {
        self.layer_experts.get(&layer_idx).and_then(|slices| {
            slices.iter().find(|s| s.expert_idx == expert_idx)
        })
    }
}
