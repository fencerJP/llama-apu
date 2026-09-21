// SPDX-License-Identifier: Apache-2.0
//! System Memory & DRM GTT Capacity Prober for Heterogeneous APUs.
//!
//! Evaluates active physical DRAM capacity, OS safety headroom, and Linux DRM GTT
//! pin limits to calculate the maximum saturated number of resident MoE expert layers
//! without causing memory thrashing or kernel OOM panics.

use std::fs;
use std::path::Path;

/// Live hardware memory metrics captured from `/proc/meminfo` and Linux DRM sysfs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemoryInfo {
    /// Total physical RAM installed in bytes.
    pub total_dram_bytes: usize,
    /// Currently available memory for unpaged allocations in bytes.
    pub available_dram_bytes: usize,
    /// Maximum pin-able GTT (Graphics Translation Table) budget from DRM in bytes.
    pub gtt_limit_bytes: usize,
    /// Dynamic OS and running application safety headroom in bytes.
    pub os_headroom_bytes: usize,
}

impl SystemMemoryInfo {
    /// Probe the host system and resolve active memory metrics.
    pub fn probe() -> Self {
        let (total_dram, available_dram) = Self::read_meminfo();
        let gtt_limit = Self::read_drm_gtt_limit().unwrap_or(total_dram);

        // Dynamic OS safety margin: min 1.0 GB, max 3.5 GB, or 5% of total DRAM
        let five_percent = total_dram / 20;
        let os_headroom = five_percent.clamp(1024 * 1024 * 1024, (35 * 1024 * 1024 * 1024) / 10);

        Self {
            total_dram_bytes: total_dram,
            available_dram_bytes: available_dram,
            gtt_limit_bytes: gtt_limit,
            os_headroom_bytes: os_headroom,
        }
    }

    /// Read `MemTotal` and `MemAvailable` from `/proc/meminfo`.
    fn read_meminfo() -> (usize, usize) {
        let mut total = 32 * 1024 * 1024 * 1024; // 32 GB fallback default
        let mut available = 24 * 1024 * 1024 * 1024; // 24 GB fallback default

        if let Ok(content) = fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = parse_meminfo_kb(line) {
                        total = kb * 1024;
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(kb) = parse_meminfo_kb(line) {
                        available = kb * 1024;
                    }
                }
            }
        }

        // Check cgroups limits if present
        if let Ok(cgroup_limit_str) = fs::read_to_string("/sys/fs/cgroup/memory.max") {
            if let Ok(limit_bytes) = cgroup_limit_str.trim().parse::<usize>() {
                if limit_bytes < total {
                    total = limit_bytes;
                    available = available.min(total);
                }
            }
        }

        (total, available)
    }

    /// Read DRM GTT total pin-able memory limit from sysfs.
    fn read_drm_gtt_limit() -> Option<usize> {
        let candidates = [
            "/sys/class/drm/card0/device/mem_info_gtt_total",
            "/sys/class/drm/card1/device/mem_info_gtt_total",
            "/sys/class/drm/renderD128/device/mem_info_gtt_total",
        ];

        for path_str in &candidates {
            if Path::new(path_str).exists() {
                if let Ok(content) = fs::read_to_string(path_str) {
                    if let Ok(bytes) = content.trim().parse::<usize>() {
                        if bytes > 0 {
                            return Some(bytes);
                        }
                    }
                }
            }
        }
        None
    }

    /// Calculate dynamic streaming slab ring count based on total system memory and UMA bus width.
    pub fn dynamic_slab_count(&self) -> usize {
        let gib_total = self.total_dram_bytes / (1024 * 1024 * 1024);
        if gib_total <= 48 {
            4
        } else {
            8
        }
    }
}

fn parse_meminfo_kb(line: &str) -> Option<usize> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1].parse::<usize>().ok()
    } else {
        None
    }
}

use crate::memory::kv_quant::KvCacheQuantType;

/// Sizing budget planner for MoE active memory partitioning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeMemoryPlan {
    /// Number of expert layers / hot slots saturated into DRAM.
    pub resident_experts_per_tensor: usize,
    /// Total bytes allocated for the hot expert pool in DRAM.
    pub hot_expert_pool_bytes: usize,
    /// Active Key-Value cache quantization format.
    pub kv_quant_type: KvCacheQuantType,
    /// Bytes reserved for worst-case KV cache expansion ($L_{\max}$).
    pub kv_cache_reserved_bytes: usize,
    /// Bytes allocated for dense baseline layers (attention, embeddings, norms).
    pub dense_weights_bytes: usize,
    /// Dynamic streaming ring slab count.
    pub streaming_slab_count: usize,
    /// Total bytes allocated for the pre-allocated DMA-BUF streaming ring.
    pub streaming_ring_bytes: usize,
    /// Safety headroom left untouched in physical DRAM.
    pub preserved_headroom_bytes: usize,
}

impl MoeMemoryPlan {
    /// Calculate optimal saturated MoE memory plan using default FP16 KV cache.
    pub fn calculate(
        mem_info: &SystemMemoryInfo,
        dense_weights_bytes: usize,
        total_experts_per_tensor: usize,
        per_expert_slice_bytes: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_context_length: usize,
        manual_hot_experts: Option<usize>,
        experts_buffering_enabled: bool,
    ) -> Self {
        Self::calculate_with_quant(
            mem_info,
            dense_weights_bytes,
            total_experts_per_tensor,
            per_expert_slice_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            max_context_length,
            manual_hot_experts,
            experts_buffering_enabled,
            KvCacheQuantType::Fp16,
        )
    }

    /// Calculate optimal saturated MoE memory plan with explicit KV cache quantization.
    pub fn calculate_with_quant(
        mem_info: &SystemMemoryInfo,
        dense_weights_bytes: usize,
        total_experts_per_tensor: usize,
        per_expert_slice_bytes: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_context_length: usize,
        manual_hot_experts: Option<usize>,
        experts_buffering_enabled: bool,
        kv_quant_type: KvCacheQuantType,
    ) -> Self {
        // 1. Calculate worst-case KV cache requirement for L_max scaled by quantization format
        let total_elements = 2 * num_layers * num_kv_heads * head_dim * max_context_length;
        let bytes_per_elem = kv_quant_type.effective_bytes_per_element();
        let kv_cache_reserved = (total_elements as f64 * bytes_per_elem as f64).ceil() as usize;

        let streaming_slab_count = mem_info.dynamic_slab_count();
        let streaming_ring_bytes = streaming_slab_count * per_expert_slice_bytes;

        // 2. Net available capacity for model weights after OS headroom and L_max KV cache
        let usable_capacity = mem_info
            .available_dram_bytes
            .min(mem_info.gtt_limit_bytes)
            .saturating_sub(mem_info.os_headroom_bytes)
            .saturating_sub(kv_cache_reserved);

        if !experts_buffering_enabled {
            // Full in-memory loading mode: pin all experts
            let total_expert_bytes = total_experts_per_tensor * per_expert_slice_bytes;
            return Self {
                resident_experts_per_tensor: total_experts_per_tensor,
                hot_expert_pool_bytes: total_expert_bytes,
                kv_quant_type,
                kv_cache_reserved_bytes: kv_cache_reserved,
                dense_weights_bytes,
                streaming_slab_count: 0,
                streaming_ring_bytes: 0,
                preserved_headroom_bytes: mem_info.os_headroom_bytes,
            };
        }

        // 3. Saturated expert pool calculation
        let available_for_experts = usable_capacity
            .saturating_sub(dense_weights_bytes)
            .saturating_sub(streaming_ring_bytes);

        let auto_resident = if per_expert_slice_bytes > 0 {
            (available_for_experts / per_expert_slice_bytes).min(total_experts_per_tensor)
        } else {
            total_experts_per_tensor
        };

        // Guarantee at least 1 resident expert if capacity permits, or apply manual override
        let final_resident = if let Some(manual) = manual_hot_experts {
            manual.min(total_experts_per_tensor)
        } else {
            auto_resident.max(1).min(total_experts_per_tensor)
        };

        let hot_expert_pool_bytes = final_resident * per_expert_slice_bytes;

        Self {
            resident_experts_per_tensor: final_resident,
            hot_expert_pool_bytes,
            kv_quant_type,
            kv_cache_reserved_bytes: kv_cache_reserved,
            dense_weights_bytes,
            streaming_slab_count,
            streaming_ring_bytes,
            preserved_headroom_bytes: mem_info.os_headroom_bytes,
        }
    }
}
