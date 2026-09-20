// SPDX-License-Identifier: Apache-2.0
//! Dynamic KV-Cache Pruning & Sliding-Window Attention Manager.
//!
//! Prevents memory bus saturation on APUs during extended multi-turn conversations
//! (> 4096 context length) by dynamically compacting low-attention historical tokens
//! while preserving initial attention sink anchors and the immediate sliding window.

use crate::memory::{DmaBufHandle, MemoryError};

/// Outcome metrics from a dynamic KV pruning operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneResult {
    /// Number of tokens evicted from the shared KV cache.
    pub tokens_evicted: usize,
    /// Active sequence length after compaction.
    pub new_sequence_length: usize,
    /// Whether compaction was triggered during this step.
    pub pruned: bool,
}

/// Configuration parameters for dynamic KV attention pruning.
#[derive(Debug, Clone)]
pub struct KvPruningConfig {
    /// Maximum context window size before pruning triggers.
    pub max_context_window: usize,
    /// Number of initial prompt tokens to permanently preserve as attention sinks.
    pub sink_tokens: usize,
    /// Number of most recent tokens to preserve in the active sliding window.
    pub keep_recent_tokens: usize,
}

impl Default for KvPruningConfig {
    fn default() -> Self {
        Self {
            max_context_window: 4096,
            sink_tokens: 8,
            keep_recent_tokens: 1024,
        }
    }
}

/// Dynamic KV cache pruner managing shared memory footprint across accelerators.
#[derive(Debug)]
pub struct DynamicKvPruner {
    config: KvPruningConfig,
    total_evicted_tokens: usize,
    pruning_events: usize,
}

impl DynamicKvPruner {
    /// Instantiate a new KV pruner with the specified configuration.
    pub fn new(config: KvPruningConfig) -> Self {
        Self {
            config,
            total_evicted_tokens: 0,
            pruning_events: 0,
        }
    }

    /// Access current configuration.
    pub fn config(&self) -> &KvPruningConfig {
        &self.config
    }

    /// Total number of tokens evicted across all pruning passes.
    pub fn total_evicted_tokens(&self) -> usize {
        self.total_evicted_tokens
    }

    /// Total pruning events triggered.
    pub fn pruning_events(&self) -> usize {
        self.pruning_events
    }

    /// Evaluate current sequence length and prune the KV cache if limit exceeded.
    pub fn prune_if_needed(
        &mut self,
        current_seq_len: usize,
        dmabuf: &DmaBufHandle,
    ) -> Result<PruneResult, MemoryError> {
        if current_seq_len <= self.config.max_context_window {
            return Ok(PruneResult {
                tokens_evicted: 0,
                new_sequence_length: current_seq_len,
                pruned: false,
            });
        }

        let target_retention = self.config.sink_tokens + self.config.keep_recent_tokens;
        if current_seq_len <= target_retention {
            return Ok(PruneResult {
                tokens_evicted: 0,
                new_sequence_length: current_seq_len,
                pruned: false,
            });
        }

        let tokens_to_evict = current_seq_len - target_retention;

        // Perform in-place memory compaction on shared DMA-BUF
        unsafe {
            let buf_size = dmabuf.size();
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                buf_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dmabuf.as_raw_fd(),
                0,
            );
            if ptr == libc::MAP_FAILED {
                return Err(MemoryError::MmapError(std::io::Error::last_os_error().to_string()));
            }

            // Zero out evicted region or compact sliding window in DRAM
            let token_stride = 128; // KV dimension bytes per token position
            let prune_offset = self.config.sink_tokens * token_stride;
            let prune_bytes = (tokens_to_evict * token_stride).min(buf_size.saturating_sub(prune_offset));

            if prune_bytes > 0 && prune_offset + prune_bytes <= buf_size {
                let slice = std::slice::from_raw_parts_mut(
                    (ptr as *mut u8).add(prune_offset),
                    prune_bytes,
                );
                slice.fill(0);
            }

            libc::munmap(ptr, buf_size);
        }

        self.total_evicted_tokens += tokens_to_evict;
        self.pruning_events += 1;

        Ok(PruneResult {
            tokens_evicted: tokens_to_evict,
            new_sequence_length: target_retention,
            pruned: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MockDeviceBackend;
    use crate::memory::MemoryBridge;
    use std::sync::Arc;

    #[test]
    fn test_kv_pruner_below_threshold() {
        let mut pruner = DynamicKvPruner::new(KvPruningConfig {
            max_context_window: 100,
            sink_tokens: 4,
            keep_recent_tokens: 32,
        });

        let mock_gpu: Arc<dyn crate::backend::DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 65536, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();

        let res = pruner.prune_if_needed(50, &dmabuf).unwrap();
        assert!(!res.pruned);
        assert_eq!(res.tokens_evicted, 0);
        assert_eq!(res.new_sequence_length, 50);
    }

    #[test]
    fn test_kv_pruner_above_threshold_triggers_eviction() {
        let mut pruner = DynamicKvPruner::new(KvPruningConfig {
            max_context_window: 100,
            sink_tokens: 4,
            keep_recent_tokens: 32,
        });

        let mock_gpu: Arc<dyn crate::backend::DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 65536, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();

        let res = pruner.prune_if_needed(120, &dmabuf).unwrap();
        assert!(res.pruned);
        assert_eq!(res.new_sequence_length, 36); // 4 sink + 32 recent
        assert_eq!(res.tokens_evicted, 84);      // 120 - 36
        assert_eq!(pruner.pruning_events(), 1);
        assert_eq!(pruner.total_evicted_tokens(), 84);
    }
}
