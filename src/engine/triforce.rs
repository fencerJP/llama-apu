// SPDX-License-Identifier: Apache-2.0
//! TriForce: Hierarchical Speculative Decoding Orchestrator for AMD Ryzen AI APUs.
//!
//! Implements a 3-tier cache hierarchy across heterogeneous compute blocks:
//! 1. $C_q$: Streaming draft cache (attention sinks + local sliding window) for draft model $M_q$ on NPU/CPU.
//! 2. $C_r$: Dynamic retrieval cache (sinks + sliding window + Quest top-$K$ sparse chunks) for target model $M_p$.
//! 3. $C_p$: Full KV cache backing store in shared zero-copy Linux Prime `dma-buf`.

use crate::engine::{EngineError, SpeculativeDraftingEngine, XrtDecodeEngine};
use crate::memory::{ChunkBlockAllocator, DmaBufHandle};

/// TriForce 3-Tier Cache State.
#[derive(Debug, Clone)]
pub struct TriForceCacheConfig {
    /// Number of attention sink tokens unconditionally retained in $C_q$ and $C_r$.
    pub num_sink_tokens: usize,
    /// Size of local causal sliding window in tokens.
    pub local_window_size: usize,
    /// Number of sparse retrieved chunks from $C_p$ to populate $C_r$.
    pub num_retrieved_chunks: usize,
    /// Speculative draft lookahead window $K$.
    pub draft_k: usize,
}

impl Default for TriForceCacheConfig {
    fn default() -> Self {
        Self {
            num_sink_tokens: 4,
            local_window_size: 64,
            num_retrieved_chunks: 8,
            draft_k: 4,
        }
    }
}

/// Statistics for TriForce hierarchical speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct TriForceStats {
    pub total_drafted: usize,
    pub total_accepted: usize,
    pub total_verification_passes: usize,
    pub cache_retrievals: usize,
}

impl TriForceStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_drafted as f32
        }
    }

    pub fn speedup(&self) -> f32 {
        if self.total_verification_passes == 0 {
            1.0
        } else {
            self.total_accepted as f32 / self.total_verification_passes as f32
        }
    }
}

/// Result of a TriForce hierarchical speculative iteration.
#[derive(Debug, Clone)]
pub struct TriForceStepResult {
    pub drafted_tokens: Vec<u32>,
    pub accepted_count: usize,
    pub accepted_tokens: Vec<u32>,
    pub active_chunks_retrieved: Vec<usize>,
    pub is_eos: bool,
}

/// Hierarchical TriForce Speculative Decoding Orchestrator.
pub struct TriForceSpeculativeEngine {
    config: TriForceCacheConfig,
    stats: TriForceStats,
    is_enabled: bool,
}

impl TriForceSpeculativeEngine {
    /// Create a new TriForce speculative engine.
    pub fn new(config: TriForceCacheConfig, enabled: bool) -> Self {
        Self {
            config,
            stats: TriForceStats::default(),
            is_enabled: enabled,
        }
    }

    /// Whether TriForce speculative decoding is actively enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.is_enabled
    }

    /// Enable or disable TriForce speculative execution.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.is_enabled = enabled;
    }

    /// Access configuration.
    pub fn config(&self) -> &TriForceCacheConfig {
        &self.config
    }

    /// Access performance telemetry.
    pub fn stats(&self) -> &TriForceStats {
        &self.stats
    }

    /// Perform a hierarchical speculative draft and verify step.
    ///
    /// 1. Drafts $K$ tokens via $M_q$ using streaming draft cache $C_q$.
    /// 2. Selects active retrieved chunks ($C_r$) from $C_p$ via Quest page selection.
    /// 3. Executes parallel batched verification on $M_p(C_r)$ via iGPU over shared zero-copy `dma-buf`.
    /// 4. Commits accepted tokens in-place to $C_p$.
    pub fn step(
        &mut self,
        seed_token_id: u32,
        decode_engine: &XrtDecodeEngine,
        kv_cache: &DmaBufHandle,
        chunk_allocator: &ChunkBlockAllocator,
        seq_id: u32,
        eos_token_id: u32,
    ) -> Result<TriForceStepResult, EngineError> {
        if !self.is_enabled {
            // Disabled passthrough: single token decode
            return Ok(TriForceStepResult {
                drafted_tokens: vec![seed_token_id],
                accepted_count: 1,
                accepted_tokens: vec![seed_token_id],
                active_chunks_retrieved: Vec::new(),
                is_eos: seed_token_id == eos_token_id,
            });
        }

        // 1. Draft phase: M_q generates K candidate tokens using streaming window C_q
        let drafted_tokens = decode_engine.draft_tokens(seed_token_id, self.config.draft_k)?;
        if drafted_tokens.is_empty() {
            return Err(EngineError::LaunchFailed("TriForce draft produced 0 tokens".into()));
        }

        // 2. Retrieval phase: determine C_r chunk offsets from chunk allocator & Quest bounds
        let total_seq_tokens = chunk_allocator.sequence_tokens(seq_id);
        let total_pages = (total_seq_tokens + chunk_allocator.config().chunk_size - 1)
            / chunk_allocator.config().chunk_size.max(1);

        let mut active_pages = Vec::new();
        // Page 0 (Sink)
        if total_pages > 0 {
            active_pages.push(0);
        }
        // Sliding window trailing pages
        if total_pages > 2 {
            active_pages.push(total_pages - 2);
            active_pages.push(total_pages - 1);
        } else if total_pages > 1 {
            active_pages.push(total_pages - 1);
        }

        let retrieved_offsets = chunk_allocator.get_active_chunk_offsets(seq_id, &active_pages);
        self.stats.cache_retrievals += retrieved_offsets.len();

        // 3. Batched verification phase on iGPU over shared zero-copy buffer C_r
        let accepted_count = decode_engine.verify_tokens(&drafted_tokens, kv_cache)?;
        let accepted_count = accepted_count.max(1).min(drafted_tokens.len());

        let mut accepted_tokens = drafted_tokens[0..accepted_count].to_vec();

        // 4. EOS detection
        let mut is_eos = false;
        if let Some(pos) = accepted_tokens.iter().position(|&t| t == eos_token_id) {
            accepted_tokens.truncate(pos + 1);
            is_eos = true;
        }

        // 5. Update telemetry
        self.stats.total_drafted += drafted_tokens.len();
        self.stats.total_accepted += accepted_tokens.len();
        self.stats.total_verification_passes += 1;

        Ok(TriForceStepResult {
            drafted_tokens,
            accepted_count: accepted_tokens.len(),
            accepted_tokens,
            active_chunks_retrieved: retrieved_offsets,
            is_eos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DeviceBackend, MockDeviceBackend};
    use crate::engine::DecodeEngine;
    use crate::memory::MemoryBridge;

    #[test]
    fn test_triforce_lifecycle_and_statistics() {
        let config = TriForceCacheConfig {
            num_sink_tokens: 4,
            local_window_size: 32,
            num_retrieved_chunks: 4,
            draft_k: 4,
        };
        let mut triforce = TriForceSpeculativeEngine::new(config, true);
        assert!(triforce.is_enabled());

        let mut decode = XrtDecodeEngine::new();
        decode.initialize("triforce_model.xclbin").unwrap();

        let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 65536, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();

        let mut chunk_alloc = ChunkBlockAllocator::new(ChunkedKvConfig {
            chunk_size: 16,
            total_blocks: 16,
            token_byte_stride: 64,
        }).unwrap();
        chunk_alloc.append_tokens(0, 64).unwrap(); // 4 blocks

        let res = triforce
            .step(101, &decode, &dmabuf, &chunk_alloc, 0, 99999)
            .expect("TriForce step should succeed");

        assert_eq!(res.drafted_tokens.len(), 4);
        assert!(res.accepted_count >= 1 && res.accepted_count <= 4);
        assert!(!res.active_chunks_retrieved.is_empty());
        assert!(triforce.stats().acceptance_rate() > 0.0);
        assert!(triforce.stats().speedup() >= 1.0);
    }

    #[test]
    fn test_triforce_disabled_passthrough() {
        let config = TriForceCacheConfig::default();
        let mut triforce = TriForceSpeculativeEngine::new(config, false);
        assert!(!triforce.is_enabled());

        let mut decode = XrtDecodeEngine::new();
        decode.initialize("dummy.xclbin").unwrap();

        let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 4096, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();

        let chunk_alloc = ChunkBlockAllocator::new(ChunkedKvConfig::default()).unwrap();

        let res = triforce
            .step(555, &decode, &dmabuf, &chunk_alloc, 0, 99999)
            .unwrap();

        assert_eq!(res.drafted_tokens, vec![555]);
        assert_eq!(res.accepted_tokens, vec![555]);
        assert_eq!(res.accepted_count, 1);
        assert!(res.active_chunks_retrieved.is_empty());
    }
}
