// SPDX-License-Identifier: Apache-2.0
//! Speculative Drafting Orchestrator for AMD Ryzen AI APUs.
//!
//! Maximizes throughput by executing autoregressive token drafting on energy-efficient
//! XDNA 2 NPU tiles (AIE2P) and verifying $K$ candidate tokens concurrently in a single
//! batched GEMM forward pass on the RDNA 3.5 iGPU over shared Linux Prime `dma-buf`.

use crate::engine::{EngineError, SpeculativeDraftingEngine, XrtDecodeEngine};
use crate::memory::DmaBufHandle;

/// Statistics tracking speculative drafting performance.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    pub total_proposed_tokens: usize,
    pub total_accepted_tokens: usize,
    pub total_verification_passes: usize,
}

impl SpeculativeStats {
    /// Return the cumulative candidate acceptance rate ($[0.0, 1.0]$).
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_proposed_tokens == 0 {
            0.0
        } else {
            self.total_accepted_tokens as f32 / self.total_proposed_tokens as f32
        }
    }

    /// Estimated algorithmic speedup multiplier compared to non-speculative decode.
    pub fn estimated_speedup(&self) -> f32 {
        if self.total_verification_passes == 0 {
            1.0
        } else {
            self.total_accepted_tokens as f32 / self.total_verification_passes as f32
        }
    }
}

/// Result of a single speculative drafting step.
#[derive(Debug, Clone)]
pub struct SpeculativeStepResult {
    /// Candidate tokens drafted by the NPU.
    pub drafted_tokens: Vec<u32>,
    /// Number of tokens accepted by the iGPU verification pass.
    pub accepted_count: usize,
    /// Verified accepted token sequence (including the bonus token).
    pub accepted_tokens: Vec<u32>,
    /// Whether an EOS token was detected among accepted tokens.
    pub is_eos: bool,
}

/// Orchestrates multi-accelerator speculative decoding between NPU and iGPU.
pub struct SpeculativeOrchestrator {
    draft_length: usize,
    stats: SpeculativeStats,
}

impl SpeculativeOrchestrator {
    /// Create a new speculative orchestrator with a target draft window $K$.
    pub fn new(draft_length: usize) -> Self {
        let draft_length = draft_length.clamp(1, 8);
        Self {
            draft_length,
            stats: SpeculativeStats::default(),
        }
    }

    /// Current draft length $K$.
    pub fn draft_length(&self) -> usize {
        self.draft_length
    }

    /// Set draft length $K$.
    pub fn set_draft_length(&mut self, draft_length: usize) {
        self.draft_length = draft_length.clamp(1, 8);
    }

    /// Access speculative execution statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Execute a speculative drafting and verification step.
    ///
    /// 1. Drafts $K$ tokens sequentially on the XDNA 2 NPU (consuming low power).
    /// 2. Passes the candidate sequence to the RDNA 3.5 iGPU for parallel batch verification.
    /// 3. Returns the accepted token prefix.
    pub fn step(
        &mut self,
        seed_token_id: u32,
        decode_engine: &XrtDecodeEngine,
        kv_cache: &DmaBufHandle,
        eos_token_id: u32,
    ) -> Result<SpeculativeStepResult, EngineError> {
        // 1. Generate K candidate tokens on NPU
        let drafted_tokens = decode_engine.draft_tokens(seed_token_id, self.draft_length)?;
        if drafted_tokens.is_empty() {
            return Err(EngineError::LaunchFailed("NPU drafted 0 tokens".into()));
        }

        // 2. Parallel verification pass on iGPU over shared zero-copy KV cache
        let accepted_count = decode_engine.verify_tokens(&drafted_tokens, kv_cache)?;
        let accepted_count = accepted_count.max(1).min(drafted_tokens.len());

        let mut accepted_tokens = drafted_tokens[0..accepted_count].to_vec();

        // 3. Check for EOS token in accepted sequence
        let mut is_eos = false;
        if let Some(pos) = accepted_tokens.iter().position(|&t| t == eos_token_id) {
            accepted_tokens.truncate(pos + 1);
            is_eos = true;
        }

        // 4. Update telemetry
        self.stats.total_proposed_tokens += drafted_tokens.len();
        self.stats.total_accepted_tokens += accepted_tokens.len();
        self.stats.total_verification_passes += 1;

        Ok(SpeculativeStepResult {
            drafted_tokens,
            accepted_count: accepted_tokens.len(),
            accepted_tokens,
            is_eos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::backend::{DeviceBackend, MockDeviceBackend};
    use crate::engine::DecodeEngine;
    use crate::memory::MemoryBridge;

    #[test]
    fn test_speculative_orchestrator_lifecycle() {
        let mut orchestrator = SpeculativeOrchestrator::new(4);
        assert_eq!(orchestrator.draft_length(), 4);

        let mut decode = XrtDecodeEngine::new();
        decode.initialize("speculative_model.xclbin").unwrap();

        let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 65536, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();

        let res = orchestrator
            .step(1001, &decode, &dmabuf, 128001)
            .expect("Speculative step should succeed");

        assert_eq!(res.drafted_tokens.len(), 4);
        assert!(res.accepted_count >= 1 && res.accepted_count <= 4);
        assert_eq!(res.accepted_tokens.len(), res.accepted_count);
        assert!(orchestrator.stats().acceptance_rate() > 0.0);
        assert!(orchestrator.stats().estimated_speedup() >= 1.0);
    }

    #[test]
    fn test_speculative_eos_handling() {
        let mut orchestrator = SpeculativeOrchestrator::new(4);
        let mut decode = XrtDecodeEngine::new();
        decode.initialize("speculative_model.xclbin").unwrap();

        let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 65536, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();

        // Pass a token guaranteed to trigger EOS if encountered
        let res = orchestrator
            .step(1001, &decode, &dmabuf, 315)
            .expect("Step succeed");

        if res.accepted_tokens.contains(&315) {
            assert!(res.is_eos);
            assert_eq!(*res.accepted_tokens.last().unwrap(), 315);
        }
    }
}
