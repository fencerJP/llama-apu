// SPDX-License-Identifier: Apache-2.0
//! Accelerator Engine Abstractions for Compute-Bound Prefill and Memory-Bound Decode.

use std::os::fd::RawFd;
use thiserror::Error;

use crate::memory::DmaBufHandle;

pub mod cpu_worker;
pub mod rocm_prefill;
pub mod sampler;
pub mod speculative;
pub mod transformer;
pub mod triforce;
pub mod xrt_decode;

pub use cpu_worker::CpuWorkerEngine;
pub use rocm_prefill::{ApuPrefillEngine, DeterministicReferenceOracle, RocmPrefillEngine};
pub use sampler::{argmax_scalar, FastRng, Sampler, SamplerConfig};
#[cfg(target_arch = "x86_64")]
pub use sampler::argmax_avx512;
pub use speculative::{SpeculativeOrchestrator, SpeculativeStats, SpeculativeStepResult};
pub use transformer::{TransformerContext};
pub use triforce::{TriForceCacheConfig, TriForceSpeculativeEngine, TriForceStats, TriForceStepResult};
pub use xrt_decode::{ApuDecodeEngine, XrtDecodeEngine};

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Device context initialization failed: {0}")]
    InitFailed(String),
    #[error("Kernel launch failed on accelerator: {0}")]
    LaunchFailed(String),
    #[error("Synchronization fence timeout after {timeout_ms} ms")]
    FenceTimeout { timeout_ms: u64 },
    #[error("Out of device memory or AIE tile capacity: {0}")]
    OutOfResources(String),
    #[error("Invalid prompt or tensor dimensionality: {0}")]
    InvalidArgument(String),
}

/// Metadata describing an active prompt prefill request.
#[derive(Debug, Clone)]
pub struct PrefillRequest<'a> {
    /// Token IDs representing the input prompt sequence.
    pub token_ids: &'a [u32],
    /// Target sequence offset where prefill KV embeddings start.
    pub start_offset: usize,
    /// Batch size (typically 1 for single-stream interactive queries).
    pub batch_size: usize,
}

/// Execution outcome of the prompt prefill phase.
#[derive(Debug, Clone)]
pub struct PrefillResult {
    /// Total tokens processed during prefill.
    pub tokens_processed: usize,
    /// Initial sampled output token ID (T_0) emitted from prompt logits.
    pub initial_token_id: u32,
    /// Hardware execution time in microseconds.
    pub execution_time_us: u64,
    /// Timeline fence point signaled upon completion.
    pub completion_fence_point: u64,
}

/// Hardware engine responsible for compute-heavy Prompt Prefill (RDNA 3.5 iGPU / ROCm).
pub trait PrefillEngine: Send + Sync {
    /// Initialize the iGPU execution context, load prefill GEMM kernels, and bind rings.
    fn initialize(&mut self, device_index: u32) -> Result<(), EngineError>;

    /// Submit a prefill forward pass over `request.token_ids`.
    ///
    /// Key and Value attention tensors are written directly into `kv_cache` via `dma-buf`.
    /// The accelerator signals `signal_timeline_point` on the provided DRM syncobj.
    fn dispatch_prefill(
        &self,
        request: PrefillRequest,
        kv_cache: &DmaBufHandle,
        syncobj_fd: RawFd,
        signal_timeline_point: u64,
    ) -> Result<PrefillResult, EngineError>;

    /// Query peak compute capacity in FP16 TFLOPs.
    fn peak_compute_tflops(&self) -> f32;
}

/// Metadata describing a single autoregressive decode step.
#[derive(Debug, Clone)]
pub struct DecodeStepRequest {
    /// Token ID emitted in the previous step.
    pub input_token_id: u32,
    /// Position index in the autoregressive sequence (starts at prefill_len).
    pub sequence_index: usize,
    /// Temperature scaling for multinomial sampling (0.0 = argmax greedy).
    pub temperature: f32,
}

/// Execution outcome of a single autoregressive decode step.
#[derive(Debug, Clone)]
pub struct DecodeStepResult {
    /// Sampled output token ID.
    pub output_token_id: u32,
    /// Whether the model produced an End-Of-Sequence (EOS) token.
    pub is_eos: bool,
    /// Tile execution latency in microseconds.
    pub tile_latency_us: u64,
    /// Timeline fence point signaled after KV update and logit projection.
    pub step_fence_point: u64,
}

/// Hardware engine responsible for memory-bound Autoregressive Decoding (XDNA 2 NPU / XRT).
pub trait DecodeEngine: Send + Sync {
    /// Initialize the XDNA AIE2P tile array, configure spatial tile DMAs, and load weights.
    fn initialize(&mut self, xclbin_path: &str) -> Result<(), EngineError>;

    /// Attach the shared zero-copy KV cache (`dma-buf`) into the NPU address space.
    fn attach_kv_cache(&mut self, kv_cache: &DmaBufHandle) -> Result<(), EngineError>;

    /// Execute one autoregressive token generation forward pass.
    ///
    /// Waits for `wait_timeline_point` on `wait_syncobj_fd` (e.g. from prefill completion)
    /// before beginning the first step, then signals `signal_timeline_point`.
    fn dispatch_decode_step(
        &self,
        request: DecodeStepRequest,
        wait_syncobj_fd: RawFd,
        wait_timeline_point: u64,
        signal_timeline_point: u64,
    ) -> Result<DecodeStepResult, EngineError>;

    /// Query total AIE2P compute tiles active.
    fn active_tiles(&self) -> u32;

    /// Query NPU INT4 / BlockFP16 peak TOPS.
    fn peak_npu_tops(&self) -> f32;
}

/// Speculative Drafting Engine for Strix Halo / high-performance configurations.
pub trait SpeculativeDraftingEngine: Send + Sync {
    /// Generate $K$ speculative candidate tokens using the energy-efficient NPU.
    fn draft_tokens(
        &self,
        seed_token_id: u32,
        draft_length: usize,
    ) -> Result<Vec<u32>, EngineError>;

    /// Verify all $K$ tokens concurrently in a single batched pass on the iGPU.
    fn verify_tokens(
        &self,
        candidate_tokens: &[u32],
        kv_cache: &DmaBufHandle,
    ) -> Result<usize, EngineError>;
}
