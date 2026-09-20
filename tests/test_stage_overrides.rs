// SPDX-License-Identifier: Apache-2.0
//! Unit tests verifying APU stage override dispatch (CPU, GPU, NPU) for BiLLM models.

use zero_copy_model_runner::backend::open_or_mock;
use zero_copy_model_runner::engine::{
    CpuWorkerEngine, DecodeEngine, DecodeStepRequest, PrefillEngine, PrefillRequest,
    RocmPrefillEngine, XrtDecodeEngine,
};

#[test]
fn test_stage_routing_engines_initialization() {
    let backend = open_or_mock("/dev/dri/renderD128").expect("Open backend");

    // 1. GPU Prefill Engine
    let mut rocm_prefill = RocmPrefillEngine::new_with_backend(backend.clone());
    assert!(rocm_prefill.initialize(0).is_ok());

    // 2. NPU Decode Engine
    let mut xrt_decode = XrtDecodeEngine::new_with_backend(backend.clone());
    assert!(xrt_decode.initialize("mock.xclbin").is_ok());

    // 3. CPU Worker Engine (handles both prefill and decode overrides)
    let mut cpu_worker = CpuWorkerEngine::new();
    assert!(<CpuWorkerEngine as PrefillEngine>::initialize(&mut cpu_worker, 0).is_ok());
    assert!(<CpuWorkerEngine as DecodeEngine>::initialize(&mut cpu_worker, "cpu").is_ok());
}

#[test]
fn test_cpu_prefill_and_decode_override_execution() {
    let mut cpu_worker = CpuWorkerEngine::new();
    assert!(<CpuWorkerEngine as PrefillEngine>::initialize(&mut cpu_worker, 0).is_ok());

    // Test Prefill on CPU
    let prompt_tokens = vec![1u32, 2, 3, 4];
    let req = PrefillRequest {
        token_ids: &prompt_tokens,
        start_offset: 0,
        batch_size: 1,
    };
    let dma_handle = unsafe { zero_copy_model_runner::memory::DmaBufHandle::from_raw_fd_unchecked(-1, 4096) };
    let prefill_res = cpu_worker.dispatch_prefill(req, &dma_handle, -1, 0).expect("CPU prefill");
    assert_eq!(prefill_res.tokens_processed, 4);

    // Test Decode step on CPU
    let dec_req = DecodeStepRequest {
        input_token_id: prefill_res.initial_token_id,
        sequence_index: 4,
        temperature: 0.7,
    };
    let dec_res = cpu_worker.dispatch_decode_step(dec_req, -1, 0, 0).expect("CPU decode");
    assert!(dec_res.output_token_id > 0);
}
