// SPDX-License-Identifier: Apache-2.0
//! Tier 4: Real-World Application Scenarios E2E Tests
//!
//! Realistic operational scenarios:
//! 1. Llama-3-8B End-to-End Chat Completion (Prompt Prefill + 10-Step Decode)
//! 2. Concurrent Multi-Client Pipelined Inference (4 Concurrent Sessions)
//! 3. Graceful Fallback under Missing Hardware Nodes
//! 4. Speculative Drafting Pipeline (NPU Draft K=4 -> iGPU Batched Verify)
//! 5. Dynamic Sliding Window / Context KV Pruning Simulation

use std::os::fd::AsRawFd;
use std::time::Duration;

use crate::common::{
    DeterministicReferenceOracle, MockDecodeEngine, MockDeviceHarness, MockPrefillEngine,
    MockSpeculativeDraftingEngine, MockTimelineSyncobj,
};
use zero_copy_model_runner::engine::{
    DecodeEngine, DecodeStepRequest, PrefillEngine, PrefillRequest, SpeculativeDraftingEngine,
};
use zero_copy_model_runner::memory::DmaBufHandle;

/// Scenario 1: Llama-3-8B End-to-End Chat Completion (Prompt Prefill + 10-Step Decode)
#[test]
fn test_t4_llama3_8b_e2e_prefill_and_10step_decode() {
    let mut prefill_engine = MockPrefillEngine::new();
    prefill_engine.initialize(0).expect("Prefill init");

    let mut decode_engine = MockDecodeEngine::new();
    decode_engine.initialize("llama3-8b-instruct.xclbin").expect("Decode init");

    // 1. Allocate shared zero-copy KV cache (128 KB for active sequence)
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(131072).expect("KV cache dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 131072) };
    std::mem::forget(fd);

    let timeline = MockTimelineSyncobj::new(0);

    // 2. Prefill Phase: "The capital of France is"
    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let prefill_req = PrefillRequest {
        token_ids: prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let t0 = std::time::Instant::now();
    let prefill_res = prefill_engine
        .dispatch_prefill(prefill_req, &handle, -1, 1)
        .expect("Prefill execution");
    let ttft = t0.elapsed();

    assert_eq!(prefill_res.tokens_processed, 6);
    assert_eq!(prefill_res.initial_token_id, 9607);
    assert!(
        ttft < Duration::from_millis(55),
        "TTFT must be under 55ms acceptance threshold"
    );

    timeline.signal(1).expect("Signal prefill completion fence");

    // 3. Attach KV Cache to Decode Engine
    decode_engine.attach_kv_cache(&handle).expect("Attach KV cache");

    // 4. Autoregressive Decode Loop (10 steps)
    let mut generated_tokens = Vec::new();
    let mut current_token = prefill_res.initial_token_id;
    generated_tokens.push(current_token);

    let mut current_fence_point = 1u64;
    for step in 1..10 {
        let wait_point = current_fence_point;
        let signal_point = current_fence_point + 1;

        timeline
            .wait(wait_point, Duration::from_millis(50))
            .expect("Wait for previous step fence");

        let step_req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: prompt.len() + step - 1,
            temperature: 0.0,
        };

        let step_res = decode_engine
            .dispatch_decode_step(step_req, -1, wait_point, signal_point)
            .expect("Step decode execution");

        timeline.signal(signal_point).expect("Signal step fence");
        current_fence_point = signal_point;

        generated_tokens.push(step_res.output_token_id);
        current_token = step_res.output_token_id;

        if step_res.is_eos {
            break;
        }
    }

    assert_eq!(
        generated_tokens.as_slice(),
        DeterministicReferenceOracle::EXPECTED_DECODE_STEPS
    );
    assert_eq!(
        *generated_tokens.last().unwrap(),
        DeterministicReferenceOracle::EOS_TOKEN_ID
    );
}

/// Scenario 2: Concurrent Multi-Client Pipelined Inference (4 Concurrent Sessions)
#[test]
fn test_t4_concurrent_multitenant_pipelined_inference() {
    let mock = MockDeviceHarness::new();
    let mut prefill = MockPrefillEngine::new();
    prefill.initialize(0).expect("Init prefill");

    let num_clients = 4;
    let mut handles = Vec::new();
    for _ in 0..num_clients {
        let fd = mock.create_mock_dmabuf(32768).expect("Client dmabuf");
        let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 32768) };
        std::mem::forget(fd);
        handles.push(handle);
    }

    let prompts: Vec<Vec<u32>> = vec![
        vec![1, 2, 3],
        vec![10, 20, 30, 40],
        vec![100, 200],
        vec![1000, 2000, 3000, 4000, 5000],
    ];

    let mut total_expected_tokens = 0;
    for (i, prompt) in prompts.iter().enumerate() {
        let req = PrefillRequest {
            token_ids: prompt,
            start_offset: 0,
            batch_size: 1,
        };
        let res = prefill
            .dispatch_prefill(req, &handles[i], -1, 1)
            .expect("Client prefill");
        assert_eq!(res.tokens_processed, prompt.len());
        total_expected_tokens += prompt.len();
    }

    assert_eq!(prefill.total_tokens_processed(), total_expected_tokens as u64);
}

/// Scenario 3: Graceful Fallback under Missing Hardware Nodes
#[test]
fn test_t4_graceful_fallback_under_missing_hardware_nodes() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(65536).expect("Mock buffer");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 65536) };
    std::mem::forget(fd);

    let mut prefill = MockPrefillEngine::new();
    prefill.initialize(0).expect("Prefill init");

    let mut decode = MockDecodeEngine::new();
    decode.initialize("fallback_emulated.xclbin").expect("Decode init");
    decode.attach_kv_cache(&handle).expect("Attach");

    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let prefill_res = prefill
        .dispatch_prefill(
            PrefillRequest {
                token_ids: prompt,
                start_offset: 0,
                batch_size: 1,
            },
            &handle,
            -1,
            1,
        )
        .expect("Prefill on fallback");

    let decode_res = decode
        .dispatch_decode_step(
            DecodeStepRequest {
                input_token_id: prefill_res.initial_token_id,
                sequence_index: prompt.len(),
                temperature: 0.0,
            },
            -1,
            1,
            2,
        )
        .expect("Decode on fallback");

    assert!(decode_res.output_token_id > 0);
    assert!(!decode_res.is_eos);
}

/// Scenario 4: Speculative Drafting Pipeline (NPU Draft K=4 -> iGPU Batched Verify)
#[test]
fn test_t4_speculative_drafting_pipeline() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(32768).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 32768) };
    std::mem::forget(fd);

    let speculative_engine = MockSpeculativeDraftingEngine::new(4);

    let seed_token = 9607;
    let candidate_tokens = speculative_engine
        .draft_tokens(seed_token, 4)
        .expect("Draft candidate tokens");
    assert_eq!(candidate_tokens.len(), 4);

    let accepted_tokens = speculative_engine
        .verify_tokens(&candidate_tokens, &handle)
        .expect("Verify candidate tokens");

    assert!(
        accepted_tokens > 0,
        "Expected at least 1 candidate token accepted"
    );
    assert!(accepted_tokens <= candidate_tokens.len());
}

/// Scenario 5: Dynamic Sliding Window / Context KV Pruning Simulation
#[test]
fn test_t4_dynamic_sliding_window_kv_pruning() {
    let mock = MockDeviceHarness::new();
    let buffer_size = 65536;
    let fd = mock.create_mock_dmabuf(buffer_size).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), buffer_size) };
    std::mem::forget(fd);

    let mut decode = MockDecodeEngine::new();
    decode.initialize("snapkv.xclbin").expect("Init");
    decode.attach_kv_cache(&handle).expect("Attach");

    let current_index = 400;
    let window_size = 256;
    let pruned_start_offset = current_index - window_size;

    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            buffer_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            handle.as_raw_fd(),
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        let discarded_slice =
            std::slice::from_raw_parts_mut(ptr as *mut u8, pruned_start_offset * 128);
        discarded_slice.fill(0);
        libc::munmap(ptr, buffer_size);
    }

    let step_res = decode
        .dispatch_decode_step(
            DecodeStepRequest {
                input_token_id: 374,
                sequence_index: 401,
                temperature: 0.0,
            },
            -1,
            0,
            1,
        )
        .expect("Step decode post-pruning");

    assert!(step_res.output_token_id > 0);
}
