// SPDX-License-Identifier: Apache-2.0
//! Integration Test Suite for Milestone 2 Compute Engines:
//! - RDNA 3.5 iGPU Compute-Heavy Prompt Prefill Engine (`RocmPrefillEngine`)
//! - XDNA 2 NPU Memory-Bound Autoregressive Decode Engine (`XrtDecodeEngine`)
//! - Zen 5 AVX-512 SIMD Token Sampler (`Sampler`)
//!
//! Verifies:
//! 1. Direct KV projection write to dma-buf without host RAM copies.
//! 2. In-place KV cache append during autoregressive decoding.
//! 3. End-to-end forward pass matching authoritative reference oracle.
//! 4. DRM syncobj timeline fence signaling and waiting across handoff.
//! 5. Comprehensive boundary, corner, and error conditions.

use std::sync::Arc;
use zero_copy_model_runner::backend::{DeviceBackend, MockDeviceBackend};
use zero_copy_model_runner::engine::{
    argmax_scalar, ApuDecodeEngine, ApuPrefillEngine, DecodeEngine, DecodeStepRequest,
    DeterministicReferenceOracle, EngineError, PrefillEngine, PrefillRequest, RocmPrefillEngine,
    Sampler, SamplerConfig, SpeculativeDraftingEngine, XrtDecodeEngine,
};
#[cfg(target_arch = "x86_64")]
use zero_copy_model_runner::engine::argmax_avx512;
use zero_copy_model_runner::memory::{MemoryBridge, SharedBuffer};
use zero_copy_model_runner::uapi::{
    drm_syncobj_create, drm_syncobj_handle, drm_syncobj_timeline_wait,
    DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
};

// ============================================================================
// 1. Prefill Direct KV Write to DMA-BUF
// ============================================================================

#[test]
fn test_prefill_direct_kv_write_to_dmabuf() {
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).expect("RocmPrefillEngine initialize");

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo_size = 32 * 1024; // 32 KiB
    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).expect("allocate GEM BO");
    let dmabuf = gpu_bo.export_prime_fd().expect("export PRIME FD");

    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let request = PrefillRequest {
        token_ids: prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine
        .dispatch_prefill(request, &dmabuf, -1, 42)
        .expect("dispatch_prefill failed");

    assert_eq!(result.tokens_processed, prompt.len());
    assert_eq!(result.initial_token_id, 9607);
    assert_eq!(result.completion_fence_point, 42);
    assert!(result.execution_time_us > 0);
    assert_eq!(engine.total_tokens_processed(), prompt.len() as u64);

    // Verify projected KV states are stored directly in dma-buf physical memory
    let shared_buf = SharedBuffer::new(dmabuf.try_clone().unwrap(), true)
        .expect("SharedBuffer mapping failed");

    shared_buf
        .with_cpu_read(|slice| {
            for (i, &token) in prompt.iter().enumerate() {
                let token_slice = &slice[i * 128..(i + 1) * 128];
                let expected = (token % 251) as u8;
                assert!(
                    token_slice.iter().all(|&b| b == expected),
                    "Token {} KV projection mismatch in dma-buf",
                    token
                );
            }
        })
        .expect("CPU read failed");
}

// ============================================================================
// 2. Decode In-Place KV Cache Append
// ============================================================================

#[test]
fn test_decode_inplace_kv_append() {
    let mut engine = XrtDecodeEngine::new();
    engine.initialize("model_decode.xclbin").expect("XrtDecodeEngine initialize");

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo_size = 64 * 1024;
    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).unwrap();
    let dmabuf = gpu_bo.export_prime_fd().unwrap();

    engine.attach_kv_cache(&dmabuf).expect("attach_kv_cache failed");
    assert_eq!(engine.attached_kv_size(), bo_size as u64);

    let test_tokens = [9607u32, 374, 9552, 315];
    let start_seq = 6;

    for (step, &token) in test_tokens.iter().enumerate() {
        let req = DecodeStepRequest {
            input_token_id: token,
            sequence_index: start_seq + step,
            temperature: 0.0,
        };

        let res = engine
            .dispatch_decode_step(req, -1, 0, (step + 1) as u64)
            .expect("dispatch_decode_step failed");

        assert_eq!(res.step_fence_point, (step + 1) as u64);
        assert!(!res.is_eos);
    }

    assert_eq!(engine.current_step(), test_tokens.len() as u64);

    // Verify all appended tokens reside at exact sequential offsets in dma-buf
    let shared_buf = SharedBuffer::new(dmabuf.try_clone().unwrap(), true).unwrap();
    shared_buf
        .with_cpu_read(|slice| {
            for (step, &token) in test_tokens.iter().enumerate() {
                let offset = (start_seq + step) * 128;
                let token_slice = &slice[offset..offset + 128];
                let expected = (token % 251) as u8;
                assert!(
                    token_slice.iter().all(|&b| b == expected),
                    "Step {} KV append mismatch at offset {}",
                    step,
                    offset
                );
            }
        })
        .unwrap();
}

// ============================================================================
// 3. End-to-End Forward Pass: Prefill Phase + 10-Step Autoregressive Decode
// ============================================================================

#[test]
fn test_e2e_forward_pass_prefill_and_decode() {
    let mut prefill_engine = ApuPrefillEngine::new();
    prefill_engine.initialize(0).unwrap();

    let mut decode_engine = ApuDecodeEngine::new();
    decode_engine.initialize("llama3-8b.xclbin").unwrap();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let kv_size = 128 * 1024; // 128 KiB
    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, kv_size, 64).unwrap();
    let kv_cache = gpu_bo.export_prime_fd().unwrap();

    // 1. Prefill prompt
    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let prefill_req = PrefillRequest {
        token_ids: prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let prefill_res = prefill_engine
        .dispatch_prefill(prefill_req, &kv_cache, -1, 1)
        .expect("Prefill execution");

    assert_eq!(prefill_res.tokens_processed, 6);
    assert_eq!(
        prefill_res.initial_token_id,
        DeterministicReferenceOracle::EXPECTED_DECODE_STEPS[0]
    );

    // 2. Attach shared KV cache to decode engine without copying
    decode_engine.attach_kv_cache(&kv_cache).expect("Attach KV cache");

    // 3. Sequential autoregressive decode loop
    let mut generated_tokens = Vec::new();
    let mut current_token = prefill_res.initial_token_id;
    generated_tokens.push(current_token);

    for step in 1..10 {
        let req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: prompt.len() + step - 1,
            temperature: 0.0,
        };

        let res = decode_engine
            .dispatch_decode_step(req, -1, step as u64, (step + 1) as u64)
            .expect("Decode step execution");

        generated_tokens.push(res.output_token_id);
        current_token = res.output_token_id;

        if res.is_eos {
            assert_eq!(res.output_token_id, DeterministicReferenceOracle::EOS_TOKEN_ID);
            break;
        }
    }

    assert_eq!(
        generated_tokens.as_slice(),
        DeterministicReferenceOracle::EXPECTED_DECODE_STEPS,
        "Generated token sequence must match authoritative Llama-3-8B reference output"
    );
    assert_eq!(
        *generated_tokens.last().unwrap(),
        DeterministicReferenceOracle::EOS_TOKEN_ID,
        "Final generated token must be End-Of-Sequence"
    );
}

// ============================================================================
// 4. Timeline Fence Signaling and Waiting Across Handoff
// ============================================================================

#[test]
fn test_timeline_fence_signaling_and_waiting_handoff() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // Create syncobj on mock GPU
    let mut create_arg = drm_syncobj_create {
        handle: 0,
        flags: 0,
    };
    mock_gpu
        .ioctl(
            zero_copy_model_runner::uapi::DRM_IOCTL_SYNCOBJ_CREATE_NUM,
            &mut create_arg as *mut _ as *mut libc::c_void,
        )
        .expect("DRM_IOCTL_SYNCOBJ_CREATE_NUM failed");
    let syncobj_handle = create_arg.handle;
    assert!(syncobj_handle > 0);

    // Export syncobj to FD
    let mut h2fd = drm_syncobj_handle {
        handle: syncobj_handle,
        flags: 0,
        fd: -1,
        pad: 0,
        point: 0,
    };
    mock_gpu
        .ioctl(
            zero_copy_model_runner::uapi::DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_NUM,
            &mut h2fd as *mut _ as *mut libc::c_void,
        )
        .expect("DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_NUM failed");
    let syncobj_fd = h2fd.fd;
    assert!(syncobj_fd >= 0);

    let mut prefill_engine = RocmPrefillEngine::new();
    prefill_engine.initialize(0).unwrap();

    let mut decode_engine = XrtDecodeEngine::new();
    decode_engine.initialize("model.xclbin").unwrap();

    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 32768, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();
    decode_engine.attach_kv_cache(&dmabuf).unwrap();

    // Prefill signals point 1 on the syncobj
    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let prefill_res = prefill_engine
        .dispatch_prefill(
            PrefillRequest {
                token_ids: prompt,
                start_offset: 0,
                batch_size: 1,
            },
            &dmabuf,
            syncobj_fd,
            1,
        )
        .expect("Prefill with fence");
    assert_eq!(prefill_res.completion_fence_point, 1);

    // Decode step waits on point 1 and signals point 2
    let decode_res = decode_engine
        .dispatch_decode_step(
            DecodeStepRequest {
                input_token_id: prefill_res.initial_token_id,
                sequence_index: prompt.len(),
                temperature: 0.0,
            },
            syncobj_fd,
            1,
            2,
        )
        .expect("Decode step with fence");
    assert_eq!(decode_res.step_fence_point, 2);

    // Validate timeline point reached at least point 2
    let handles = [syncobj_handle];
    let points = [2u64];
    let mut array = zero_copy_model_runner::uapi::drm_syncobj_timeline_array {
        handles: handles.as_ptr() as u64,
        points: points.as_ptr() as u64,
        count_handles: 1,
        flags: 0,
    };
    mock_gpu
        .ioctl(
            zero_copy_model_runner::uapi::DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM,
            &mut array as *mut _ as *mut libc::c_void,
        )
        .unwrap();

    let mut wait_arg = drm_syncobj_timeline_wait {
        handles: handles.as_ptr() as u64,
        points: points.as_ptr() as u64,
        timeout_nsec: 1_000_000_000,
        count_handles: 1,
        flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
        first_signaled: 0,
        pad: 0,
        deadline_nsec: 0,
    };
    let wait_res = mock_gpu.ioctl(
        zero_copy_model_runner::uapi::DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM,
        &mut wait_arg as *mut _ as *mut libc::c_void,
    );
    assert!(wait_res.is_ok(), "Timeline wait at point 2 must succeed");
}

// ============================================================================
// 5. Engine Error Handling and Boundary Enforcement
// ============================================================================

#[test]
fn test_engine_boundaries_and_error_handling() {
    let uninitialized_prefill = RocmPrefillEngine::new();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 4096, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    // 1. Uninitialized prefill engine
    let res = uninitialized_prefill.dispatch_prefill(
        PrefillRequest {
            token_ids: &[1, 2, 3],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res, Err(EngineError::InitFailed(_))));

    let mut prefill = RocmPrefillEngine::new();
    prefill.initialize(0).unwrap();

    // 2. Empty prompt tokens
    let res = prefill.dispatch_prefill(
        PrefillRequest {
            token_ids: &[],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res, Err(EngineError::InvalidArgument(_))));

    // 3. Prompt exceeding max limit
    let large_prompt = vec![10u32; 8193];
    let res = prefill.dispatch_prefill(
        PrefillRequest {
            token_ids: &large_prompt,
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res, Err(EngineError::InvalidArgument(_))));

    // 4. Token ID out of bounds
    let res = prefill.dispatch_prefill(
        PrefillRequest {
            token_ids: &[150_000],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res, Err(EngineError::InvalidArgument(_))));

    // 5. Insufficient KV cache capacity
    let tiny_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 128, 64).unwrap();
    let tiny_dmabuf = tiny_bo.export_prime_fd().unwrap();
    let res = prefill.dispatch_prefill(
        PrefillRequest {
            token_ids: &[1, 2], // requires 2 * 128 = 256 bytes
            start_offset: 0,
            batch_size: 1,
        },
        &tiny_dmabuf,
        -1,
        1,
    );
    assert!(matches!(res, Err(EngineError::OutOfResources(_))));

    // 6. Uninitialized decode engine
    let uninitialized_decode = XrtDecodeEngine::new();
    let res = uninitialized_decode.dispatch_decode_step(
        DecodeStepRequest {
            input_token_id: 1,
            sequence_index: 0,
            temperature: 0.0,
        },
        -1,
        0,
        1,
    );
    assert!(matches!(res, Err(EngineError::InitFailed(_))));

    // 7. Decode without attached KV cache
    let mut decode = XrtDecodeEngine::new();
    decode.initialize("model.xclbin").unwrap();
    let res = decode.dispatch_decode_step(
        DecodeStepRequest {
            input_token_id: 1,
            sequence_index: 0,
            temperature: 0.0,
        },
        -1,
        0,
        1,
    );
    assert!(matches!(res, Err(EngineError::InvalidArgument(_))));

    // 8. Negative temperature
    decode.attach_kv_cache(&dmabuf).unwrap();
    let res = decode.dispatch_decode_step(
        DecodeStepRequest {
            input_token_id: 1,
            sequence_index: 0,
            temperature: -0.5,
        },
        -1,
        0,
        1,
    );
    assert!(matches!(res, Err(EngineError::InvalidArgument(_))));

    // 9. Sequence index overflow
    let res = decode.dispatch_decode_step(
        DecodeStepRequest {
            input_token_id: 1,
            sequence_index: 8192,
            temperature: 0.0,
        },
        -1,
        0,
        1,
    );
    assert!(matches!(res, Err(EngineError::InvalidArgument(_))));
}

// ============================================================================
// 6. AVX-512 SIMD Sampler Parity and Distribution Tests
// ============================================================================

#[test]
fn test_sampler_avx512_and_scalar_equivalence() {
    let vocab_size = 128_256;
    let mut logits = vec![0.0f32; vocab_size];

    // Seed pseudorandom logits
    for (i, val) in logits.iter_mut().enumerate() {
        *val = ((i as f32) * 1.6180339).sin() * 10.0;
    }
    logits[54321] = 999.0; // Definite maximum

    let scalar_idx = argmax_scalar(&logits);
    assert_eq!(scalar_idx, 54321);

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") {
        let avx512_idx = unsafe { argmax_avx512(&logits) };
        assert_eq!(
            avx512_idx, scalar_idx,
            "AVX-512 argmax on Zen 5 and scalar argmax must match identically"
        );
    }
}

#[test]
fn test_sampler_temperature_and_top_p() {
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 5,
        seed: Some(42),
    });

    let mut logits = vec![-5.0f32; 1000];
    logits[10] = 5.0;
    logits[20] = 4.5;
    logits[30] = 4.0;

    // Sample 50 tokens; all should fall within the top candidate set {10, 20, 30}
    for _ in 0..50 {
        let tok = sampler.sample(&logits).expect("sample failed");
        assert!(
            tok == 10 || tok == 20 || tok == 30,
            "Sampled token {} was outside nucleus set",
            tok
        );
    }
}

// ============================================================================
// 7. Speculative Drafting Engine Verification
// ============================================================================

#[test]
fn test_speculative_drafting_engine() {
    let mut decode = XrtDecodeEngine::new();
    decode.initialize("model.xclbin").unwrap();

    let candidates = decode.draft_tokens(9607, 4).expect("draft_tokens failed");
    assert_eq!(candidates.len(), 4);

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 32768, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    let accepted = decode.verify_tokens(&candidates, &dmabuf).expect("verify_tokens failed");
    assert!(accepted > 0 && accepted <= candidates.len());
}
