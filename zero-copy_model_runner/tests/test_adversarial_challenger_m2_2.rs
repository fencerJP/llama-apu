// SPDX-License-Identifier: Apache-2.0
//! Adversarial Challenge Test Suite: Decode Engine & Timeline Sync Stress Testing.
//!
//! Conducted by Challenger 2 (challenger_m2_2) for Milestone 2.
//!
//! Objectives:
//! 1. Multi-step decode loops (20+ steps, 32 steps, 128 steps) appending KV states in-place.
//! 2. Timeline fence synchronization across prefill -> decode steps (asserting no host CPU busy-spinning).
//! 3. EOS detection and boundary rejections.
//! 4. Concurrency, multi-tenant isolation, and resource leak absence (zero FD/VMA leaks).

use std::collections::BTreeSet;
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use zero_copy_model_runner::backend::{DeviceBackend, MockDeviceBackend};
use zero_copy_model_runner::engine::{
    DecodeEngine, DecodeStepRequest, DeterministicReferenceOracle, EngineError, PrefillEngine,
    PrefillRequest, RocmPrefillEngine, SpeculativeDraftingEngine, XrtDecodeEngine,
};
use zero_copy_model_runner::memory::{MemoryBridge, SharedBuffer};
use zero_copy_model_runner::uapi::{
    drm_syncobj_create, drm_syncobj_timeline_array, drm_syncobj_timeline_wait,
    DRM_IOCTL_SYNCOBJ_CREATE_NUM, DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM,
    DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
};

/// Global test mutex to serialize tests inspecting `/proc/self/fd` and `/proc/self/maps`.
static PROC_INSPECTION_MUTEX: Mutex<()> = Mutex::new(());

fn acquire_lock() -> std::sync::MutexGuard<'static, ()> {
    PROC_INSPECTION_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Query currently open file descriptors from `/proc/self/fd`.
fn get_open_fds() -> BTreeSet<i32> {
    let mut fds = BTreeSet::new();
    if let Ok(entries) = fs::read_dir("/proc/self/fd") {
        for entry in entries.flatten() {
            if let Ok(num) = entry.file_name().to_string_lossy().parse::<i32>() {
                fds.insert(num);
            }
        }
    }
    fds
}

/// Parse virtual memory mapping count from `/proc/self/maps`.
fn count_accelerator_mappings() -> usize {
    let content = fs::read_to_string("/proc/self/maps").unwrap_or_default();
    content
        .lines()
        .filter(|line| {
            line.contains("memfd:mock_amdgpu")
                || line.contains("dma_buf")
                || line.contains("renderD128")
        })
        .count()
}

/// Measure thread CPU time (user + system) in microseconds using `libc::getrusage`.
fn get_thread_cpu_time_us() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_THREAD, usage.as_mut_ptr()) };
    if ret == 0 {
        let usage = unsafe { usage.assume_init() };
        let utime_us =
            (usage.ru_utime.tv_sec as u64) * 1_000_000 + (usage.ru_utime.tv_usec as u64);
        let stime_us =
            (usage.ru_stime.tv_sec as u64) * 1_000_000 + (usage.ru_stime.tv_usec as u64);
        utime_us + stime_us
    } else {
        0
    }
}

// ============================================================================
// 1. Multi-Step Decode Loop: 32 Steps In-Place KV Append
// ============================================================================

#[test]
fn test_adversarial_m2_decode_multistep_32_inplace_kv_append() {
    let _lock = acquire_lock();
    let mut prefill_engine = RocmPrefillEngine::new();
    prefill_engine.initialize(0).expect("Prefill init");

    let mut decode_engine = XrtDecodeEngine::new();
    decode_engine
        .initialize("model_decode.xclbin")
        .expect("Decode init");

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo_size = 64 * 1024; // 64 KiB fits 512 tokens
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).expect("GEM alloc");
    let dmabuf = bo.export_prime_fd().expect("export PRIME");

    // 1. Prefill reference prompt (6 tokens)
    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let prefill_res = prefill_engine
        .dispatch_prefill(
            PrefillRequest {
                token_ids: prompt,
                start_offset: 0,
                batch_size: 1,
            },
            &dmabuf,
            -1,
            1,
        )
        .expect("Prefill dispatch");

    assert_eq!(prefill_res.tokens_processed, prompt.len());
    assert_eq!(prefill_res.initial_token_id, 9607);

    // 2. Attach shared KV cache to decode engine without copying
    decode_engine
        .attach_kv_cache(&dmabuf)
        .expect("Attach KV cache");
    assert_eq!(decode_engine.attached_kv_size(), bo_size as u64);

    // 3. Execute 32-step autoregressive decode loop
    let total_steps = 32;
    let mut sequence_tokens = Vec::with_capacity(prompt.len() + total_steps);
    sequence_tokens.extend_from_slice(prompt);

    let mut current_token = prefill_res.initial_token_id;

    for step in 0..total_steps {
        let seq_index = prompt.len() + step;
        let req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: seq_index,
            temperature: 0.0,
        };

        let res = decode_engine
            .dispatch_decode_step(req, -1, step as u64, (step + 1) as u64)
            .unwrap_or_else(|e| panic!("Decode step {} failed: {:?}", step, e));

        assert_eq!(
            res.step_fence_point,
            (step + 1) as u64,
            "Step fence point mismatch"
        );
        sequence_tokens.push(current_token);
        current_token = res.output_token_id;
    }

    assert_eq!(
        decode_engine.current_step(),
        total_steps as u64,
        "Engine current_step counter must equal 32"
    );

    // 4. In-place memory inspection: verify all 38 token states directly in DMA-BUF
    let shared_buf = SharedBuffer::new(dmabuf.try_clone().unwrap(), true)
        .expect("SharedBuffer mapping failed");

    shared_buf
        .with_cpu_read(|slice| {
            // Verify prompt tokens (0..6)
            for (i, &token) in prompt.iter().enumerate() {
                let offset = i * 128;
                let token_slice = &slice[offset..offset + 128];
                let expected = (token % 251) as u8;
                assert!(
                    token_slice.iter().all(|&b| b == expected),
                    "Prompt token {} at offset {} mismatch",
                    i,
                    offset
                );
            }

            // Verify 32 appended decode steps (offsets 6*128 .. 38*128)
            for step in 0..total_steps {
                let seq_index = prompt.len() + step;
                let offset = seq_index * 128;
                let token_slice = &slice[offset..offset + 128];
                let input_tok = sequence_tokens[seq_index];
                let expected = (input_tok % 251) as u8;
                assert!(
                    token_slice.iter().all(|&b| b == expected),
                    "Decode step {} (seq {}) at offset {} mismatch: expected {}, got {:?}",
                    step,
                    seq_index,
                    offset,
                    expected,
                    &token_slice[0..4]
                );
            }

            // Verify subsequent memory remains uncorrupted / zero-initialized
            let unwritten_offset = (prompt.len() + total_steps) * 128;
            let unwritten_slice = &slice[unwritten_offset..unwritten_offset + 128];
            assert!(
                unwritten_slice.iter().all(|&b| b == 0),
                "Memory beyond step 32 was unexpectedly overwritten"
            );
        })
        .expect("CPU read failed");
}

// ============================================================================
// 2. Massive Multi-Step Decode Loop: 128 Steps Sustained Stress
// ============================================================================

#[test]
fn test_adversarial_m2_decode_long_loop_128_steps_stress() {
    let _lock = acquire_lock();
    let mut decode_engine = XrtDecodeEngine::new();
    decode_engine
        .initialize("massive_model.xclbin")
        .expect("Decode init");

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo_size = 32 * 1024; // 32 KiB fits 256 tokens
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).expect("GEM alloc");
    let dmabuf = bo.export_prime_fd().expect("export PRIME");

    decode_engine
        .attach_kv_cache(&dmabuf)
        .expect("Attach KV cache");

    let steps = 128;
    let mut token_history = Vec::with_capacity(steps);
    let mut current_tok = 1337u32;

    for step in 0..steps {
        token_history.push(current_tok);
        let req = DecodeStepRequest {
            input_token_id: current_tok,
            sequence_index: step,
            temperature: 0.0,
        };

        let res = decode_engine
            .dispatch_decode_step(req, -1, 0, (step + 1) as u64)
            .unwrap_or_else(|e| panic!("Massive decode step {} failed: {:?}", step, e));

        assert_eq!(res.step_fence_point, (step + 1) as u64);
        current_tok = res.output_token_id;
    }

    assert_eq!(decode_engine.current_step(), steps as u64);

    // Verify all 128 tokens reside at exact sequential 128-byte offsets
    let shared_buf = SharedBuffer::new(dmabuf.try_clone().unwrap(), true).unwrap();
    shared_buf
        .with_cpu_read(|slice| {
            for (step, &token) in token_history.iter().enumerate() {
                let offset = step * 128;
                let token_slice = &slice[offset..offset + 128];
                let expected = (token % 251) as u8;
                assert!(
                    token_slice.iter().all(|&b| b == expected),
                    "Step {} KV append corrupted at offset {}",
                    step,
                    offset
                );
            }
        })
        .unwrap();
}

// ============================================================================
// 3. Timeline Fence Synchronization: Asserting No Host CPU Busy-Spinning
// ============================================================================

#[test]
fn test_adversarial_m2_timeline_fence_no_cpu_busy_spinning() {
    let _lock = acquire_lock();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // Create timeline syncobj on mock GPU
    let mut create_arg = drm_syncobj_create {
        handle: 0,
        flags: 0,
    };
    mock_gpu
        .ioctl(
            DRM_IOCTL_SYNCOBJ_CREATE_NUM,
            &mut create_arg as *mut _ as *mut libc::c_void,
        )
        .expect("Create syncobj");
    let syncobj_handle = create_arg.handle;
    assert!(syncobj_handle > 0);

    // Case A: Wait on a future fence point (simulating cross-device prefill -> decode handoff)
    let target_point = 1u64;
    let sleep_delay_ms = 60u64;

    let gpu_producer = Arc::clone(&mock_gpu);
    let producer_handle = thread::spawn(move || {
        // Simulate compute execution latency on iGPU
        thread::sleep(Duration::from_millis(sleep_delay_ms));

        // Signal completion timeline point 1
        let handles = [syncobj_handle];
        let points = [target_point];
        let mut array = drm_syncobj_timeline_array {
            handles: handles.as_ptr() as u64,
            points: points.as_ptr() as u64,
            count_handles: 1,
            flags: 0,
        };
        gpu_producer
            .ioctl(
                DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM,
                &mut array as *mut _ as *mut libc::c_void,
            )
            .expect("Producer signal");
    });

    // Consumer thread: waits on timeline fence and measures CPU time
    let gpu_consumer = Arc::clone(&mock_gpu);
    let consumer_handle = thread::spawn(move || {
        let cpu_start_us = get_thread_cpu_time_us();
        let wall_start = Instant::now();

        let handles = [syncobj_handle];
        let points = [target_point];
        let mut wait_arg = drm_syncobj_timeline_wait {
            handles: handles.as_ptr() as u64,
            points: points.as_ptr() as u64,
            timeout_nsec: 500_000_000, // 500ms timeout
            count_handles: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: 0,
        };

        let wait_res = gpu_consumer.ioctl(
            DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM,
            &mut wait_arg as *mut _ as *mut libc::c_void,
        );

        let wall_elapsed = wall_start.elapsed();
        let cpu_end_us = get_thread_cpu_time_us();
        let cpu_consumed_us = cpu_end_us.saturating_sub(cpu_start_us);

        (wait_res, wall_elapsed, cpu_consumed_us)
    });

    producer_handle.join().expect("Producer join");
    let (wait_res, wall_elapsed, cpu_consumed_us) = consumer_handle.join().expect("Consumer join");

    assert!(wait_res.is_ok(), "Fence wait should succeed");
    assert!(
        wall_elapsed >= Duration::from_millis(50),
        "Wall time ({:?}) must reflect producer delay",
        wall_elapsed
    );

    // CRITICAL EMPIRICAL PROOF:
    // If the thread was busy-spinning, CPU time would equal wall time (~60,000 us).
    // Under proper kernel/condvar suspension, CPU time must be < 5,000 us (< 5ms).
    println!(
        "[EMPIRICAL EVIDENCE] Wall elapsed: {:?}, CPU time consumed: {} us",
        wall_elapsed, cpu_consumed_us
    );
    assert!(
        cpu_consumed_us < 5_000,
        "Host CPU busy-spinning detected! Consumed {} us CPU time over {:?}",
        cpu_consumed_us,
        wall_elapsed
    );

    // Case B: Unsignaled fence timeout expiration must also NOT busy-spin
    let timeout_consumer = thread::spawn(move || {
        let cpu_start_us = get_thread_cpu_time_us();
        let wall_start = Instant::now();

        let handles = [syncobj_handle];
        let points = [999u64]; // Never signaled
        let mut wait_arg = drm_syncobj_timeline_wait {
            handles: handles.as_ptr() as u64,
            points: points.as_ptr() as u64,
            timeout_nsec: 30_000_000, // 30ms timeout
            count_handles: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: 0,
        };

        let wait_res = mock_gpu.ioctl(
            DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM,
            &mut wait_arg as *mut _ as *mut libc::c_void,
        );

        let wall_elapsed = wall_start.elapsed();
        let cpu_end_us = get_thread_cpu_time_us();
        let cpu_consumed_us = cpu_end_us.saturating_sub(cpu_start_us);

        (wait_res, wall_elapsed, cpu_consumed_us)
    });

    let (timeout_res, timeout_wall, timeout_cpu_us) =
        timeout_consumer.join().expect("Timeout consumer join");

    assert_eq!(
        timeout_res.unwrap_err(),
        nix::Error::ETIMEDOUT,
        "Expired fence must return ETIMEDOUT"
    );
    assert!(
        timeout_wall >= Duration::from_millis(25),
        "Timeout wall time ({:?}) must reach requested duration",
        timeout_wall
    );
    assert!(
        timeout_cpu_us < 5_000,
        "Host CPU busy-spinning detected during timeout wait! Consumed {} us CPU time",
        timeout_cpu_us
    );
}

// ============================================================================
// 4. Monotonic Fence Progression Chain Across 25 Decode Iterations
// ============================================================================

#[test]
fn test_adversarial_m2_monotonic_fence_handoff_25_steps() {
    let _lock = acquire_lock();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    let mut create_arg = drm_syncobj_create {
        handle: 0,
        flags: 0,
    };
    mock_gpu
        .ioctl(
            DRM_IOCTL_SYNCOBJ_CREATE_NUM,
            &mut create_arg as *mut _ as *mut libc::c_void,
        )
        .unwrap();
    let syncobj_handle = create_arg.handle;

    let total_steps = 25;
    let cpu_start = get_thread_cpu_time_us();

    // Prefill signals point 1
    let handles = [syncobj_handle];
    let points = [1u64];
    let mut array = drm_syncobj_timeline_array {
        handles: handles.as_ptr() as u64,
        points: points.as_ptr() as u64,
        count_handles: 1,
        flags: 0,
    };
    mock_gpu
        .ioctl(
            DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM,
            &mut array as *mut _ as *mut libc::c_void,
        )
        .unwrap();

    // 25 sequential decode handoffs
    for step in 1..=total_steps {
        let wait_point = step as u64;
        let signal_point = (step + 1) as u64;

        // Wait on step
        let wait_points = [wait_point];
        let mut wait_arg = drm_syncobj_timeline_wait {
            handles: handles.as_ptr() as u64,
            points: wait_points.as_ptr() as u64,
            timeout_nsec: 100_000_000, // 100ms
            count_handles: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: 0,
        };
        mock_gpu
            .ioctl(
                DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM,
                &mut wait_arg as *mut _ as *mut libc::c_void,
            )
            .unwrap();

        // Signal next point
        let sig_points = [signal_point];
        let mut sig_array = drm_syncobj_timeline_array {
            handles: handles.as_ptr() as u64,
            points: sig_points.as_ptr() as u64,
            count_handles: 1,
            flags: 0,
        };
        mock_gpu
            .ioctl(
                DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM,
                &mut sig_array as *mut _ as *mut libc::c_void,
            )
            .unwrap();
    }

    let cpu_elapsed = get_thread_cpu_time_us().saturating_sub(cpu_start);
    println!(
        "25 fence transitions completed. Total CPU time: {} us",
        cpu_elapsed
    );

    // Final verification: waiting at point 26 must succeed immediately
    let final_points = [26u64];
    let mut final_wait = drm_syncobj_timeline_wait {
        handles: handles.as_ptr() as u64,
        points: final_points.as_ptr() as u64,
        timeout_nsec: 10_000_000,
        count_handles: 1,
        flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
        first_signaled: 0,
        pad: 0,
        deadline_nsec: 0,
    };
    let res = mock_gpu.ioctl(
        DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM,
        &mut final_wait as *mut _ as *mut libc::c_void,
    );
    assert!(res.is_ok(), "Timeline point 26 must be signaled");
}

// ============================================================================
// 5. Concurrent Multi-Thread Timeline Fence Contention (8 Threads)
// ============================================================================

#[test]
fn test_adversarial_m2_concurrent_timeline_fence_stress_8_threads() {
    let _lock = acquire_lock();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    let mut create_arg = drm_syncobj_create {
        handle: 0,
        flags: 0,
    };
    mock_gpu
        .ioctl(
            DRM_IOCTL_SYNCOBJ_CREATE_NUM,
            &mut create_arg as *mut _ as *mut libc::c_void,
        )
        .unwrap();
    let syncobj_handle = create_arg.handle;

    let num_threads = 8;
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let target_point = ((t + 1) * 10) as u64; // 10, 20, 30, ... 80
        let gpu = Arc::clone(&mock_gpu);

        let h = thread::spawn(move || {
            let h_arr = [syncobj_handle];
            let p_arr = [target_point];
            let mut wait_arg = drm_syncobj_timeline_wait {
                handles: h_arr.as_ptr() as u64,
                points: p_arr.as_ptr() as u64,
                timeout_nsec: 2_000_000_000, // 2s timeout
                count_handles: 1,
                flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
                first_signaled: 0,
                pad: 0,
                deadline_nsec: 0,
            };
            let ret = gpu.ioctl(
                DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT_NUM,
                &mut wait_arg as *mut _ as *mut libc::c_void,
            );
            assert!(
                ret.is_ok(),
                "Thread {} failed waiting for point {}",
                t,
                target_point
            );
        });
        handles.push(h);
    }

    // Driver thread pulses through points in order
    for t in 0..num_threads {
        thread::sleep(Duration::from_millis(15));
        let point = ((t + 1) * 10) as u64;
        let h_arr = [syncobj_handle];
        let p_arr = [point];
        let mut sig_arr = drm_syncobj_timeline_array {
            handles: h_arr.as_ptr() as u64,
            points: p_arr.as_ptr() as u64,
            count_handles: 1,
            flags: 0,
        };
        mock_gpu
            .ioctl(
                DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL_NUM,
                &mut sig_arr as *mut _ as *mut libc::c_void,
            )
            .unwrap();
    }

    for h in handles {
        h.join().expect("Worker thread join failed");
    }
}

// ============================================================================
// 6. EOS Detection and Continuation Semantics
// ============================================================================

#[test]
fn test_adversarial_m2_eos_detection_and_continuation_semantics() {
    let _lock = acquire_lock();
    let mut decode_engine = XrtDecodeEngine::new();
    decode_engine.initialize("llama3.xclbin").unwrap();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 32768, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();
    decode_engine.attach_kv_cache(&dmabuf).unwrap();

    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT; // length 6

    // Run decode steps 1..9 (non-EOS tokens)
    let mut current_token = DeterministicReferenceOracle::EXPECTED_DECODE_STEPS[0];
    for step in 1..10 {
        let req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: prompt.len() + step - 1,
            temperature: 0.0,
        };
        let res = decode_engine
            .dispatch_decode_step(req, -1, 0, step as u64)
            .unwrap();

        current_token = res.output_token_id;
        if step < 9 {
            assert!(
                !res.is_eos,
                "Step {} should NOT be EOS, got token {}",
                step,
                res.output_token_id
            );
        } else {
            // Step 9 (10th token overall) must produce EOS
            assert!(res.is_eos, "Step 9 MUST produce EOS token");
            assert_eq!(
                res.output_token_id,
                DeterministicReferenceOracle::EOS_TOKEN_ID
            );
        }
    }

    // Stress test: continuing generation PAST EOS for 15 additional steps
    for step in 10..25 {
        let req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: prompt.len() + step - 1,
            temperature: 0.0,
        };
        let res = decode_engine
            .dispatch_decode_step(req, -1, 0, (step + 1) as u64)
            .unwrap();

        assert!(
            res.is_eos,
            "Continuation step {} must remain EOS",
            step
        );
        assert_eq!(
            res.output_token_id,
            DeterministicReferenceOracle::EOS_TOKEN_ID
        );
    }

    // Verify all 25 decode steps were recorded in the engine and KV buffer
    assert_eq!(decode_engine.current_step(), 24); // 9 + 15 steps
}

// ============================================================================
// 7. Exact Buffer Boundary and OutOfResources Enforcement
// ============================================================================

#[test]
fn test_adversarial_m2_buffer_capacity_exact_boundary_enforcement() {
    let _lock = acquire_lock();
    let mut decode_engine = XrtDecodeEngine::new();
    decode_engine.initialize("model.xclbin").unwrap();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    // Exactly 20 tokens capacity: 20 * 128 = 2560 bytes
    let bo_size = 20 * 128;
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    decode_engine.attach_kv_cache(&dmabuf).unwrap();

    // Steps 0 through 19 (all 20 slots) must succeed
    for step in 0..20 {
        let req = DecodeStepRequest {
            input_token_id: 100 + (step as u32),
            sequence_index: step,
            temperature: 0.0,
        };
        let res = decode_engine.dispatch_decode_step(req, -1, 0, (step + 1) as u64);
        assert!(res.is_ok(), "Step {} within capacity must succeed", step);
    }

    // Step 20 (offset 2560 + 128 = 2688 > 2560) MUST fail with OutOfResources
    let req_overflow = DecodeStepRequest {
        input_token_id: 999,
        sequence_index: 20,
        temperature: 0.0,
    };
    let err_res = decode_engine.dispatch_decode_step(req_overflow, -1, 0, 21);
    match err_res {
        Err(EngineError::OutOfResources(msg)) => {
            assert!(
                msg.contains("capacity"),
                "Error message should mention capacity: {}",
                msg
            );
        }
        other => panic!("Expected OutOfResources error, got {:?}", other),
    }

    // Verify all 20 valid slots were preserved uncorrupted
    let shared_buf = SharedBuffer::new(dmabuf.try_clone().unwrap(), true).unwrap();
    shared_buf
        .with_cpu_read(|slice| {
            for step in 0..20 {
                let offset = step * 128;
                let token_slice = &slice[offset..offset + 128];
                let expected = ((100 + step as u32) % 251) as u8;
                assert!(token_slice.iter().all(|&b| b == expected));
            }
        })
        .unwrap();
}

// ============================================================================
// 8. Comprehensive Boundary & Corner Case Rejections
// ============================================================================

#[test]
fn test_adversarial_m2_comprehensive_boundary_rejections() {
    let _lock = acquire_lock();
    let mut decode_engine = XrtDecodeEngine::new();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 1024 * 1024 + 128, 64).unwrap(); // > 1MB
    let dmabuf = bo.export_prime_fd().unwrap();

    // 1. Decode before initialization
    let res = decode_engine.dispatch_decode_step(
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

    // 2. Empty XCLBIN path
    assert!(matches!(
        decode_engine.initialize(""),
        Err(EngineError::InvalidArgument(_))
    ));

    // Initialize properly
    decode_engine.initialize("model.xclbin").unwrap();

    // 3. Decode without attached KV cache
    let res = decode_engine.dispatch_decode_step(
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

    decode_engine.attach_kv_cache(&dmabuf).unwrap();

    // 4. Max allowed sequence index (8191) succeeds
    let res_max_valid = decode_engine.dispatch_decode_step(
        DecodeStepRequest {
            input_token_id: 42,
            sequence_index: 8191,
            temperature: 0.0,
        },
        -1,
        0,
        1,
    );
    assert!(res_max_valid.is_ok(), "Sequence index 8191 must be valid");

    // 5. Sequence index overflow (8192 boundary) rejected
    let res_overflow = decode_engine.dispatch_decode_step(
        DecodeStepRequest {
            input_token_id: 42,
            sequence_index: 8192,
            temperature: 0.0,
        },
        -1,
        0,
        2,
    );
    assert!(matches!(res_overflow, Err(EngineError::InvalidArgument(_))));

    // 6. Extreme sequence index rejected
    let res_extreme = decode_engine.dispatch_decode_step(
        DecodeStepRequest {
            input_token_id: 42,
            sequence_index: usize::MAX - 1,
            temperature: 0.0,
        },
        -1,
        0,
        3,
    );
    assert!(matches!(res_extreme, Err(EngineError::InvalidArgument(_))));

    // 7. Negative temperatures rejected
    for neg_temp in [-0.001f32, -1.0, -100.0, f32::NEG_INFINITY] {
        let res_temp = decode_engine.dispatch_decode_step(
            DecodeStepRequest {
                input_token_id: 42,
                sequence_index: 0,
                temperature: neg_temp,
            },
            -1,
            0,
            4,
        );
        assert!(
            matches!(res_temp, Err(EngineError::InvalidArgument(_))),
            "Negative temp {} should be rejected",
            neg_temp
        );
    }

    // 8. Speculative drafting boundary rejections
    assert!(matches!(
        decode_engine.draft_tokens(10, 0),
        Err(EngineError::InvalidArgument(_))
    ));
    assert_eq!(decode_engine.verify_tokens(&[], &dmabuf).unwrap(), 0);

    // 9. Prefill boundary checks
    let mut prefill = RocmPrefillEngine::new();
    prefill.initialize(0).unwrap();

    let invalid_token_prompt = vec![128_257u32]; // > 128,256
    let res_tok = prefill.dispatch_prefill(
        PrefillRequest {
            token_ids: &invalid_token_prompt,
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res_tok, Err(EngineError::InvalidArgument(_))));
}

// ============================================================================
// 9. Multi-Tenant Zero-Copy Isolation Under Concurrent Decoding
// ============================================================================

#[test]
fn test_adversarial_m2_multitenant_concurrent_decode_isolation() {
    let _lock = acquire_lock();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let num_tenants = 4;
    let steps_per_tenant = 15;
    let mut thread_handles = Vec::new();

    for tenant_id in 0..num_tenants {
        let gpu = Arc::clone(&mock_gpu);

        let h = thread::spawn(move || {
            // Allocate tenant's private DMA-BUF
            let bo = MemoryBridge::allocate_gem_bo(&gpu, 32768, 64).unwrap();
            let dmabuf = bo.export_prime_fd().unwrap();

            // Create thread-local decode engine instance bound to tenant buffer
            let mut tenant_engine = XrtDecodeEngine::new();
            tenant_engine.initialize("tenant_engine.xclbin").unwrap();
            tenant_engine.attach_kv_cache(&dmabuf).unwrap();

            let token_base = (tenant_id + 1) * 1000;
            for step in 0..steps_per_tenant {
                let token = (token_base + step) as u32;
                let req = DecodeStepRequest {
                    input_token_id: token,
                    sequence_index: step,
                    temperature: 0.0,
                };
                let res = tenant_engine
                    .dispatch_decode_step(req, -1, 0, (step + 1) as u64)
                    .unwrap();
                assert_eq!(res.step_fence_point, (step + 1) as u64);
            }

            // Verify tenant's DMA-BUF contains strictly its own tokens
            let shared_buf = SharedBuffer::new(dmabuf, true).unwrap();
            shared_buf
                .with_cpu_read(|slice| {
                    for step in 0..steps_per_tenant {
                        let offset = step * 128;
                        let token_slice = &slice[offset..offset + 128];
                        let expected = (((token_base + step) as u32) % 251) as u8;
                        assert!(
                            token_slice.iter().all(|&b| b == expected),
                            "Tenant {} data contamination at step {}",
                            tenant_id,
                            step
                        );
                    }
                })
                .unwrap();
        });
        thread_handles.push(h);
    }

    for h in thread_handles {
        h.join().expect("Tenant thread failed");
    }
}

// ============================================================================
// 10. File Descriptor and VMA Leak Absence Across 100 Decode Iterations
// ============================================================================

#[test]
fn test_adversarial_m2_fd_and_vma_leak_free_100_iterations() {
    let _lock = acquire_lock();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let initial_fds = get_open_fds();
    let initial_vmas = count_accelerator_mappings();

    // Run 100 full cycles of allocate -> export -> attach -> decode -> drop
    for cycle in 0..100 {
        let mut engine = XrtDecodeEngine::new();
        engine.initialize("cycle.xclbin").unwrap();

        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 4096, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();
        engine.attach_kv_cache(&dmabuf).unwrap();

        let req = DecodeStepRequest {
            input_token_id: (cycle + 1) as u32,
            sequence_index: 0,
            temperature: 0.0,
        };
        let _ = engine.dispatch_decode_step(req, -1, 0, 1).unwrap();

        // Dropping engine and dmabuf releases all FDs and munmaps memory
    }

    let final_fds = get_open_fds();
    let final_vmas = count_accelerator_mappings();

    let leaked_fds: Vec<_> = final_fds.difference(&initial_fds).collect();
    for &fd in &leaked_fds {
        if let Ok(target) = fs::read_link(format!("/proc/self/fd/{}", fd)) {
            println!("Leaked FD {}: -> {:?}", fd, target);
        }
    }
    assert!(
        leaked_fds.is_empty(),
        "Detected {} leaked file descriptors after 100 decode cycles: {:?}",
        leaked_fds.len(),
        leaked_fds
    );

    assert!(
        final_vmas <= initial_vmas,
        "Detected unmapped VMA leak: initial {}, final {}",
        initial_vmas,
        final_vmas
    );
}
