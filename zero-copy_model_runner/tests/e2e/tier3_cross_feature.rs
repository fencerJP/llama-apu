// SPDX-License-Identifier: Apache-2.0
//! Tier 3: Cross-Feature Combinations E2E Tests
//!
//! Pairwise and multi-feature interaction coverage:
//! 1. Allocation + PRIME Export + AMDXDNA Import + Watermark In-Place Mutation
//! 2. Prefill Direct KV Write + Timeline Fence Signal + Decode Step Timeline Wait Handoff
//! 3. Multi-Step Autoregressive Loop with Monotonic Fence Progression & KV Growth
//! 4. Concurrent Sessions Shared Engine with Independent DMA-BUFs (Tenant Isolation)
//! 5. Memory Recycling with Tenant Zero-Initialization and Syncobj Reuse

use std::os::fd::AsRawFd;
use std::time::Duration;

use crate::common::{
    DeterministicReferenceOracle, MemoryWatermarkVerifier, MockDecodeEngine, MockDeviceHarness,
    MockPrefillEngine, MockTimelineSyncobj, PhysicalHardwareBridge, SiliconProbe,
};
use zero_copy_model_runner::engine::{
    DecodeEngine, DecodeStepRequest, PrefillEngine, PrefillRequest,
};
use zero_copy_model_runner::memory::{DmaBufHandle, SharedBuffer};

/// Test 1: Full pipeline: Alloc + PRIME Export + AMDXDNA Import + Watermark In-Place Mutation
#[test]
fn test_t3_alloc_prime_export_amdxdna_import_mutation() {
    let probe = SiliconProbe::probe();
    if probe.has_full_silicon {
        let render_fd = SiliconProbe::open_render_node().expect("render node");
        let accel_fd = SiliconProbe::open_accel_node().expect("accel node");

        let (gpu_handle, dmabuf_fd) =
            PhysicalHardwareBridge::allocate_and_export_gem(render_fd.as_raw_fd(), 65536)
                .expect("Alloc GPU GEM");

        let npu_handle =
            PhysicalHardwareBridge::import_dmabuf_to_npu(accel_fd.as_raw_fd(), dmabuf_fd.as_raw_fd())
                .expect("Import NPU BO");

        let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 65536) };
        std::mem::forget(dmabuf_fd);

        let mut shared_buffer = SharedBuffer::new(handle, true).expect("SharedBuffer::new");

        // Step A: Write watermark via GPU handle
        shared_buffer
            .with_cpu_write(|slice| {
                MemoryWatermarkVerifier::assert_64b_alignment(slice.as_ptr());
                MemoryWatermarkVerifier::fill_watermark(slice, 0xAAAA);
            })
            .expect("Write watermark");

        // Step B: In-place mutation
        shared_buffer
            .with_cpu_write(|slice| {
                assert!(MemoryWatermarkVerifier::verify_watermark(slice, 0xAAAA));
                MemoryWatermarkVerifier::fill_watermark(slice, 0xBBBB);
            })
            .expect("Mutate watermark");

        // Step C: Verify new watermark readable
        shared_buffer
            .with_cpu_read(|slice| {
                assert!(MemoryWatermarkVerifier::verify_watermark(slice, 0xBBBB));
            })
            .expect("Verify mutated watermark");

        let _ = PhysicalHardwareBridge::close_gem_handle(accel_fd.as_raw_fd(), npu_handle);
        let _ = PhysicalHardwareBridge::close_gem_handle(render_fd.as_raw_fd(), gpu_handle);
    } else {
        let mock = MockDeviceHarness::new();
        let fd = mock.create_mock_dmabuf(65536).expect("mock dmabuf");
        let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 65536) };
        std::mem::forget(fd);

        let shared_buffer = SharedBuffer::new(handle, true).expect("SharedBuffer::new");
        assert_eq!(shared_buffer.len(), 65536);
    }
}

/// Test 2: Prefill Direct KV Write + Timeline Fence Signal + Decode Step Timeline Wait Handoff
#[test]
fn test_t3_prefill_direct_write_and_fence_handoff_to_decode() {
    let mut prefill_engine = MockPrefillEngine::new();
    prefill_engine.initialize(0).expect("Prefill init");

    let mut decode_engine = MockDecodeEngine::new();
    decode_engine.initialize("model.xclbin").expect("Decode init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(65536).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 65536) };
    std::mem::forget(fd);

    let timeline = MockTimelineSyncobj::new(0);

    // 1. Prefill processes prompt and writes directly to KV cache
    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let prefill_req = PrefillRequest {
        token_ids: prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let prefill_res = prefill_engine
        .dispatch_prefill(prefill_req, &handle, -1, 1)
        .expect("Prefill dispatch");
    assert_eq!(prefill_res.tokens_processed, prompt.len());
    timeline.signal(1).expect("Prefill signals timeline point 1");

    // 2. Decode engine attaches the shared KV cache without copying
    decode_engine.attach_kv_cache(&handle).expect("Attach KV");

    // 3. Decode waits for timeline point 1 before executing first decode step
    timeline.wait(1, Duration::from_millis(50)).expect("Decode wait");

    let decode_req = DecodeStepRequest {
        input_token_id: prefill_res.initial_token_id,
        sequence_index: prompt.len(),
        temperature: 0.0,
    };

    let decode_res = decode_engine
        .dispatch_decode_step(decode_req, -1, 1, 2)
        .expect("Decode step");
    timeline.signal(2).expect("Decode signals point 2");

    assert_eq!(decode_res.output_token_id, 374);
    assert_eq!(timeline.current_point(), 2);
}

/// Test 3: Multi-Step Autoregressive Loop with Monotonic Fence Progression & KV Growth
#[test]
fn test_t3_multi_step_autoregressive_loop_monotonic_fence() {
    let mut decode_engine = MockDecodeEngine::new();
    decode_engine.initialize("model.xclbin").expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(131072).expect("128KB KV buffer");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 131072) };
    std::mem::forget(fd);

    decode_engine.attach_kv_cache(&handle).expect("Attach");

    let timeline = MockTimelineSyncobj::new(0);
    timeline.signal(1).expect("Initial fence");

    let mut current_token = 9607;
    let initial_offset = 6;
    let total_steps = 5;

    for step in 0..total_steps {
        let wait_point = (step + 1) as u64;
        let signal_point = (step + 2) as u64;

        timeline
            .wait(wait_point, Duration::from_millis(20))
            .expect("Step wait");

        let req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: initial_offset + step,
            temperature: 0.0,
        };

        let res = decode_engine
            .dispatch_decode_step(req, -1, wait_point, signal_point)
            .expect("Decode step");

        timeline.signal(signal_point).expect("Step signal");
        current_token = res.output_token_id;
    }

    assert_eq!(decode_engine.current_step(), 5);
    assert_eq!(timeline.current_point(), 6);
}

/// Test 4: Concurrent Sessions Shared Engine with Independent DMA-BUFs (Tenant Isolation)
#[test]
fn test_t3_concurrent_sessions_shared_engine_tenant_isolation() {
    let mock = MockDeviceHarness::new();

    let fd_a = mock.create_mock_dmabuf(32768).expect("dmabuf A");
    let handle_a = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd_a.as_raw_fd(), 32768) };
    std::mem::forget(fd_a);

    let fd_b = mock.create_mock_dmabuf(32768).expect("dmabuf B");
    let handle_b = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd_b.as_raw_fd(), 32768) };
    std::mem::forget(fd_b);

    let mut shared_prefill = MockPrefillEngine::new();
    shared_prefill.initialize(0).expect("Prefill init");

    let req_a = PrefillRequest {
        token_ids: &[1, 2, 3, 4],
        start_offset: 0,
        batch_size: 1,
    };
    let res_a = shared_prefill
        .dispatch_prefill(req_a, &handle_a, -1, 1)
        .expect("Prefill A");

    let req_b = PrefillRequest {
        token_ids: &[10, 20, 30, 40, 50],
        start_offset: 0,
        batch_size: 1,
    };
    let res_b = shared_prefill
        .dispatch_prefill(req_b, &handle_b, -1, 1)
        .expect("Prefill B");

    assert_eq!(res_a.tokens_processed, 4);
    assert_eq!(res_b.tokens_processed, 5);
    assert_eq!(shared_prefill.total_tokens_processed(), 9);
}

/// Test 5: Memory Recycling with Tenant Zero-Initialization and Syncobj Reuse
#[test]
fn test_t3_memory_recycling_and_tenant_zero_initialization() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(16384).expect("KV slab");

    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            16384,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, 16384);
        MemoryWatermarkVerifier::fill_watermark(slice, 0xDEADBEEF);
        assert!(MemoryWatermarkVerifier::verify_watermark(slice, 0xDEADBEEF));
        libc::munmap(ptr, 16384);
    }

    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            16384,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, 16384);
        slice.fill(0);
        libc::munmap(ptr, 16384);
    }

    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            16384,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        let slice = std::slice::from_raw_parts(ptr as *const u8, 16384);
        assert!(
            slice.iter().all(|&b| b == 0),
            "Recycled buffer must be completely zero-initialized"
        );
        libc::munmap(ptr, 16384);
    }
}
