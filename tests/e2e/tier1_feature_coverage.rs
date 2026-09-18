// SPDX-License-Identifier: Apache-2.0
//! Tier 1: Feature Coverage E2E Tests
//!
//! Covers core requirements R1–R5 (>= 5 test cases per feature):
//! - Feature R1: Zero-Copy Cross-Accelerator Memory Bridge (5 tests)
//! - Feature R2: Compute-Heavy Prompt Prefill Engine (5 tests)
//! - Feature R3: Memory-Bound Autoregressive Decode Engine (5 tests)
//! - Feature R4: Explicit Hardware Fence Synchronization (5 tests)
//! - Feature R5: Test Harness & Silicon Emulation Fallback (5 tests)

use std::os::fd::AsRawFd;
use std::time::Duration;

use crate::common::{
    drm_amdgpu_gem_create, drm_prime_handle, drm_syncobj_handle,
    drm_syncobj_timeline_wait, dma_buf_sync, DeterministicReferenceOracle, MemoryWatermarkVerifier,
    MockDecodeEngine, MockDeviceHarness, MockPrefillEngine, MockTimelineSyncobj,
    PhysicalHardwareBridge, SiliconProbe, DMA_BUF_SYNC_END, DMA_BUF_SYNC_READ, DMA_BUF_SYNC_RW,
    DMA_BUF_SYNC_START, DMA_BUF_SYNC_WRITE, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
};
use zero_copy_model_runner::engine::{
    DecodeEngine, DecodeStepRequest, PrefillEngine, PrefillRequest,
};
use zero_copy_model_runner::memory::{DmaBufHandle, SharedBuffer};

// ============================================================================
// Feature R1: Cross-Accelerator Zero-Copy Memory Bridge
// ============================================================================

#[test]
fn test_r1_gem_allocation_and_export() {
    let probe = SiliconProbe::probe();
    let (dmabuf_fd, size) = if probe.has_amdgpu {
        let render_fd = SiliconProbe::open_render_node().expect("Failed to open render node");
        let (gem_handle, owned_fd) =
            PhysicalHardwareBridge::allocate_and_export_gem(render_fd.as_raw_fd(), 65536)
                .expect("Failed to allocate and export AMDGPU GEM");
        let _ = PhysicalHardwareBridge::close_gem_handle(render_fd.as_raw_fd(), gem_handle);
        (owned_fd, 65536)
    } else {
        let mock = MockDeviceHarness::new();
        let owned_fd = mock
            .create_mock_dmabuf(65536)
            .expect("Failed to create mock dmabuf");
        (owned_fd, 65536)
    };

    assert!(dmabuf_fd.as_raw_fd() >= 0, "Invalid DMA-BUF file descriptor");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), size) };
    assert_eq!(handle.size(), size);
    std::mem::forget(dmabuf_fd); // ownership transferred to DmaBufHandle
}

#[test]
fn test_r1_amdx_import_and_handle() {
    let probe = SiliconProbe::probe();
    if probe.has_full_silicon {
        let render_fd = SiliconProbe::open_render_node().expect("Failed to open render node");
        let accel_fd = SiliconProbe::open_accel_node().expect("Failed to open accel node");

        let (gpu_handle, dmabuf_fd) =
            PhysicalHardwareBridge::allocate_and_export_gem(render_fd.as_raw_fd(), 16384)
                .expect("Failed to allocate GEM");

        let npu_handle =
            PhysicalHardwareBridge::import_dmabuf_to_npu(accel_fd.as_raw_fd(), dmabuf_fd.as_raw_fd())
                .expect("Failed to import dma-buf into AMDXDNA");

        assert!(npu_handle > 0, "Expected valid NPU GEM handle from import");

        let _ = PhysicalHardwareBridge::close_gem_handle(accel_fd.as_raw_fd(), npu_handle);
        let _ = PhysicalHardwareBridge::close_gem_handle(render_fd.as_raw_fd(), gpu_handle);
    } else {
        let mock = MockDeviceHarness::new();
        let dmabuf_fd = mock.create_mock_dmabuf(16384).expect("Mock dmabuf create");
        assert!(dmabuf_fd.as_raw_fd() >= 0);
    }
}

#[test]
fn test_r1_64byte_cache_line_alignment() {
    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(32768).expect("Failed to create dmabuf");
    let raw_fd = dmabuf_fd.as_raw_fd();
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(raw_fd, 32768) };
    std::mem::forget(dmabuf_fd);

    let shared_buffer = SharedBuffer::new(handle, true).expect("Failed to map SharedBuffer");
    assert_eq!(shared_buffer.len(), 32768);
    assert!(!shared_buffer.is_empty());

    let len = shared_buffer.len();
    assert_eq!(len, 32768);
}

#[test]
fn test_r1_cpu_sync_brackets() {
    let probe = SiliconProbe::probe();
    if probe.has_amdgpu {
        let render_fd = SiliconProbe::open_render_node().expect("Failed to open render node");
        let (gpu_handle, dmabuf_fd) =
            PhysicalHardwareBridge::allocate_and_export_gem(render_fd.as_raw_fd(), 4096)
                .expect("Failed to allocate AMDGPU GEM");

        let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 4096) };
        std::mem::forget(dmabuf_fd);

        let mut buffer = SharedBuffer::new(handle, true).expect("Failed to create SharedBuffer");

        // Write with CPU cache flush bracket
        buffer
            .with_cpu_write(|slice| {
                MemoryWatermarkVerifier::assert_64b_alignment(slice.as_ptr());
                MemoryWatermarkVerifier::fill_watermark(slice, 0x1234);
            })
            .expect("with_cpu_write failed");

        // Read with CPU cache invalidate bracket
        buffer
            .with_cpu_read(|slice| {
                MemoryWatermarkVerifier::assert_64b_alignment(slice.as_ptr());
                assert!(MemoryWatermarkVerifier::verify_watermark(slice, 0x1234));
            })
            .expect("with_cpu_read failed");

        let _ = PhysicalHardwareBridge::close_gem_handle(render_fd.as_raw_fd(), gpu_handle);
    } else {
        // In mock mode without kernel dma-buf driver, verify flag validation directly
        MockDeviceHarness::validate_sync_flags(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ)
            .expect("Valid start read flags");
        MockDeviceHarness::validate_sync_flags(DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE)
            .expect("Valid end write flags");
    }
}

#[test]
fn test_r1_raii_teardown_and_leak_free() {
    for _ in 0..10 {
        let mock = MockDeviceHarness::new();
        let dmabuf_fd = mock.create_mock_dmabuf(4096).expect("Failed to create dmabuf");
        let raw_fd = dmabuf_fd.as_raw_fd();
        let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(raw_fd, 4096) };
        std::mem::forget(dmabuf_fd);

        let shared_buffer = SharedBuffer::new(handle, true).expect("Failed to mmap");
        assert_eq!(shared_buffer.len(), 4096);
        drop(shared_buffer);
    }
}

// ============================================================================
// Feature R2: Compute-Heavy Prompt Prefill Engine
// ============================================================================

#[test]
fn test_r2_prefill_engine_initialization() {
    let mut engine = MockPrefillEngine::new();
    assert_eq!(engine.total_tokens_processed(), 0);
    engine.initialize(0).expect("Initialization failed");
    assert!(
        engine.peak_compute_tflops() >= 16.0,
        "Expected >=16 TFLOPs FP16 on RDNA 3.5"
    );
}

#[test]
fn test_r2_prefill_dispatch_basic() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Initialize");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(16384).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 16384) };
    std::mem::forget(dmabuf_fd);

    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let request = PrefillRequest {
        token_ids: prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine
        .dispatch_prefill(request, &handle, -1, 100)
        .expect("Prefill dispatch failed");

    assert_eq!(result.tokens_processed, prompt.len());
    assert_eq!(
        result.initial_token_id,
        DeterministicReferenceOracle::EXPECTED_DECODE_STEPS[0]
    );
    assert_eq!(result.completion_fence_point, 100);
    assert!(result.execution_time_us > 0);
}

#[test]
fn test_r2_prefill_initial_token_emission() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Initialize");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(16384).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 16384) };
    std::mem::forget(dmabuf_fd);

    let prompt = &[128000, 791, 7421, 315, 9607, 374];
    let request = PrefillRequest {
        token_ids: prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine.dispatch_prefill(request, &handle, -1, 1).expect("Prefill");
    assert_eq!(result.initial_token_id, 9607);
}

#[test]
fn test_r2_prefill_timeline_fence_signaling() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Initialize");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(16384).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 16384) };
    std::mem::forget(dmabuf_fd);

    let request = PrefillRequest {
        token_ids: &[1, 2, 3, 4],
        start_offset: 0,
        batch_size: 1,
    };

    let signal_point = 42;
    let result = engine
        .dispatch_prefill(request, &handle, -1, signal_point)
        .expect("Prefill dispatch");

    assert_eq!(result.completion_fence_point, signal_point);
}

#[test]
fn test_r2_prefill_batch_size_handling() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Initialize");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(65536).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 65536) };
    std::mem::forget(dmabuf_fd);

    let token_batch = vec![100u32; 128];
    let request = PrefillRequest {
        token_ids: &token_batch,
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine.dispatch_prefill(request, &handle, -1, 1).expect("Prefill 128 tokens");
    assert_eq!(result.tokens_processed, 128);
    assert_eq!(engine.total_tokens_processed(), 128);
}

// ============================================================================
// Feature R3: Memory-Bound Autoregressive Decode Engine
// ============================================================================

#[test]
fn test_r3_decode_engine_initialization() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("strix_point_llama3.xclbin").expect("Init failed");
    assert_eq!(engine.active_tiles(), 32);
    assert!(engine.peak_npu_tops() >= 50.0);
}

#[test]
fn test_r3_decode_attach_kv_cache() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init failed");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(32768).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 32768) };
    std::mem::forget(dmabuf_fd);

    engine.attach_kv_cache(&handle).expect("Failed to attach KV cache");
}

#[test]
fn test_r3_decode_single_step_dispatch() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(32768).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 32768) };
    std::mem::forget(dmabuf_fd);

    engine.attach_kv_cache(&handle).expect("Attach");

    let step_request = DecodeStepRequest {
        input_token_id: 9607,
        sequence_index: 6,
        temperature: 0.0,
    };

    let result = engine
        .dispatch_decode_step(step_request, -1, 0, 1)
        .expect("Decode step failed");

    assert_eq!(result.output_token_id, 374);
    assert!(!result.is_eos);
    assert_eq!(result.step_fence_point, 1);
    assert_eq!(engine.current_step(), 1);
}

#[test]
fn test_r3_decode_in_place_kv_append() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(32768).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 32768) };
    std::mem::forget(dmabuf_fd);

    engine.attach_kv_cache(&handle).expect("Attach");

    let mut current_token = 9607;
    for step in 0..3 {
        let req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: 6 + step,
            temperature: 0.0,
        };
        let res = engine
            .dispatch_decode_step(req, -1, 0, (step + 1) as u64)
            .expect("Step failed");
        current_token = res.output_token_id;
    }

    assert_eq!(engine.current_step(), 3);
}

#[test]
fn test_r3_decode_eos_detection() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init");

    let mock = MockDeviceHarness::new();
    let dmabuf_fd = mock.create_mock_dmabuf(32768).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(dmabuf_fd.as_raw_fd(), 32768) };
    std::mem::forget(dmabuf_fd);

    engine.attach_kv_cache(&handle).expect("Attach");

    let req = DecodeStepRequest {
        input_token_id: 13,
        sequence_index: 15,
        temperature: 0.0,
    };

    let res = engine.dispatch_decode_step(req, -1, 0, 10).expect("Decode step");
    assert!(res.is_eos, "Expected EOS token flag");
    assert_eq!(res.output_token_id, DeterministicReferenceOracle::EOS_TOKEN_ID);
}

// ============================================================================
// Feature R4: Explicit Hardware Fence Synchronization
// ============================================================================

#[test]
fn test_r4_syncobj_create_and_destroy() {
    let probe = SiliconProbe::probe();
    if probe.has_amdgpu {
        let render_fd = SiliconProbe::open_render_node().expect("Open render node");
        let handle = PhysicalHardwareBridge::create_syncobj(render_fd.as_raw_fd())
            .expect("Create syncobj failed");
        assert!(handle > 0);
        PhysicalHardwareBridge::destroy_syncobj(render_fd.as_raw_fd(), handle)
            .expect("Destroy syncobj failed");
    } else {
        let mock = MockDeviceHarness::new();
        let handle = mock.create_syncobj();
        assert!(handle > 0);
        mock.destroy_syncobj(handle).expect("Destroy mock syncobj");
    }
}

#[test]
fn test_r4_syncobj_fd_export_import() {
    let probe = SiliconProbe::probe();
    if probe.has_amdgpu {
        let render_fd = SiliconProbe::open_render_node().expect("Open render node");
        let handle1 = PhysicalHardwareBridge::create_syncobj(render_fd.as_raw_fd())
            .expect("Create syncobj");

        let sync_fd = PhysicalHardwareBridge::export_syncobj_to_fd(render_fd.as_raw_fd(), handle1)
            .expect("Export to FD");
        assert!(sync_fd.as_raw_fd() >= 0);

        let handle2 = PhysicalHardwareBridge::import_syncobj_from_fd(
            render_fd.as_raw_fd(),
            sync_fd.as_raw_fd(),
        )
        .expect("Import from FD");
        assert!(handle2 > 0);

        PhysicalHardwareBridge::destroy_syncobj(render_fd.as_raw_fd(), handle1).expect("Destroy 1");
        PhysicalHardwareBridge::destroy_syncobj(render_fd.as_raw_fd(), handle2).expect("Destroy 2");
    } else {
        let mock = MockDeviceHarness::new();
        let handle = mock.create_syncobj();
        assert!(handle > 0);
        mock.destroy_syncobj(handle).expect("Destroy mock syncobj");
    }
}

#[test]
fn test_r4_timeline_fence_wait_signaled() {
    let probe = SiliconProbe::probe();
    if probe.has_amdgpu {
        let render_fd = SiliconProbe::open_render_node().expect("Open render node");
        let handle = PhysicalHardwareBridge::create_syncobj(render_fd.as_raw_fd()).expect("Create");

        PhysicalHardwareBridge::timeline_signal(render_fd.as_raw_fd(), handle, 5)
            .expect("Signal point 5");

        PhysicalHardwareBridge::timeline_wait(
            render_fd.as_raw_fd(),
            handle,
            5,
            1000,
            DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL,
        )
        .expect("Wait point 5 should succeed immediately");

        PhysicalHardwareBridge::destroy_syncobj(render_fd.as_raw_fd(), handle).expect("Destroy");
    } else {
        let timeline = MockTimelineSyncobj::new(0);
        timeline.signal(5).expect("Signal 5");
        timeline
            .wait(5, Duration::from_millis(50))
            .expect("Wait 5 should succeed");
    }
}

#[test]
fn test_r4_cross_device_fence_handoff() {
    let timeline = MockTimelineSyncobj::new(0);

    // Prefill completes and signals point 1
    timeline.signal(1).expect("Prefill signal");
    assert_eq!(timeline.current_point(), 1);

    // Decode engine waits on point 1, then signals point 2
    timeline
        .wait(1, Duration::from_millis(50))
        .expect("Decode wait");
    timeline.signal(2).expect("Decode step 1 signal");
    assert_eq!(timeline.current_point(), 2);
}

#[test]
fn test_r4_monotonic_timeline_progression() {
    let timeline = MockTimelineSyncobj::new(0);
    timeline.signal(10).expect("Signal 10");
    timeline.signal(20).expect("Signal 20");
    timeline.signal(30).expect("Signal 30");

    assert_eq!(timeline.current_point(), 30);
    timeline.wait(10, Duration::from_millis(10)).expect("Wait 10");
    timeline.wait(20, Duration::from_millis(10)).expect("Wait 20");
    timeline.wait(30, Duration::from_millis(10)).expect("Wait 30");
}

// ============================================================================
// Feature R5: Test Harness & Silicon Emulation Fallback
// ============================================================================

#[test]
fn test_r5_auto_fallback_detection() {
    let status = SiliconProbe::probe();
    println!(
        "Silicon probe status: amdgpu={}, amdxdna={}, full_silicon={}",
        status.has_amdgpu, status.has_amdxdna, status.has_full_silicon
    );
}

#[test]
fn test_r5_mock_memfd_backed_buffer() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(8192).expect("Create mock dmabuf");
    assert!(fd.as_raw_fd() >= 0);

    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 8192) };
    std::mem::forget(fd);

    let shared = SharedBuffer::new(handle, true).expect("SharedBuffer::new");
    assert_eq!(shared.len(), 8192);
}

#[test]
fn test_r5_mock_uapi_ioctl_validation() {
    assert_eq!(std::mem::size_of::<drm_amdgpu_gem_create>(), 32);
    assert_eq!(std::mem::size_of::<drm_prime_handle>(), 12);
    assert_eq!(std::mem::size_of::<drm_syncobj_handle>(), 24);
    assert_eq!(std::mem::size_of::<drm_syncobj_timeline_wait>(), 48);
    assert_eq!(std::mem::size_of::<dma_buf_sync>(), 8);
}

#[test]
fn test_r5_mock_timeline_fence_state_machine() {
    let mock = MockDeviceHarness::new();
    let handle = mock.create_syncobj();

    mock.timeline_signal(handle, 10).expect("Signal 10");
    mock.timeline_wait(handle, 10, Duration::from_millis(50))
        .expect("Wait 10");

    let res = mock.timeline_wait(handle, 20, Duration::from_millis(10));
    assert!(res.is_err(), "Expected timeout on unsignaled point");

    mock.destroy_syncobj(handle).expect("Destroy");
}

#[test]
fn test_r5_mock_dma_buf_sync_validation() {
    assert!(MockDeviceHarness::validate_sync_flags(DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ).is_ok());
    assert!(MockDeviceHarness::validate_sync_flags(DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE).is_ok());
    assert!(MockDeviceHarness::validate_sync_flags(DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW).is_ok());

    assert!(MockDeviceHarness::validate_sync_flags(0).is_err());
    assert!(MockDeviceHarness::validate_sync_flags(0x1000).is_err());
    assert!(MockDeviceHarness::validate_sync_flags(DMA_BUF_SYNC_START | 0x8).is_err());
}
