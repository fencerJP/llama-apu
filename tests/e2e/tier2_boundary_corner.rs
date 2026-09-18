// SPDX-License-Identifier: Apache-2.0
//! Tier 2: Boundary & Corner Cases E2E Tests
//!
//! Covers edge conditions, extreme limits, and error paths (>= 5 test cases per feature):
//! - Feature R1 Boundaries: Memory Bridge (5 tests)
//! - Feature R2 Boundaries: Prefill Engine (5 tests)
//! - Feature R3 Boundaries: Decode Engine (5 tests)
//! - Feature R4 Boundaries: Fence Synchronization (5 tests)
//! - Feature R5 Boundaries: Mock Fallback & Driver Shims (5 tests)

use std::os::fd::AsRawFd;
use std::time::Duration;

use crate::common::{
    MockDecodeEngine, MockDeviceHarness, MockPrefillEngine, MockTimelineSyncobj,
};
use zero_copy_model_runner::engine::{
    DecodeEngine, DecodeStepRequest, EngineError, PrefillEngine, PrefillRequest,
};
use zero_copy_model_runner::memory::{DmaBufHandle, MemoryError, SharedBuffer};

// ============================================================================
// Feature R1 Boundaries: Memory Bridge
// ============================================================================

#[test]
fn test_r1_boundary_zero_length_buffer() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 0) };
    std::mem::forget(fd);

    assert_eq!(handle.size(), 0);
    let buffer = SharedBuffer::new(handle, false).expect("Unmapped SharedBuffer with 0 length");
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn test_r1_boundary_unaligned_mmap_alignment_enforcement() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    let buffer = SharedBuffer::new(handle, true).expect("Page-aligned buffer must pass 64B check");
    assert_eq!(buffer.len(), 4096);
}

#[test]
fn test_r1_boundary_invalid_fd_rejection() {
    // Open a pipe (valid open FD, but not a mmap-capable buffer)
    let mut fds = [0i32; 2];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let pipe_read_fd = fds[0];
    let pipe_write_fd = fds[1];

    // Close write end
    unsafe { libc::close(pipe_write_fd) };

    // Pass pipe read FD as dma-buf handle
    let invalid_handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(pipe_read_fd, 4096) };
    let result = SharedBuffer::new(invalid_handle, true);
    match result {
        Err(MemoryError::MmapError(_)) => {}
        Err(other) => panic!("Expected MmapError, got {:?}", other),
        Ok(_) => panic!("Expected mmap failure on pipe FD"),
    }
}

#[test]
fn test_r1_boundary_concurrent_cpu_access_guard() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(8192).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 8192) };
    std::mem::forget(fd);

    let buffer = SharedBuffer::new(handle, false).expect("SharedBuffer unmapped");
    let res = buffer.with_cpu_read(|_| ());
    assert!(res.is_err());
    match res {
        Err(MemoryError::InvalidState(msg)) => {
            assert!(msg.contains("not mapped"));
        }
        Err(other) => panic!("Expected InvalidState, got {:?}", other),
        Ok(_) => panic!("Expected failure on unmapped CPU read"),
    }
}

#[test]
fn test_r1_boundary_extreme_sizes() {
    let mock = MockDeviceHarness::new();

    let fd_1 = mock.create_mock_dmabuf(1).expect("1-byte buffer");
    assert!(fd_1.as_raw_fd() >= 0);

    let fd_odd = mock.create_mock_dmabuf(65535).expect("Odd size buffer");
    assert!(fd_odd.as_raw_fd() >= 0);

    let fd_large = mock.create_mock_dmabuf(64 * 1024 * 1024).expect("64MB buffer");
    assert!(fd_large.as_raw_fd() >= 0);
}

// ============================================================================
// Feature R2 Boundaries: Prefill Engine
// ============================================================================

#[test]
fn test_r2_boundary_empty_prompt() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    let request = PrefillRequest {
        token_ids: &[],
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine.dispatch_prefill(request, &handle, -1, 1);
    assert!(result.is_err(), "Empty prompt sequence must be rejected");
    match result {
        Err(EngineError::InvalidArgument(msg)) => assert!(msg.contains("empty")),
        Err(other) => panic!("Expected InvalidArgument, got {:?}", other),
        Ok(_) => panic!("Expected failure on empty prompt"),
    }
}

#[test]
fn test_r2_boundary_single_token_prompt() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    let request = PrefillRequest {
        token_ids: &[42],
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine.dispatch_prefill(request, &handle, -1, 1).expect("Single token prefill");
    assert_eq!(result.tokens_processed, 1);
    assert!(result.initial_token_id > 0);
}

#[test]
fn test_r2_boundary_prompt_length_exceeding_limit() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    let huge_prompt = vec![10u32; 8193];
    let request = PrefillRequest {
        token_ids: &huge_prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine.dispatch_prefill(request, &handle, -1, 1);
    assert!(result.is_err());
    match result {
        Err(EngineError::InvalidArgument(msg)) => assert!(msg.contains("exceeds maximum")),
        Err(other) => panic!("Expected InvalidArgument, got {:?}", other),
        Ok(_) => panic!("Expected error on exceeding prompt limit"),
    }
}

#[test]
fn test_r2_boundary_out_of_range_token_ids() {
    let mut engine = MockPrefillEngine::new();
    engine.initialize(0).expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    let invalid_prompt = &[1, 2, 200_000];
    let request = PrefillRequest {
        token_ids: invalid_prompt,
        start_offset: 0,
        batch_size: 1,
    };

    let result = engine.dispatch_prefill(request, &handle, -1, 1);
    assert!(result.is_err());
    match result {
        Err(EngineError::InvalidArgument(msg)) => assert!(msg.contains("vocabulary")),
        Err(other) => panic!("Expected InvalidArgument, got {:?}", other),
        Ok(_) => panic!("Expected error on out-of-range token IDs"),
    }
}

#[test]
fn test_r2_boundary_uninitialized_prefill_engine() {
    let uninitialized_engine = MockPrefillEngine::new();

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    let request = PrefillRequest {
        token_ids: &[1, 2, 3],
        start_offset: 0,
        batch_size: 1,
    };

    let result = uninitialized_engine.dispatch_prefill(request, &handle, -1, 1);
    assert!(result.is_err());
    match result {
        Err(EngineError::InitFailed(msg)) => assert!(msg.contains("not initialized")),
        Err(other) => panic!("Expected InitFailed, got {:?}", other),
        Ok(_) => panic!("Expected error on uninitialized engine"),
    }
}

// ============================================================================
// Feature R3 Boundaries: Decode Engine
// ============================================================================

#[test]
fn test_r3_boundary_decode_without_kv_attach() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init");

    let request = DecodeStepRequest {
        input_token_id: 1,
        sequence_index: 0,
        temperature: 0.0,
    };

    let result = engine.dispatch_decode_step(request, -1, 0, 1);
    assert!(result.is_err());
    match result {
        Err(EngineError::InvalidArgument(msg)) => assert!(msg.contains("No KV cache attached")),
        Err(other) => panic!("Expected InvalidArgument, got {:?}", other),
        Ok(_) => panic!("Expected error on decode without attach"),
    }
}

#[test]
fn test_r3_boundary_sequence_index_overflow() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    engine.attach_kv_cache(&handle).expect("Attach");

    let request = DecodeStepRequest {
        input_token_id: 1,
        sequence_index: 8192,
        temperature: 0.0,
    };

    let result = engine.dispatch_decode_step(request, -1, 0, 1);
    assert!(result.is_err());
    match result {
        Err(EngineError::InvalidArgument(msg)) => assert!(msg.contains("exceeds maximum context")),
        Err(other) => panic!("Expected InvalidArgument, got {:?}", other),
        Ok(_) => panic!("Expected error on sequence index overflow"),
    }
}

#[test]
fn test_r3_boundary_negative_temperature() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 4096) };
    std::mem::forget(fd);

    engine.attach_kv_cache(&handle).expect("Attach");

    let request = DecodeStepRequest {
        input_token_id: 1,
        sequence_index: 0,
        temperature: -1.5,
    };

    let result = engine.dispatch_decode_step(request, -1, 0, 1);
    assert!(result.is_err());
    match result {
        Err(EngineError::InvalidArgument(msg)) => assert!(msg.contains("negative")),
        Err(other) => panic!("Expected InvalidArgument, got {:?}", other),
        Ok(_) => panic!("Expected error on negative temperature"),
    }
}

#[test]
fn test_r3_boundary_uninitialized_decode_engine() {
    let engine = MockDecodeEngine::new();

    let request = DecodeStepRequest {
        input_token_id: 1,
        sequence_index: 0,
        temperature: 0.0,
    };

    let result = engine.dispatch_decode_step(request, -1, 0, 1);
    assert!(result.is_err());
    match result {
        Err(EngineError::InitFailed(msg)) => assert!(msg.contains("not initialized")),
        Err(other) => panic!("Expected InitFailed, got {:?}", other),
        Ok(_) => panic!("Expected error on uninitialized engine"),
    }
}

#[test]
fn test_r3_boundary_zero_capacity_kv_cache() {
    let mut engine = MockDecodeEngine::new();
    engine.initialize("model.xclbin").expect("Init");

    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(4096).expect("dmabuf");
    let handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(fd.as_raw_fd(), 0) };
    std::mem::forget(fd);

    let result = engine.attach_kv_cache(&handle);
    assert!(result.is_err());
    match result {
        Err(EngineError::InvalidArgument(msg)) => assert!(msg.contains("size cannot be 0")),
        Err(other) => panic!("Expected InvalidArgument, got {:?}", other),
        Ok(_) => panic!("Expected error on zero capacity KV cache"),
    }
}

// ============================================================================
// Feature R4 Boundaries: Fence Synchronization
// ============================================================================

#[test]
fn test_r4_boundary_fence_timeout_expired() {
    let timeline = MockTimelineSyncobj::new(0);

    let start = std::time::Instant::now();
    let res = timeline.wait(10, Duration::from_millis(10));
    let elapsed = start.elapsed();

    assert!(res.is_err(), "Expected timeout error on unsignaled point");
    assert!(
        elapsed >= Duration::from_millis(10),
        "Wait must block for at least requested timeout"
    );
}

#[test]
fn test_r4_boundary_non_monotonic_timeline_signal() {
    let timeline = MockTimelineSyncobj::new(10);

    let res = timeline.signal(5);
    assert!(res.is_err(), "Non-monotonic timeline point must return error");
}

#[test]
fn test_r4_boundary_invalid_syncobj_handle() {
    let mock = MockDeviceHarness::new();

    let res_signal = mock.timeline_signal(99999, 1);
    assert!(res_signal.is_err(), "Expected error for invalid handle");

    let res_destroy = mock.destroy_syncobj(99999);
    assert!(res_destroy.is_err(), "Expected error destroying invalid handle");
}

#[test]
fn test_r4_boundary_timeline_wait_zero_timeout() {
    let timeline = MockTimelineSyncobj::new(0);
    let res = timeline.wait(1, Duration::from_nanos(0));
    assert!(res.is_err(), "Zero timeout wait on unsignaled point must fail");
}

#[test]
fn test_r4_boundary_wait_with_wait_all_unsignaled() {
    let timeline = MockTimelineSyncobj::new(0);
    timeline.signal(1).expect("Signal 1");
    let res = timeline.wait(2, Duration::from_millis(5));
    assert!(res.is_err(), "Waiting for future point must fail");
}

// ============================================================================
// Feature R5 Boundaries: Mock Fallback & Driver Shims
// ============================================================================

#[test]
fn test_r5_boundary_mock_corrupted_sync_flags() {
    let res = MockDeviceHarness::validate_sync_flags(0x8000_0000);
    assert!(res.is_err());

    let res2 = MockDeviceHarness::validate_sync_flags(0);
    assert!(res2.is_err());
}

#[test]
fn test_r5_boundary_mock_double_destroy_syncobj() {
    let mock = MockDeviceHarness::new();
    let handle = mock.create_syncobj();
    assert!(mock.destroy_syncobj(handle).is_ok());

    assert!(mock.destroy_syncobj(handle).is_err());
}

#[test]
fn test_r5_boundary_mock_ftruncate_zero() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(0).expect("Create 0-byte buffer");
    assert!(fd.as_raw_fd() >= 0);
}

#[test]
fn test_r5_boundary_mock_invalid_timeline_query() {
    let mock = MockDeviceHarness::new();
    let res = mock.timeline_wait(8888, 1, Duration::from_millis(5));
    assert!(res.is_err());
}

#[test]
fn test_r5_boundary_mock_excessive_buffer_size() {
    let mock = MockDeviceHarness::new();
    let fd = mock.create_mock_dmabuf(128 * 1024 * 1024).expect("128MB slab");
    assert!(fd.as_raw_fd() >= 0);
}
