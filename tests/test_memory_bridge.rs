// SPDX-License-Identifier: Apache-2.0
//! Unit and integration tests for Cross-Accelerator Memory Bridge.

use std::sync::Arc;
use zero_copy_model_runner::backend::{
    open_or_mock, DeviceBackend, DeviceType, MockDeviceBackend, PhysicalDeviceBackend,
};
use zero_copy_model_runner::memory::{
    MemoryBridge, MemoryError, SharedBuffer, SyncDirection,
};

#[test]
fn test_mock_gem_allocation_and_prime_export() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 64 * 1024; // 64 KiB

    // 1. Allocate GEM BO in GTT domain
    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 64)
        .expect("Failed to allocate mock GEM BO");
    assert!(gpu_bo.handle() > 0);
    assert_eq!(gpu_bo.size(), size);

    // 2. Export PRIME FD
    let dmabuf = gpu_bo.export_prime_fd().expect("Failed to export PRIME FD");
    assert!(dmabuf.as_raw_fd() >= 0);
    assert_eq!(dmabuf.size(), size);

    // 3. Clone handle
    let cloned_handle = dmabuf.try_clone().expect("Failed to clone DmaBufHandle");
    assert!(cloned_handle.as_raw_fd() >= 0);
    assert_ne!(dmabuf.as_raw_fd(), cloned_handle.as_raw_fd());
    assert_eq!(cloned_handle.size(), size);

    // 4. Test explicit CPU cache synchronization brackets
    dmabuf.begin_cpu_access(SyncDirection::ReadWrite).expect("begin_cpu_access failed");
    dmabuf.end_cpu_access(SyncDirection::ReadWrite).expect("end_cpu_access failed");
}

#[test]
fn test_mock_npu_import() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());
    let size = 128 * 1024; // 128 KiB

    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
    let dmabuf = gpu_bo.export_prime_fd().unwrap();

    let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &dmabuf)
        .expect("Failed to import dma-buf to mock NPU");
    assert!(npu_bo.handle() > 0);
    assert_eq!(npu_bo.size(), size);
    assert!(npu_bo.xdna_addr() > 0);
}

#[test]
fn test_shared_buffer_mmap_and_64byte_alignment() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 64 * 1024;

    let (_gpu_bo, mut shared) = MemoryBridge::allocate_shared(&mock_gpu, size, true)
        .expect("Failed to allocate shared buffer");

    assert_eq!(shared.len(), size);
    assert!(!shared.is_empty());

    let raw_ptr = shared.as_ptr().expect("Buffer should be CPU-mapped");
    assert_eq!(
        raw_ptr as usize % 64,
        0,
        "CPU virtual address must adhere to strict 64-byte alignment"
    );

    // Write sentinel pattern
    shared
        .with_cpu_write(|slice| {
            slice[0..8].copy_from_slice(b"SENTINEL");
        })
        .expect("Failed to write to CPU buffer");

    // Read back sentinel pattern
    let read_val = shared
        .with_cpu_read(|slice| slice[0..8].to_vec())
        .expect("Failed to read from CPU buffer");
    assert_eq!(&read_val, b"SENTINEL");
}

#[test]
fn test_unmapped_shared_buffer_returns_invalid_state() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let (_gpu_bo, mut unmapped) = MemoryBridge::allocate_shared(&mock_gpu, 4096, false)
        .expect("Failed to allocate unmapped buffer");

    assert!(unmapped.as_ptr().is_none());

    let read_res = unmapped.with_cpu_read(|_| ());
    match read_res {
        Err(MemoryError::InvalidState(_)) => {}
        other => panic!("Expected InvalidState error, got {:?}", other),
    }

    let write_res = unmapped.with_cpu_write(|_| ());
    match write_res {
        Err(MemoryError::InvalidState(_)) => {}
        other => panic!("Expected InvalidState error, got {:?}", other),
    }
}

#[test]
fn test_misaligned_allocation_rejected() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let misaligned_size = 63; // not a multiple of 64

    let res = MemoryBridge::allocate_gem_bo(&mock_gpu, misaligned_size, 64);
    match res {
        Err(MemoryError::AlignmentError { required: 64, .. }) => {}
        other => panic!("Expected AlignmentError for size 63, got {:?}", other),
    }
}

#[test]
fn test_open_or_mock_device_factory() {
    // 1. GPU render node (physical or mock)
    let gpu = open_or_mock("/dev/dri/renderD128").expect("open_or_mock failed for GPU");
    assert!(gpu.as_raw_fd() >= 0);
    assert_eq!(gpu.device_type(), DeviceType::Gpu);

    // 2. NPU accel node (physical or mock)
    let npu = open_or_mock("/dev/accel/accel0").expect("open_or_mock failed for NPU");
    assert!(npu.as_raw_fd() >= 0);
    assert_eq!(npu.device_type(), DeviceType::Npu);

    // 3. Non-existent device falls back to mock
    let fallback = open_or_mock("/dev/nonexistent_device_node").expect("Fallback should succeed");
    assert!(fallback.is_mock());
    assert!(fallback.as_raw_fd() >= 0);
}

#[test]
fn test_physical_silicon_if_present() {
    // If physical nodes exist and are readable/writable, test real hardware path
    if let (Ok(gpu), Ok(npu)) = (
        PhysicalDeviceBackend::open("/dev/dri/renderD128", DeviceType::Gpu),
        PhysicalDeviceBackend::open("/dev/accel/accel0", DeviceType::Npu),
    ) {
        let gpu_arc: Arc<dyn DeviceBackend> = Arc::new(gpu);
        let npu_arc: Arc<dyn DeviceBackend> = Arc::new(npu);
        let size = 64 * 1024;

        let gpu_bo = MemoryBridge::allocate_gem_bo(&gpu_arc, size, 4096)
            .expect("Physical AMDGPU GEM allocation failed");
        let dmabuf = gpu_bo
            .export_prime_fd()
            .expect("Physical PRIME export failed");

        let npu_bo = MemoryBridge::import_dma_buf(&npu_arc, &dmabuf)
            .expect("Physical AMDXDNA PRIME import failed");
        assert!(npu_bo.handle() > 0);
        assert_eq!(npu_bo.size(), size);

        let mut shared = SharedBuffer::new(dmabuf, true).expect("Physical mmap failed");
        assert_eq!(shared.as_ptr().unwrap() as usize % 64, 0);

        shared
            .with_cpu_write(|slice| {
                slice[0..4].copy_from_slice(b"LIVE");
            })
            .expect("Physical CPU write failed");

        let val = shared
            .with_cpu_read(|slice| slice[0..4].to_vec())
            .expect("Physical CPU read failed");
        assert_eq!(&val, b"LIVE");
    }
}
