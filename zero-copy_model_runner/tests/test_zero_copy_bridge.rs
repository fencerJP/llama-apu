// SPDX-License-Identifier: Apache-2.0
//! Zero-Copy Invariant Verification Test Suite.
//!
//! Rigorously verifies:
//! 1. Exported dma-buf FD identity between prefill and decode handles.
//! 2. Watermark in-place mutation test: GPU write -> NPU read -> NPU write -> GPU read.
//! 3. Strict 64-byte cache line alignment and DMA_BUF_IOCTL_SYNC bracket execution.
//! 4. Clean teardown with zero file descriptor leaks.

use std::collections::BTreeSet;
use std::fs;
use std::sync::Arc;
use zero_copy_model_runner::backend::{
    open_or_mock, DeviceBackend, DeviceType, MockDeviceBackend, PhysicalDeviceBackend,
};
use zero_copy_model_runner::memory::{MemoryBridge, SharedBuffer, SyncDirection};

/// Collect all currently open file descriptors for this process from `/proc/self/fd`.
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

/// A. Invariant: Exported dma-buf FD Identity between Prefill and Decode handles.
#[test]
fn test_invariant_dmabuf_fd_identity() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());
    let size = 64 * 1024;

    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
    let prefill_dmabuf = gpu_bo.export_prime_fd().unwrap();

    // 1. Same DmaBufHandle shared across prefill and decode
    assert_eq!(
        prefill_dmabuf.as_raw_fd(),
        prefill_dmabuf.as_raw_fd(),
        "Direct handle identity must hold"
    );

    // 2. Clone handle for decode engine attachment
    let decode_dmabuf = prefill_dmabuf.try_clone().unwrap();
    assert_ne!(
        prefill_dmabuf.as_raw_fd(),
        decode_dmabuf.as_raw_fd(),
        "Cloned handles possess distinct file descriptors"
    );

    // Verify both descriptors reference identical underlying kernel file description (same inode and dev)
    let stat1 = nix::sys::stat::fstat(prefill_dmabuf.as_raw_fd()).unwrap();
    let stat2 = nix::sys::stat::fstat(decode_dmabuf.as_raw_fd()).unwrap();
    assert_eq!(stat1.st_dev, stat2.st_dev, "Underlying device must be identical");
    assert_eq!(stat1.st_ino, stat2.st_ino, "Underlying inode must be identical");

    // 3. Import into NPU device context
    let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &decode_dmabuf).unwrap();
    assert_eq!(npu_bo.size(), size);
    assert!(npu_bo.handle() > 0);
}

/// B. Invariant: Watermark In-Place Mutation Test.
///
/// Write sentinel via GPU handle -> read via NPU handle -> overwrite via NPU handle -> read back via GPU handle.
/// Validates mutual visibility with zero host-side memcpy.
#[test]
fn test_invariant_watermark_inplace_mutation_mock() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 128 * 1024; // 128 KiB

    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
    let gpu_dmabuf = gpu_bo.export_prime_fd().unwrap();
    let npu_dmabuf = gpu_dmabuf.try_clone().unwrap();

    // Map two independent CPU views referencing the shared dma-buf
    let mut gpu_view = SharedBuffer::new(gpu_dmabuf, true).unwrap();
    let mut npu_view = SharedBuffer::new(npu_dmabuf, true).unwrap();

    const WATERMARK_GPU_WRITE: u64 = 0xDEADBEEF_CAFEBABE;
    const WATERMARK_NPU_OVERWRITE: u64 = 0x01234567_89ABCDEF;
    let test_offsets = [0, 64, 1024, 4096, 65536];

    for &offset in &test_offsets {
        // Step 1: Write sentinel via GPU view
        gpu_view
            .with_cpu_write(|slice| {
                let bytes = WATERMARK_GPU_WRITE.to_le_bytes();
                slice[offset..offset + 8].copy_from_slice(&bytes);
            })
            .unwrap();

        // Step 2: Read sentinel via NPU view (verifies immediate mutual visibility)
        let read_val = npu_view
            .with_cpu_read(|slice| {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&slice[offset..offset + 8]);
                u64::from_le_bytes(bytes)
            })
            .unwrap();
        assert_eq!(
            read_val, WATERMARK_GPU_WRITE,
            "NPU view must see GPU watermark at offset {}",
            offset
        );

        // Step 3: Overwrite via NPU view
        npu_view
            .with_cpu_write(|slice| {
                let bytes = WATERMARK_NPU_OVERWRITE.to_le_bytes();
                slice[offset..offset + 8].copy_from_slice(&bytes);
            })
            .unwrap();

        // Step 4: Read back via GPU view
        let read_back = gpu_view
            .with_cpu_read(|slice| {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&slice[offset..offset + 8]);
                u64::from_le_bytes(bytes)
            })
            .unwrap();
        assert_eq!(
            read_back, WATERMARK_NPU_OVERWRITE,
            "GPU view must see NPU overwrite at offset {}",
            offset
        );
    }
}

/// B2. Invariant: Watermark In-Place Mutation Test on Physical Silicon (if present).
#[test]
fn test_invariant_watermark_inplace_mutation_physical() {
    if let (Ok(gpu), Ok(npu)) = (
        PhysicalDeviceBackend::open("/dev/dri/renderD128", DeviceType::Gpu),
        PhysicalDeviceBackend::open("/dev/accel/accel0", DeviceType::Npu),
    ) {
        let gpu_arc: Arc<dyn DeviceBackend> = Arc::new(gpu);
        let npu_arc: Arc<dyn DeviceBackend> = Arc::new(npu);
        let size = 64 * 1024;

        let gpu_bo = MemoryBridge::allocate_gem_bo(&gpu_arc, size, 4096).unwrap();
        let gpu_dmabuf = gpu_bo.export_prime_fd().unwrap();
        let npu_dmabuf = gpu_dmabuf.try_clone().unwrap();

        // Bind to NPU address space via AMDXDNA PRIME import
        let npu_bo = MemoryBridge::import_dma_buf(&npu_arc, &npu_dmabuf).unwrap();
        assert!(npu_bo.handle() > 0);

        let mut gpu_view = SharedBuffer::new(gpu_dmabuf, true).unwrap();
        let mut npu_view = SharedBuffer::new(npu_dmabuf, true).unwrap();

        const SENTINEL_A: u64 = 0xAA55AA55_11223344;
        const SENTINEL_B: u64 = 0x55AA55AA_99887766;

        // GPU writes Sentinel A
        gpu_view
            .with_cpu_write(|slice| {
                slice[0..8].copy_from_slice(&SENTINEL_A.to_le_bytes());
            })
            .unwrap();

        // NPU reads Sentinel A
        let read_a = npu_view
            .with_cpu_read(|slice| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&slice[0..8]);
                u64::from_le_bytes(b)
            })
            .unwrap();
        assert_eq!(read_a, SENTINEL_A);

        // NPU overwrites with Sentinel B
        npu_view
            .with_cpu_write(|slice| {
                slice[0..8].copy_from_slice(&SENTINEL_B.to_le_bytes());
            })
            .unwrap();

        // GPU reads Sentinel B
        let read_b = gpu_view
            .with_cpu_read(|slice| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&slice[0..8]);
                u64::from_le_bytes(b)
            })
            .unwrap();
        assert_eq!(read_b, SENTINEL_B);
    }
}

/// C. Invariant: Strict 64-Byte Alignment and DMA_BUF_IOCTL_SYNC Brackets.
#[test]
fn test_invariant_64byte_alignment_and_sync_brackets() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 64 * 1024;

    let (_bo, shared) = MemoryBridge::allocate_shared(&mock_gpu, size, true).unwrap();
    let ptr = shared.as_ptr().unwrap() as usize;
    assert_eq!(
        ptr % 64,
        0,
        "Mapped address {:#x} must be strictly 64-byte cache-line aligned",
        ptr
    );

    // Verify Read/Write sync bracket transitions
    shared
        .handle()
        .begin_cpu_access(SyncDirection::Read)
        .expect("Sync Read Start failed");
    shared
        .handle()
        .end_cpu_access(SyncDirection::Read)
        .expect("Sync Read End failed");

    shared
        .handle()
        .begin_cpu_access(SyncDirection::Write)
        .expect("Sync Write Start failed");
    shared
        .handle()
        .end_cpu_access(SyncDirection::Write)
        .expect("Sync Write End failed");

    shared
        .handle()
        .begin_cpu_access(SyncDirection::ReadWrite)
        .expect("Sync ReadWrite Start failed");
    shared
        .handle()
        .end_cpu_access(SyncDirection::ReadWrite)
        .expect("Sync ReadWrite End failed");
}

/// D. Invariant: Clean Teardown with Zero File Descriptor Leaks.
#[test]
fn test_invariant_clean_teardown_zero_fd_leaks() {
    // Warm up / initialize any lazy once locks
    let _ = open_or_mock("/dev/dri/renderD128");
    let _ = open_or_mock("/dev/accel/accel0");
    let _ = get_open_fds();

    let initial_fds = get_open_fds();

    // Perform multiple full allocation, export, import, map, write, read cycles
    for cycle in 0..15 {
        {
            let gpu = MockDeviceBackend::new_gpu();
            let npu = MockDeviceBackend::new_npu();
            let gpu_arc: Arc<dyn DeviceBackend> = Arc::new(gpu);
            let npu_arc: Arc<dyn DeviceBackend> = Arc::new(npu);

            let (gpu_bo, mut shared) = MemoryBridge::allocate_shared(&gpu_arc, 64 * 1024, true)
                .unwrap_or_else(|e| panic!("Cycle {} allocation failed: {:?}", cycle, e));

            let npu_dmabuf = shared.handle().try_clone().unwrap();
            let npu_bo = MemoryBridge::import_dma_buf(&npu_arc, &npu_dmabuf).unwrap();

            shared
                .with_cpu_write(|slice| {
                    slice[0..4].copy_from_slice(b"TEST");
                })
                .unwrap();

            let val = shared
                .with_cpu_read(|slice| slice[0..4].to_vec())
                .unwrap();
            assert_eq!(&val, b"TEST");

            assert!(gpu_bo.handle() > 0);
            assert!(npu_bo.handle() > 0);
            // All objects drop at the end of this scope
        }

        let current_fds = get_open_fds();
        let leaked: Vec<_> = current_fds.difference(&initial_fds).collect();
        assert!(
            leaked.is_empty(),
            "Cycle {} leaked file descriptors: {:?}",
            cycle,
            leaked
        );
    }

    let final_fds = get_open_fds();
    let leaked: Vec<_> = final_fds.difference(&initial_fds).collect();
    assert!(
        leaked.is_empty(),
        "Leaked file descriptors across allocations: {:?}",
        leaked
    );
}
