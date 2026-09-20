// SPDX-License-Identifier: Apache-2.0
//! Adversarial Empirical Test Harness for Milestone 1 Invariants.
//!
//! Evaluates:
//! 1. Zero host memcpy during cross-accelerator buffer handoff (heap tracking & O(1) latency).
//! 2. Watermark mutations across offsets (0, 64, 4096, 65536, size-64) and large buffers (1MB, 16MB, 64MB).
//! 3. CPU cache invalidation/flush via DMA_BUF_IOCTL_SYNC brackets on physical silicon & mock shims.
//! 4. Concurrency, leak-free teardown, and alignment boundaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::time::Instant;

use zero_copy_model_runner::backend::{
    open_or_mock, DeviceBackend, DeviceType, MockDeviceBackend, PhysicalDeviceBackend,
};
use zero_copy_model_runner::memory::{MemoryBridge, MemoryError, SharedBuffer, SyncDirection};

use std::cell::Cell;

// -----------------------------------------------------------------------------
// Thread-Local Tracking Allocator to Empirically Measure Host Heap Allocations
// -----------------------------------------------------------------------------
struct TrackingAllocator;

thread_local! {
    static TRACKING_ENABLED: Cell<bool> = const { Cell::new(false) };
    static THREAD_ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // If thread-local tracking is enabled for this thread, record allocated size
        TRACKING_ENABLED.with(|enabled| {
            if enabled.get() {
                THREAD_ALLOCATED_BYTES.with(|b| {
                    b.set(b.get() + layout.size());
                });
            }
        });
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn start_thread_heap_tracking() {
    TRACKING_ENABLED.with(|e| e.set(true));
    THREAD_ALLOCATED_BYTES.with(|b| b.set(0));
}

fn stop_thread_heap_tracking() -> usize {
    TRACKING_ENABLED.with(|e| e.set(false));
    THREAD_ALLOCATED_BYTES.with(|b| b.get())
}

// -----------------------------------------------------------------------------
// 1. Empirical Proof: Zero Host Memcpy during Cross-Accelerator Handoff
// -----------------------------------------------------------------------------

#[test]
fn test_empirical_zero_host_memcpy_heap_tracking() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());

    let test_sizes = [
        1 * 1024 * 1024,      // 1 MB
        16 * 1024 * 1024,     // 16 MB
        64 * 1024 * 1024,     // 64 MB
    ];

    for &size in &test_sizes {
        // Allocate GEM BO
        let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();

        // Measure heap allocations strictly during export & import handoff for this thread
        start_thread_heap_tracking();
        let dmabuf = gpu_bo.export_prime_fd().unwrap();
        let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &dmabuf).unwrap();
        let delta = stop_thread_heap_tracking();

        println!(
            "[Zero-Copy Mock Test] Buffer size: {} MB ({} bytes), Host Heap Allocated during handoff: {} bytes",
            size / (1024 * 1024),
            size,
            delta
        );

        // If host memcpy occurred, delta would be >= size (e.g. 64 MB = 67,108,864 bytes).
        // For true zero-copy, only lightweight Rust structs (Arc, NpuBufferObject) are allocated (< 1024 bytes).
        assert!(
            delta < 1024,
            "Host heap allocation during {} MB handoff was {} bytes (must be < 1024 bytes for zero-copy)",
            size / (1024 * 1024),
            delta
        );

        assert_eq!(npu_bo.size(), size);
    }

    // Also verify physical silicon if present
    if let (Ok(gpu), Ok(npu)) = (
        PhysicalDeviceBackend::open("/dev/dri/renderD128", DeviceType::Gpu),
        PhysicalDeviceBackend::open("/dev/accel/accel0", DeviceType::Npu),
    ) {
        let gpu_arc: Arc<dyn DeviceBackend> = Arc::new(gpu);
        let npu_arc: Arc<dyn DeviceBackend> = Arc::new(npu);
        for &size in &test_sizes {
            let gpu_bo = MemoryBridge::allocate_gem_bo(&gpu_arc, size, 4096).unwrap();

            start_thread_heap_tracking();
            let dmabuf = gpu_bo.export_prime_fd().unwrap();
            let npu_bo = MemoryBridge::import_dma_buf(&npu_arc, &dmabuf).unwrap();
            let delta = stop_thread_heap_tracking();

            println!(
                "[Zero-Copy Physical Silicon Test] Buffer size: {} MB, Host Heap Allocated during handoff: {} bytes",
                size / (1024 * 1024),
                delta
            );

            assert!(
                delta < 1024,
                "Physical silicon host heap allocation during {} MB handoff was {} bytes (must be < 1024 bytes)",
                size / (1024 * 1024),
                delta
            );
            assert_eq!(npu_bo.size(), size);
        }
    }
}

#[test]
fn test_empirical_handoff_latency_o1_scaling() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());

    let sizes = [
        1 * 1024 * 1024,   // 1 MB
        16 * 1024 * 1024,  // 16 MB
        64 * 1024 * 1024,  // 64 MB
    ];

    let mut latencies_us = Vec::new();

    for &size in &sizes {
        let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();

        let start = Instant::now();
        let dmabuf = gpu_bo.export_prime_fd().unwrap();
        let _npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &dmabuf).unwrap();
        let elapsed = start.elapsed();

        let us = elapsed.as_micros();
        latencies_us.push(us);
        println!(
            "[Latency Test] Buffer size: {} MB -> Handoff time: {} us",
            size / (1024 * 1024),
            us
        );
    }

    // If host memcpy were involved, 64MB at ~20GB/s would take ~3200 us, 64x slower than 1MB (~50 us).
    // In zero-copy, all operations are simple kernel handle duplications completing in < 500 us regardless of size.
    for (i, &lat) in latencies_us.iter().enumerate() {
        assert!(
            lat < 2000,
            "Handoff latency for size {} MB exceeded 2000 us (took {} us)",
            sizes[i] / (1024 * 1024),
            lat
        );
    }
}

// -----------------------------------------------------------------------------
// 2. Stress Test: Watermark Mutations Across Multiple Offsets & Large Buffers
// -----------------------------------------------------------------------------

#[test]
fn test_stress_watermark_mutations_mock() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());

    let test_sizes = [
        1 * 1024 * 1024,      // 1 MB
        16 * 1024 * 1024,     // 16 MB
        64 * 1024 * 1024,     // 64 MB
    ];

    for &size in &test_sizes {
        let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
        let gpu_dmabuf = gpu_bo.export_prime_fd().unwrap();
        let npu_dmabuf = gpu_dmabuf.try_clone().unwrap();

        let _npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &npu_dmabuf).unwrap();

        let mut gpu_view = SharedBuffer::new(gpu_dmabuf, true).unwrap();
        let mut npu_view = SharedBuffer::new(npu_dmabuf, true).unwrap();

        assert_eq!(gpu_view.len(), size);
        assert_eq!(npu_view.len(), size);
        assert_eq!(gpu_view.as_ptr().unwrap() as usize % 64, 0);
        assert_eq!(npu_view.as_ptr().unwrap() as usize % 64, 0);

        // Required offsets: 0, 64, 4096, 65536, plus large buffer edge offsets
        let test_offsets = [0, 64, 4096, 65536, size - 64, size - 8];

        for &offset in &test_offsets {
            let watermark_gpu = 0xA1B2C3D4_E5F60718u64 ^ (offset as u64);
            let watermark_npu = 0xFEDCBA98_76543210u64 ^ (offset as u64);

            // 1. GPU writes watermark
            gpu_view
                .with_cpu_write(|slice| {
                    slice[offset..offset + 8].copy_from_slice(&watermark_gpu.to_le_bytes());
                })
                .unwrap();

            // 2. NPU reads watermark (immediate mutual visibility)
            let read_npu = npu_view
                .with_cpu_read(|slice| {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&slice[offset..offset + 8]);
                    u64::from_le_bytes(buf)
                })
                .unwrap();
            assert_eq!(
                read_npu, watermark_gpu,
                "Size {} MB: NPU failed to read GPU watermark at offset {}",
                size / (1024 * 1024),
                offset
            );

            // 3. NPU overwrites watermark
            npu_view
                .with_cpu_write(|slice| {
                    slice[offset..offset + 8].copy_from_slice(&watermark_npu.to_le_bytes());
                })
                .unwrap();

            // 4. GPU reads back mutated watermark
            let read_back_gpu = gpu_view
                .with_cpu_read(|slice| {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&slice[offset..offset + 8]);
                    u64::from_le_bytes(buf)
                })
                .unwrap();
            assert_eq!(
                read_back_gpu, watermark_npu,
                "Size {} MB: GPU failed to read mutated watermark at offset {}",
                size / (1024 * 1024),
                offset
            );
        }

        println!(
            "[Stress Test Mock Passed] Size: {} MB, all {} offsets verified successfully",
            size / (1024 * 1024),
            test_offsets.len()
        );
    }
}

#[test]
fn test_stress_watermark_mutations_physical_silicon() {
    let gpu_res = PhysicalDeviceBackend::open("/dev/dri/renderD128", DeviceType::Gpu);
    let npu_res = PhysicalDeviceBackend::open("/dev/accel/accel0", DeviceType::Npu);

    if let (Ok(gpu), Ok(npu)) = (gpu_res, npu_res) {
        println!("[Physical Silicon] Testing real AMD Ryzen AI APU hardware...");
        let gpu_arc: Arc<dyn DeviceBackend> = Arc::new(gpu);
        let npu_arc: Arc<dyn DeviceBackend> = Arc::new(npu);

        let test_sizes = [
            1 * 1024 * 1024,      // 1 MB
            16 * 1024 * 1024,     // 16 MB
            64 * 1024 * 1024,     // 64 MB
        ];

        for &size in &test_sizes {
            let gpu_bo = MemoryBridge::allocate_gem_bo(&gpu_arc, size, 4096)
                .expect("Physical AMDGPU GTT allocation failed");
            let gpu_dmabuf = gpu_bo
                .export_prime_fd()
                .expect("Physical PRIME export failed");
            let npu_dmabuf = gpu_dmabuf.try_clone().expect("PRIME FD clone failed");

            let npu_bo = MemoryBridge::import_dma_buf(&npu_arc, &npu_dmabuf)
                .expect("Physical AMDXDNA PRIME import failed");
            assert!(npu_bo.handle() > 0);
            assert_eq!(npu_bo.size(), size);

            let mut gpu_view = SharedBuffer::new(gpu_dmabuf, true).expect("GPU mmap failed");
            let mut npu_view = SharedBuffer::new(npu_dmabuf, true).expect("NPU mmap failed");

            assert_eq!(gpu_view.as_ptr().unwrap() as usize % 64, 0);
            assert_eq!(npu_view.as_ptr().unwrap() as usize % 64, 0);

            let test_offsets = [0, 64, 4096, 65536, size - 64, size - 8];

            for &offset in &test_offsets {
                let sentinel_gpu = 0x55AA_33CC_77EE_1122u64 ^ (offset as u64);
                let sentinel_npu = 0xAA55_CC33_EE77_2211u64 ^ (offset as u64);

                // GPU writes
                gpu_view
                    .with_cpu_write(|slice| {
                        slice[offset..offset + 8].copy_from_slice(&sentinel_gpu.to_le_bytes());
                    })
                    .unwrap();

                // NPU reads
                let read_npu = npu_view
                    .with_cpu_read(|slice| {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slice[offset..offset + 8]);
                        u64::from_le_bytes(b)
                    })
                    .unwrap();
                assert_eq!(
                    read_npu, sentinel_gpu,
                    "Physical silicon mismatch at offset {} (size {} MB)",
                    offset,
                    size / (1024 * 1024)
                );

                // NPU overwrites
                npu_view
                    .with_cpu_write(|slice| {
                        slice[offset..offset + 8].copy_from_slice(&sentinel_npu.to_le_bytes());
                    })
                    .unwrap();

                // GPU reads back
                let read_back_gpu = gpu_view
                    .with_cpu_read(|slice| {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slice[offset..offset + 8]);
                        u64::from_le_bytes(b)
                    })
                    .unwrap();
                assert_eq!(
                    read_back_gpu, sentinel_npu,
                    "Physical silicon read back mismatch at offset {} (size {} MB)",
                    offset,
                    size / (1024 * 1024)
                );
            }

            println!(
                "[Physical Silicon Passed] Size: {} MB (all {} offsets verified)",
                size / (1024 * 1024),
                test_offsets.len()
            );
        }
    } else {
        println!("[Notice] Physical silicon nodes not present; test skipped.");
    }
}

// -----------------------------------------------------------------------------
// 3. CPU Cache Invalidation & Flush via DMA_BUF_IOCTL_SYNC Brackets
// -----------------------------------------------------------------------------

#[test]
fn test_cache_invalidation_sync_brackets_physical_and_mock() {
    let backend = open_or_mock("/dev/dri/renderD128").unwrap();
    let size = 1 * 1024 * 1024; // 1 MB

    let (_gpu_bo, mut shared) = MemoryBridge::allocate_shared(&backend, size, true).unwrap();

    // 1. Validate full range of sync directions
    for dir in [SyncDirection::Read, SyncDirection::Write, SyncDirection::ReadWrite] {
        shared.handle().begin_cpu_access(dir).expect("begin_cpu_access failed");
        shared.handle().end_cpu_access(dir).expect("end_cpu_access failed");
    }

    // 2. Validate that invalid file descriptors are rejected by begin_cpu_access
    let invalid_handle = unsafe { zero_copy_model_runner::memory::DmaBufHandle::from_raw_fd_unchecked(-1, 4096) };
    let res = invalid_handle.begin_cpu_access(SyncDirection::Read);
    assert!(
        res.is_err(),
        "Invalid file descriptor -1 must be rejected by begin_cpu_access"
    );

    // 3. Validate closure error propagation and bracket execution safety
    let res: Result<(), MemoryError> = shared.with_cpu_write(|slice| {
        slice[0] = 0xAA;
    });
    assert!(res.is_ok());

    let val = shared.with_cpu_read(|slice| slice[0]).unwrap();
    assert_eq!(val, 0xAA);
}

// -----------------------------------------------------------------------------
// 4. Edge Cases: Alignment Boundaries, Extremes, Concurrency
// -----------------------------------------------------------------------------

#[test]
fn test_alignment_boundaries_and_rejections() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // Misaligned allocations must be strictly rejected
    for invalid_size in [1, 7, 63, 65, 127, 4095] {
        let res = MemoryBridge::allocate_gem_bo(&mock_gpu, invalid_size, 64);
        assert!(
            res.is_err(),
            "Size {} is not 64-byte aligned and must fail",
            invalid_size
        );
    }

    // Size 0 allocation must be rejected
    let res_zero = MemoryBridge::allocate_gem_bo(&mock_gpu, 0, 64);
    assert!(res_zero.is_err(), "Size 0 must be rejected");

    // Valid 64-byte multiples must succeed
    for valid_size in [64, 128, 4096, 65536] {
        let res = MemoryBridge::allocate_gem_bo(&mock_gpu, valid_size, 64);
        assert!(
            res.is_ok(),
            "Valid size {} failed: {:?}",
            valid_size,
            res.err()
        );
    }
}

#[test]
fn test_concurrent_multithreaded_shared_buffer_access() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 16 * 1024 * 1024; // 16 MB

    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
    let gpu_dmabuf = gpu_bo.export_prime_fd().unwrap();

    let npu_dmabuf = gpu_dmabuf.try_clone().unwrap();

    let shared_gpu = Arc::new(parking_lot_or_std_mutex(SharedBuffer::new(gpu_dmabuf, true).unwrap()));
    let shared_npu = Arc::new(parking_lot_or_std_mutex(SharedBuffer::new(npu_dmabuf, true).unwrap()));

    let num_threads = 8;
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let sg = Arc::clone(&shared_gpu);
        let sn = Arc::clone(&shared_npu);
        let handle = std::thread::spawn(move || {
            let offset = t * 65536; // Each thread operates on independent 64 KiB block
            let thread_magic = 0xC0FFEE_00000000u64 | (t as u64);

            // Writer: GPU view
            {
                let mut guard = sg.lock().unwrap();
                guard
                    .with_cpu_write(|slice| {
                        slice[offset..offset + 8].copy_from_slice(&thread_magic.to_le_bytes());
                    })
                    .unwrap();
            }

            // Reader: NPU view
            {
                let guard = sn.lock().unwrap();
                let read = guard
                    .with_cpu_read(|slice| {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slice[offset..offset + 8]);
                        u64::from_le_bytes(b)
                    })
                    .unwrap();
                assert_eq!(read, thread_magic);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }
}

fn parking_lot_or_std_mutex<T>(val: T) -> std::sync::Mutex<T> {
    std::sync::Mutex::new(val)
}

// -----------------------------------------------------------------------------
// 5. Full Buffer Stream PRNG Watermark Verification (16 MB = 4,194,304 u32s)
// -----------------------------------------------------------------------------

#[test]
fn test_full_buffer_prng_watermark_verification() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());

    let size = 16 * 1024 * 1024; // 16 MB = 4,194,304 u32 elements
    let (gpu_bo, mut gpu_view) = MemoryBridge::allocate_shared(&mock_gpu, size, true).unwrap();
    let npu_dmabuf = gpu_bo.export_prime_fd().unwrap();
    let _npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &npu_dmabuf).unwrap();
    let mut npu_view = SharedBuffer::new(npu_dmabuf, true).unwrap();

    const SEED_GPU: u32 = 0xCAFE_BABE;
    const SEED_NPU: u32 = 0xDEAD_1234;

    // 1. GPU writes stream of pseudo-random u32 values across entire 16 MB
    gpu_view
        .with_cpu_write(|slice| {
            for (i, chunk) in slice.chunks_exact_mut(4).enumerate() {
                let val = (i as u32).wrapping_mul(1103515245).wrapping_add(12345) ^ SEED_GPU;
                chunk.copy_from_slice(&val.to_le_bytes());
            }
        })
        .unwrap();

    // 2. NPU verifies all 4,194,304 values match expected stream
    let npu_verified = npu_view
        .with_cpu_read(|slice| {
            for (i, chunk) in slice.chunks_exact(4).enumerate() {
                let expected = (i as u32).wrapping_mul(1103515245).wrapping_add(12345) ^ SEED_GPU;
                let actual = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if actual != expected {
                    return false;
                }
            }
            true
        })
        .unwrap();
    assert!(npu_verified, "NPU failed full 16 MB stream verification");

    // 3. NPU overwrites stream
    npu_view
        .with_cpu_write(|slice| {
            for (i, chunk) in slice.chunks_exact_mut(4).enumerate() {
                let val = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223) ^ SEED_NPU;
                chunk.copy_from_slice(&val.to_le_bytes());
            }
        })
        .unwrap();

    // 4. GPU verifies overwritten stream
    let gpu_verified = gpu_view
        .with_cpu_read(|slice| {
            for (i, chunk) in slice.chunks_exact(4).enumerate() {
                let expected = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223) ^ SEED_NPU;
                let actual = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if actual != expected {
                    return false;
                }
            }
            true
        })
        .unwrap();
    assert!(gpu_verified, "GPU failed full 16 MB overwrite verification");
    println!("[Full Stream PRNG Passed] 4,194,304 u32 elements (16 MB) verified across GPU & NPU views");
}

// -----------------------------------------------------------------------------
// 6. Stress Test: Rapid Multi-Buffer 64 MB Allocation & Teardown Lifecycle
// -----------------------------------------------------------------------------

#[test]
fn test_rapid_multi_buffer_64mb_lifecycle() {
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());

    let size = 64 * 1024 * 1024; // 64 MB each
    const NUM_ITERATIONS: usize = 5;

    for iter in 0..NUM_ITERATIONS {
        let (gpu_bo, mut shared_gpu) = MemoryBridge::allocate_shared(&mock_gpu, size, true).unwrap();
        let npu_dmabuf = gpu_bo.export_prime_fd().unwrap();
        let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &npu_dmabuf).unwrap();
        let shared_npu = SharedBuffer::new(npu_dmabuf, true).unwrap();

        // Write marker at end of buffer
        let marker = 0xBAAD_F00D_00000000u64 | (iter as u64);
        shared_gpu
            .with_cpu_write(|slice| {
                slice[size - 8..size].copy_from_slice(&marker.to_le_bytes());
            })
            .unwrap();

        let val = shared_npu
            .with_cpu_read(|slice| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&slice[size - 8..size]);
                u64::from_le_bytes(b)
            })
            .unwrap();
        assert_eq!(val, marker);

        assert!(gpu_bo.handle() > 0);
        assert!(npu_bo.handle() > 0);
        // Clean drop at end of iteration
    }
    println!("[Rapid 64 MB Lifecycle Passed] 5 iterations x 64 MB = 320 MB total cycled cleanly");
}

// -----------------------------------------------------------------------------
// 7. Kernel UAPI dma_buf_sync Error Handling & Flag Validation
// -----------------------------------------------------------------------------

#[test]
fn test_dma_buf_sync_raw_ioctl_invalid_flags() {
    // If physical silicon is present, verify kernel rejects invalid flags with EINVAL
    if let Ok(gpu) = PhysicalDeviceBackend::open("/dev/dri/renderD128", DeviceType::Gpu) {
        let gpu_arc: Arc<dyn DeviceBackend> = Arc::new(gpu);
        let gpu_bo = MemoryBridge::allocate_gem_bo(&gpu_arc, 64 * 1024, 4096).unwrap();
        let dmabuf = gpu_bo.export_prime_fd().unwrap();

        // 1. Valid flags: DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ
        let mut valid_sync = zero_copy_model_runner::uapi::dma_buf_sync {
            flags: zero_copy_model_runner::uapi::DMA_BUF_SYNC_START
                | zero_copy_model_runner::uapi::DMA_BUF_SYNC_READ,
        };
        let ret = unsafe { zero_copy_model_runner::uapi::dma_buf_ioctl_sync(dmabuf.as_raw_fd(), &mut valid_sync) };
        assert!(ret.is_ok(), "Valid dma_buf_sync failed: {:?}", ret);

        // 2. Invalid flags: e.g. bit 31 set (not part of VALID_FLAGS_MASK)
        let mut invalid_sync = zero_copy_model_runner::uapi::dma_buf_sync {
            flags: 0x8000_0000_0000_0000u64,
        };
        let ret_invalid = unsafe { zero_copy_model_runner::uapi::dma_buf_ioctl_sync(dmabuf.as_raw_fd(), &mut invalid_sync) };
        assert!(
            ret_invalid.is_err(),
            "Kernel must reject invalid dma_buf_sync flags"
        );
        match ret_invalid {
            Err(nix::Error::EINVAL) => {}
            other => panic!("Expected EINVAL for invalid flags, got {:?}", other),
        }
        println!("[DMA_BUF_IOCTL_SYNC Passed] Physical kernel correctly returned EINVAL on invalid flags");
    }
}

