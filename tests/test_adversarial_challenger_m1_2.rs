// SPDX-License-Identifier: Apache-2.0
//! Adversarial Challenge Test Suite: Resource Management, RAII Lifetimes, & Concurrency.
//!
//! Conducted by Challenger 2 (challenger_m1_2) for Milestone 1.
//!
//! Objectives:
//! 1. Stress test repeated allocation/export/import/teardown cycles (50+ iterations),
//!    checking /proc/self/fd to guarantee zero file descriptor leaks.
//! 2. Verify that unmapped pointers are not dereferenced and no memory leaks occur in /proc/self/maps.
//! 3. Test edge case lifecycles:
//!    - Drop GPU buffer while NPU buffer is alive.
//!    - Drop NPU buffer while GPU buffer is alive (reverse order).
//!    - Multi-generation try_clone on DmaBufHandle.
//! 4. High-contention multi-threaded concurrency stress test (320 cycles across 16 threads).
//! 5. Boundary conditions, zero-size, and unaligned allocations.

use std::collections::BTreeSet;
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;

use zero_copy_model_runner::backend::{
    open_or_mock, DeviceBackend, DeviceType, MockDeviceBackend, PhysicalDeviceBackend,
};
use zero_copy_model_runner::memory::{
    DmaBufHandle, MemoryBridge, MemoryError, SharedBuffer,
};

/// Global test mutex to serialize tests that observe process-wide `/proc/self/fd`
/// and `/proc/self/maps` state against parallel test runners.
static PROC_INSPECTION_MUTEX: Mutex<()> = Mutex::new(());

fn acquire_lock() -> std::sync::MutexGuard<'static, ()> {
    PROC_INSPECTION_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ============================================================================
// Helper Utilities for Invariant Inspection
// ============================================================================

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

/// Parsed Virtual Memory Area from `/proc/self/maps`.
#[derive(Debug, Clone)]
struct VmaRegion {
    start: usize,
    end: usize,
    path: String,
}

/// Parse all virtual memory mappings from `/proc/self/maps`.
fn get_maps() -> Vec<VmaRegion> {
    let content = fs::read_to_string("/proc/self/maps")
        .expect("Failed to read /proc/self/maps for empirical validation");
    let mut regions = Vec::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let addrs: Vec<&str> = parts[0].split('-').collect();
        if addrs.len() != 2 {
            continue;
        }
        if let (Ok(start), Ok(end)) = (
            usize::from_str_radix(addrs[0], 16),
            usize::from_str_radix(addrs[1], 16),
        ) {
            let path = parts.get(5).unwrap_or(&"").to_string();
            regions.push(VmaRegion {
                start,
                end,
                path,
            });
        }
    }
    regions
}

/// Count active memfd / dma-buf mappings in `/proc/self/maps`.
fn count_accelerator_mappings() -> usize {
    get_maps()
        .into_iter()
        .filter(|r| {
            r.path.contains("memfd:mock_amdgpu")
                || r.path.contains("dma_buf")
                || r.path.contains("renderD128")
                || r.path.contains("accel0")
        })
        .count()
}

// Warm up once locks and directory readers so initial fd set is stable.
fn warmup_runtime() {
    let _ = open_or_mock("/dev/dri/renderD128");
    let _ = open_or_mock("/dev/accel/accel0");
    let _ = get_open_fds();
    let _ = get_maps();
}

// ============================================================================
// Objective 1: Repeated Allocation/Export/Import/Teardown (50+ iterations)
// ============================================================================

#[test]
fn test_adversarial_fd_leak_stress_mock_75_iterations() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_fds = get_open_fds();

    let test_sizes = [
        64,             // Minimum 64-byte alignment
        4 * 1024,       // 4 KiB
        64 * 1024,      // 64 KiB
        1024 * 1024,    // 1 MiB
        16 * 1024 * 1024, // 16 MiB
    ];

    for iteration in 0..75 {
        let size = test_sizes[iteration % test_sizes.len()];
        {
            let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
            let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());

            let (gpu_bo, mut shared) = MemoryBridge::allocate_shared(&mock_gpu, size, true)
                .unwrap_or_else(|e| panic!("Iter {} alloc failed: {:?}", iteration, e));

            let cloned_dmabuf = shared.handle().try_clone().unwrap();
            let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &cloned_dmabuf)
                .unwrap_or_else(|e| panic!("Iter {} import failed: {:?}", iteration, e));

            assert!(gpu_bo.handle() > 0);
            assert!(npu_bo.handle() > 0);

            // Mutate and verify
            shared
                .with_cpu_write(|slice| {
                    slice[0] = (iteration % 255) as u8;
                    slice[size - 1] = 0xAA;
                })
                .unwrap();

            shared
                .with_cpu_read(|slice| {
                    assert_eq!(slice[0], (iteration % 255) as u8);
                    assert_eq!(slice[size - 1], 0xAA);
                })
                .unwrap();

            // Scope exit triggers Drop for gpu_bo, npu_bo, cloned_dmabuf, shared, mock_gpu, mock_npu
        }

        let current_fds = get_open_fds();
        let leaked: Vec<_> = current_fds.difference(&initial_fds).collect();
        assert!(
            leaked.is_empty(),
            "Mock Iteration {} leaked file descriptors: {:?}",
            iteration,
            leaked
        );
    }

    let final_fds = get_open_fds();
    assert_eq!(
        initial_fds, final_fds,
        "File descriptor set must remain strictly invariant after 75 mock iterations"
    );
}

#[test]
fn test_adversarial_fd_leak_stress_physical_75_iterations() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_fds = get_open_fds();

    // Only run if physical silicon is present and accessible
    if let (Ok(gpu), Ok(npu)) = (
        PhysicalDeviceBackend::open("/dev/dri/renderD128", DeviceType::Gpu),
        PhysicalDeviceBackend::open("/dev/accel/accel0", DeviceType::Npu),
    ) {
        let gpu_arc: Arc<dyn DeviceBackend> = Arc::new(gpu);
        let npu_arc: Arc<dyn DeviceBackend> = Arc::new(npu);

        let test_sizes = [
            64 * 1024,        // 64 KiB
            128 * 1024,       // 128 KiB
            1024 * 1024,      // 1 MiB
            4 * 1024 * 1024,  // 4 MiB
        ];

        for iteration in 0..75 {
            let size = test_sizes[iteration % test_sizes.len()];
            {
                let (gpu_bo, mut shared) = MemoryBridge::allocate_shared(&gpu_arc, size, true)
                    .unwrap_or_else(|e| panic!("Physical iter {} alloc failed: {:?}", iteration, e));

                let cloned_dmabuf = shared.handle().try_clone().unwrap();
                let npu_bo = MemoryBridge::import_dma_buf(&npu_arc, &cloned_dmabuf)
                    .unwrap_or_else(|e| panic!("Physical iter {} import failed: {:?}", iteration, e));

                assert!(gpu_bo.handle() > 0);
                assert!(npu_bo.handle() > 0);

                shared
                    .with_cpu_write(|slice| {
                        slice[0] = (iteration & 0xFF) as u8;
                        slice[63] = 0x55;
                    })
                    .unwrap();

                shared
                    .with_cpu_read(|slice| {
                        assert_eq!(slice[0], (iteration & 0xFF) as u8);
                        assert_eq!(slice[63], 0x55);
                    })
                    .unwrap();
            }

            let current_fds = get_open_fds();
            // During iteration, gpu_arc and npu_arc are held, so current_fds will have those 2 extra descriptors.
            let leaked: Vec<_> = current_fds.difference(&initial_fds).collect();
            // Exactly the 2 device node fds (renderD128 and accel0) should be open beyond initial_fds.
            assert_eq!(
                leaked.len(),
                2,
                "Physical Iteration {} leaked unexpected file descriptors: {:?}",
                iteration,
                leaked
            );
        }

        drop(gpu_arc);
        drop(npu_arc);

        let final_fds = get_open_fds();
        assert_eq!(
            initial_fds, final_fds,
            "Physical silicon file descriptors strictly invariant after 75 iterations"
        );
    }
}

// ============================================================================
// Objective 2: /proc/self/maps Leak Verification & Unmapped Safety
// ============================================================================

#[test]
fn test_adversarial_proc_self_maps_leak_detection_60_iterations() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_accel_mappings = count_accelerator_mappings();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 128 * 1024; // 128 KiB

    for iteration in 0..60 {
        let ptr_val: usize;
        {
            let (_gpu_bo, shared) = MemoryBridge::allocate_shared(&mock_gpu, size, true).unwrap();
            let ptr = shared.as_ptr().expect("Must have CPU mapping");
            ptr_val = ptr as usize;

            // 1. Verify that while alive, the exact virtual address range appears in /proc/self/maps
            let maps = get_maps();
            let found_active = maps.iter().any(|r| r.start == ptr_val && r.end == ptr_val + size);
            assert!(
                found_active,
                "Iteration {}: Active mmap {:#x}-{:#x} must exist in /proc/self/maps",
                iteration,
                ptr_val,
                ptr_val + size
            );

            // 2. Verify 64-byte alignment
            assert_eq!(ptr_val % 64, 0);
        }

        // 3. Immediately after Drop, verify the exact virtual address range is unmapped
        let maps_after = get_maps();
        let found_after = maps_after
            .iter()
            .any(|r| r.start == ptr_val && r.end == ptr_val + size);
        assert!(
            !found_after,
            "Iteration {}: Mmap {:#x}-{:#x} was NOT cleaned up from /proc/self/maps upon Drop!",
            iteration,
            ptr_val,
            ptr_val + size
        );
    }

    drop(mock_gpu);

    let final_accel_mappings = count_accelerator_mappings();
    assert_eq!(
        initial_accel_mappings, final_accel_mappings,
        "Total accelerator mappings in /proc/self/maps must return to baseline after 60 iterations"
    );
}

#[test]
fn test_adversarial_unmapped_pointer_safety() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_maps = get_maps().len();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 64 * 1024;

    // Allocate unmapped SharedBuffer (map_cpu = false)
    let (_gpu_bo, mut unmapped) = MemoryBridge::allocate_shared(&mock_gpu, size, false).unwrap();

    // 1. Verify pointers are None
    assert!(unmapped.as_ptr().is_none());
    assert!(unmapped.as_mut_ptr().is_none());

    // 2. Verify accessing CPU read/write safely errors with InvalidState without dereferencing
    let read_result = unmapped.with_cpu_read(|_| ());
    match read_result {
        Err(MemoryError::InvalidState(msg)) => {
            assert!(msg.contains("not mapped"));
        }
        other => panic!("Expected InvalidState error, got {:?}", other),
    }

    let write_result = unmapped.with_cpu_write(|_| ());
    match write_result {
        Err(MemoryError::InvalidState(msg)) => {
            assert!(msg.contains("not mapped"));
        }
        other => panic!("Expected InvalidState error, got {:?}", other),
    }

    drop(unmapped);
    drop(_gpu_bo);
    drop(mock_gpu);

    let final_maps = get_maps().len();
    assert_eq!(
        initial_maps, final_maps,
        "Unmapped SharedBuffer must not leak entries in /proc/self/maps"
    );
}

// ============================================================================
// Objective 3: Edge Case Lifecycles & Drop Orders
// ============================================================================

#[test]
fn test_adversarial_edge_lifecycle_drop_gpu_while_npu_alive() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_fds = get_open_fds();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());
    let size = 64 * 1024;

    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
    let dmabuf = gpu_bo.export_prime_fd().unwrap();
    let npu_dmabuf = dmabuf.try_clone().unwrap();

    let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &npu_dmabuf).unwrap();
    let mut shared = SharedBuffer::new(dmabuf, true).unwrap();

    // Write initial watermark
    shared
        .with_cpu_write(|slice| {
            slice[0..8].copy_from_slice(b"PRE_DROP");
        })
        .unwrap();

    // DROP GPU BO EXPLICITLY while NPU BO and SharedBuffer are still alive!
    drop(gpu_bo);

    // NPU BO must remain fully valid and accessible
    assert!(npu_bo.handle() > 0);
    assert_eq!(npu_bo.size(), size);

    // SharedBuffer must continue functioning seamlessly
    let val = shared
        .with_cpu_read(|slice| slice[0..8].to_vec())
        .unwrap();
    assert_eq!(&val, b"PRE_DROP");

    shared
        .with_cpu_write(|slice| {
            slice[0..8].copy_from_slice(b"PST_DROP");
        })
        .unwrap();

    let val2 = shared
        .with_cpu_read(|slice| slice[0..8].to_vec())
        .unwrap();
    assert_eq!(&val2, b"PST_DROP");

    // Drop remaining objects and backends
    drop(shared);
    drop(npu_dmabuf);
    drop(npu_bo);
    drop(mock_gpu);
    drop(mock_npu);

    let final_fds = get_open_fds();
    assert_eq!(
        initial_fds, final_fds,
        "Zero FD leaks when GPU BO dropped before NPU BO"
    );
}

#[test]
fn test_adversarial_edge_lifecycle_drop_npu_while_gpu_alive() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_fds = get_open_fds();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());
    let size = 64 * 1024;

    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
    let dmabuf = gpu_bo.export_prime_fd().unwrap();
    let npu_dmabuf = dmabuf.try_clone().unwrap();

    let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &npu_dmabuf).unwrap();
    let mut shared = SharedBuffer::new(dmabuf, true).unwrap();

    shared
        .with_cpu_write(|slice| {
            slice[0..8].copy_from_slice(b"REV_INIT");
        })
        .unwrap();

    // DROP NPU BO FIRST while GPU BO and SharedBuffer are still alive!
    drop(npu_bo);
    drop(npu_dmabuf);

    // GPU BO must remain valid and operational
    assert!(gpu_bo.handle() > 0);
    assert_eq!(gpu_bo.size(), size);

    // SharedBuffer must continue functioning
    let val = shared
        .with_cpu_read(|slice| slice[0..8].to_vec())
        .unwrap();
    assert_eq!(&val, b"REV_INIT");

    shared
        .with_cpu_write(|slice| {
            slice[0..8].copy_from_slice(b"REV_CONT");
        })
        .unwrap();

    assert_eq!(
        shared.with_cpu_read(|s| s[0..8].to_vec()).unwrap(),
        b"REV_CONT"
    );

    // Drop remaining objects and backends
    drop(shared);
    drop(gpu_bo);
    drop(mock_gpu);
    drop(mock_npu);

    let final_fds = get_open_fds();
    assert_eq!(
        initial_fds, final_fds,
        "Zero FD leaks when NPU BO dropped before GPU BO"
    );
}

#[test]
fn test_adversarial_try_clone_multi_generation_tree() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_fds = get_open_fds();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 64 * 1024;

    let gpu_bo = MemoryBridge::allocate_gem_bo(&mock_gpu, size, 4096).unwrap();
    let root_handle = gpu_bo.export_prime_fd().unwrap();

    // Create a tree of clones: 10 generations
    let mut clones = Vec::new();
    let mut cur = root_handle;
    for _ in 0..10 {
        let child = cur.try_clone().expect("try_clone must succeed");
        assert_ne!(cur.as_raw_fd(), child.as_raw_fd());
        clones.push(cur);
        cur = child;
    }
    clones.push(cur); // 11 handles total

    // Verify each clone points to the exact same underlying file description
    let root_stat = nix::sys::stat::fstat(clones[0].as_raw_fd()).unwrap();
    for h in &clones {
        let st = nix::sys::stat::fstat(h.as_raw_fd()).unwrap();
        assert_eq!(st.st_dev, root_stat.st_dev);
        assert_eq!(st.st_ino, root_stat.st_ino);
    }

    // Drop first 5 clones
    clones.drain(0..5);
    assert_eq!(clones.len(), 6);

    // Use clone 0 to write data
    let mut view_a = SharedBuffer::new(clones.remove(0), true).unwrap();
    view_a
        .with_cpu_write(|slice| {
            slice[0..8].copy_from_slice(b"TREE_VAL");
        })
        .unwrap();

    // Use clone 4 to read data
    let view_b = SharedBuffer::new(clones.pop().unwrap(), true).unwrap();
    let read_val = view_b.with_cpu_read(|slice| slice[0..8].to_vec()).unwrap();
    assert_eq!(&read_val, b"TREE_VAL");

    drop(view_a);
    drop(view_b);
    drop(clones);
    drop(gpu_bo);
    drop(mock_gpu);

    let final_fds = get_open_fds();
    assert_eq!(
        initial_fds, final_fds,
        "Zero FD leaks across 10-generation clone tree"
    );
}

// ============================================================================
// Objective 4: High-Contention Concurrency & Multi-Threaded Stress
// ============================================================================

#[test]
fn test_adversarial_concurrency_stress_16_threads_320_cycles() {
    let _guard = acquire_lock();
    warmup_runtime();
    let initial_fds = get_open_fds();

    let num_threads = 16;
    let cycles_per_thread = 20; // 320 total cycles
    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let handle = thread::spawn(move || {
            let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
            let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());
            let size = 64 * 1024;

            for cycle in 0..cycles_per_thread {
                let (gpu_bo, mut shared) = MemoryBridge::allocate_shared(&mock_gpu, size, true)
                    .unwrap_or_else(|e| {
                        panic!("Thread {} cycle {} alloc failed: {:?}", thread_id, cycle, e)
                    });

                let npu_dmabuf = shared.handle().try_clone().unwrap();
                let npu_bo = MemoryBridge::import_dma_buf(&mock_npu, &npu_dmabuf).unwrap();

                let watermark = ((thread_id as u64) << 32) | (cycle as u64);
                shared
                    .with_cpu_write(|slice| {
                        slice[0..8].copy_from_slice(&watermark.to_le_bytes());
                    })
                    .unwrap();

                let read_val = shared
                    .with_cpu_read(|slice| {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slice[0..8]);
                        u64::from_le_bytes(b)
                    })
                    .unwrap();
                assert_eq!(read_val, watermark);

                assert!(gpu_bo.handle() > 0);
                assert!(npu_bo.handle() > 0);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("Worker thread panicked during concurrency stress");
    }

    let final_fds = get_open_fds();
    assert_eq!(
        initial_fds, final_fds,
        "File descriptor set must remain strictly invariant after 320 concurrent lifecycles"
    );
}

#[test]
fn test_adversarial_concurrent_readers_on_shared_buffer() {
    let _guard = acquire_lock();
    warmup_runtime();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let size = 64 * 1024;

    let (_gpu_bo, mut shared) = MemoryBridge::allocate_shared(&mock_gpu, size, true).unwrap();

    const WATERMARK: &[u8] = b"CONCURRENT_READS";
    shared
        .with_cpu_write(|slice| {
            slice[0..16].copy_from_slice(WATERMARK);
        })
        .unwrap();

    let shared_arc = Arc::new(shared);
    let mut handles = Vec::new();

    for _ in 0..16 {
        let buf = Arc::clone(&shared_arc);
        let h = thread::spawn(move || {
            for _ in 0..100 {
                let data = buf
                    .with_cpu_read(|slice| slice[0..16].to_vec())
                    .expect("with_cpu_read must succeed concurrently");
                assert_eq!(&data, WATERMARK);
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().unwrap();
    }
}

// ============================================================================
// Objective 5: Boundary Conditions, Extremes, and Error Robustness
// ============================================================================

#[test]
fn test_adversarial_boundary_conditions() {
    let _guard = acquire_lock();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // 1. Allocation of 0 bytes rejected
    let res_zero = MemoryBridge::allocate_gem_bo(&mock_gpu, 0, 64);
    assert!(res_zero.is_err());

    // 2. Misaligned allocation sizes rejected
    for &bad_size in &[1usize, 63, 65, 127, 4095] {
        let res = MemoryBridge::allocate_gem_bo(&mock_gpu, bad_size, 64);
        match res {
            Err(MemoryError::AlignmentError { required: 64, .. }) => {}
            other => panic!("Expected AlignmentError for size {}, got {:?}", bad_size, other),
        }
    }

    // 3. Misaligned alignment argument rejected
    let res_align = MemoryBridge::allocate_gem_bo(&mock_gpu, 128, 63);
    match res_align {
        Err(MemoryError::AlignmentError { required: 64, .. }) => {}
        other => panic!("Expected AlignmentError for alignment 63, got {:?}", other),
    }

    // 4. Import of invalid FD returns error without panic or leaking FD
    let mock_npu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_npu());
    let bad_dmabuf = unsafe { DmaBufHandle::from_raw_fd_unchecked(-1, 4096) };
    let import_res = MemoryBridge::import_dma_buf(&mock_npu, &bad_dmabuf);
    assert!(import_res.is_err());

    // 5. try_clone on invalid FD returns IoError
    let clone_res = bad_dmabuf.try_clone();
    assert!(clone_res.is_err());
    std::mem::forget(bad_dmabuf); // Avoid close(-1)
}
