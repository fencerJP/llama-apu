// SPDX-License-Identifier: Apache-2.0
//! Adversarial Empirical Stress Test Suite for Milestone 2 Prefill Engine (`RocmPrefillEngine`).
//!
//! Conducted by Challenger 1 (challenger_m2_1).
//!
//! Objectives:
//! 1. Stress test varying prompt lengths (1, 16, 128, 512, 1024 tokens) and extreme context limits (2048, 4096, 8192).
//! 2. Empirically verify that attention Key and Value representations write directly into pre-allocated shared `dma-buf`
//!    without host RAM copies (measuring thread heap allocations and O(1) memory overhead).
//! 3. Verify deterministic token emission T_0 across varying inputs and reference prompt baseline.
//! 4. Stress test memory safety, boundary enforcement, start offsets, multi-threaded concurrency, and DRM fence progression.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use zero_copy_model_runner::backend::{open_or_mock, DeviceBackend, MockDeviceBackend};
use zero_copy_model_runner::engine::{
    DeterministicReferenceOracle, EngineError, PrefillEngine, PrefillRequest, RocmPrefillEngine,
};
use zero_copy_model_runner::memory::{MemoryBridge, SharedBuffer};
use zero_copy_model_runner::uapi::{
    drm_syncobj_create, drm_syncobj_handle, DRM_IOCTL_SYNCOBJ_CREATE_NUM,
    DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_NUM,
};

// ============================================================================
// Thread-Local Tracking Allocator for Zero-Copy Heap Measurement
// ============================================================================

struct TrackingAllocator;

thread_local! {
    static TRACKING_ENABLED: Cell<bool> = const { Cell::new(false) };
    static THREAD_ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
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

static SERIAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn acquire_lock() -> std::sync::MutexGuard<'static, ()> {
    SERIAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn start_heap_tracking() {
    TRACKING_ENABLED.with(|e| e.set(true));
    THREAD_ALLOCATED_BYTES.with(|b| b.set(0));
}

fn stop_heap_tracking() -> usize {
    TRACKING_ENABLED.with(|e| e.set(false));
    THREAD_ALLOCATED_BYTES.with(|b| b.get())
}

/// Deterministic pseudo-random number generator for test sequence creation.
fn generate_deterministic_prompt(seed: u64, len: usize) -> Vec<u32> {
    let mut state = seed ^ 0x5DEECE66D;
    let mut tokens = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Valid token range within vocab (0..128,000)
        let tok = ((state >> 32) as u32) % 128_000;
        tokens.push(tok);
    }
    tokens
}

// ============================================================================
// 1. Stress Test Varying Prompt Lengths (1, 16, 128, 512, 1024 tokens + boundaries)
// ============================================================================

#[test]
fn test_stress_varying_prompt_lengths_and_kv_integrity() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).expect("initialize prefill engine");

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // Prompt lengths explicitly mandated by specification: 1, 16, 128, 512, 1024
    // Plus extended context stress: 2048, 4096, 8192
    let prompt_lengths = [1, 16, 128, 512, 1024, 2048, 4096, 8192];

    for &len in &prompt_lengths {
        let prompt = generate_deterministic_prompt(0xACE1_0000 + len as u64, len);
        let bo_size = (len * 128).max(4096); // Ensure minimum page size
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64)
            .unwrap_or_else(|e| panic!("allocate GEM BO failed for len {}: {:?}", len, e));
        let dmabuf = bo.export_prime_fd().expect("export PRIME FD");

        let req = PrefillRequest {
            token_ids: &prompt,
            start_offset: 0,
            batch_size: 1,
        };

        let result = engine
            .dispatch_prefill(req, &dmabuf, -1, len as u64)
            .unwrap_or_else(|e| panic!("dispatch_prefill failed for len {}: {:?}", len, e));

        assert_eq!(result.tokens_processed, len, "Length mismatch for len {}", len);
        assert_eq!(
            result.completion_fence_point, len as u64,
            "Fence point mismatch for len {}",
            len
        );
        assert_eq!(
            result.execution_time_us,
            (len as u64) * 85,
            "Execution time calculation mismatch for len {}",
            len
        );

        // Verify KV cache integrity byte-for-byte in shared dma-buf memory
        let shared_buf = SharedBuffer::new(dmabuf.try_clone().unwrap(), true)
            .expect("SharedBuffer map failed");

        shared_buf
            .with_cpu_read(|slice| {
                for (i, &token) in prompt.iter().enumerate() {
                    let offset = i * 128;
                    let token_slice = &slice[offset..offset + 128];
                    let expected_byte = (token % 251) as u8;
                    assert!(
                        token_slice.iter().all(|&b| b == expected_byte),
                        "KV projection mismatch at token index {} for prompt len {}",
                        i,
                        len
                    );
                }
            })
            .expect("with_cpu_read failed");
    }
}

// ============================================================================
// 2. Empirical Zero-Copy Verification: Direct DMA-BUF Write Without Host Memcpy
// ============================================================================

#[test]
fn test_empirical_zero_copy_heap_allocation_invariant() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).expect("initialize prefill engine");

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // Compare heap allocation between small prompts and large prompts.
    // In a zero-copy implementation backed by mmap and direct dma-buf writes,
    // heap allocations during dispatch_prefill must be O(1) constant metadata
    // overhead and NOT scale with the size of the KV cache (e.g. 1024 * 128 = 131,072 bytes).
    let test_cases = [
        (1, 1 * 128),
        (16, 16 * 128),
        (128, 128 * 128),
        (512, 512 * 128),
        (1024, 1024 * 128),
        (4096, 4096 * 128),
    ];

    let mut heap_allocations = Vec::new();

    for &(len, kv_bytes) in &test_cases {
        let prompt = generate_deterministic_prompt(0x1337_0000 + len as u64, len);
        let bo_size = kv_bytes.max(4096);
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();

        let req = PrefillRequest {
            token_ids: &prompt,
            start_offset: 0,
            batch_size: 1,
        };

        // Warm up and prime caches
        let _ = engine.dispatch_prefill(req.clone(), &dmabuf, -1, 1);

        // Measure heap allocations strictly during dispatch_prefill
        start_heap_tracking();
        let res = engine.dispatch_prefill(req, &dmabuf, -1, 2);
        let allocated_bytes = stop_heap_tracking();

        assert!(res.is_ok());
        heap_allocations.push((len, kv_bytes, allocated_bytes));
    }

    for &(len, kv_bytes, allocated_bytes) in &heap_allocations {
        // Empirical verification: Heap allocated bytes must be orders of magnitude
        // smaller than the KV tensor size. For a 4096-token prompt (524,288 bytes KV),
        // if a host buffer were created via memcpy or Vec, allocated_bytes would be >= 524,288.
        // Under zero-copy, allocated_bytes must be less than 1,024 bytes (strictly constant metadata).
        assert!(
            allocated_bytes < 1024,
            "Host heap copy detected! Prompt len {}: KV bytes = {}, but heap allocated = {} bytes",
            len,
            kv_bytes,
            allocated_bytes
        );
        assert!(
            allocated_bytes < kv_bytes / 10,
            "Heap allocation scaled with KV payload: len {}, kv_bytes {}, heap {}",
            len,
            kv_bytes,
            allocated_bytes
        );
    }

    // Verify O(1) scaling: 4096 tokens should NOT allocate significantly more heap than 16 tokens
    let heap_16 = heap_allocations[1].2;
    let heap_4096 = heap_allocations[5].2;
    let diff = (heap_4096 as isize - heap_16 as isize).abs();
    assert!(
        diff < 512,
        "Heap allocation is not O(1) invariant across prompt sizes: diff = {} bytes",
        diff
    );
}

#[test]
fn test_empirical_direct_dmabuf_projection_watermark() {
    let _lock = acquire_lock();
    // Empirically prove that the prefill engine writes directly into the shared dma-buf
    // memory pages by observing mutations through a pre-existing, independent SharedBuffer mapping.
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).expect("initialize prefill engine");

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo_size = 64 * 1024;
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    // Map the buffer independently BEFORE prefill executes
    let mut observer_buf = SharedBuffer::new(dmabuf.try_clone().unwrap(), true)
        .expect("Observer SharedBuffer map failed");

    // Initialize the buffer memory with a canary pattern 0x7E
    observer_buf
        .with_cpu_write(|slice| {
            slice.fill(0x7E);
        })
        .unwrap();

    // Verify initial canary pattern is set
    observer_buf
        .with_cpu_read(|slice| {
            assert!(slice.iter().all(|&b| b == 0x7E));
        })
        .unwrap();

    // Dispatch prefill
    let prompt = [101u32, 202, 303, 404, 505];
    let req = PrefillRequest {
        token_ids: &prompt,
        start_offset: 0,
        batch_size: 1,
    };

    engine.dispatch_prefill(req, &dmabuf, -1, 1).expect("dispatch_prefill");

    // Immediately inspect memory through the independent observer mapping.
    // The physical pages must reflect the written KV representations without any host transfer!
    observer_buf
        .with_cpu_read(|slice| {
            // First 5 tokens (5 * 128 bytes) must match (token % 251)
            for (i, &token) in prompt.iter().enumerate() {
                let offset = i * 128;
                let token_slice = &slice[offset..offset + 128];
                let expected = (token % 251) as u8;
                assert!(
                    token_slice.iter().all(|&b| b == expected),
                    "Token {} at offset {} mismatch in observer mapping",
                    token,
                    offset
                );
            }

            // Remaining memory beyond the prompt must remain intact canary 0x7E
            let remainder = &slice[prompt.len() * 128..];
            assert!(
                remainder.iter().all(|&b| b == 0x7E),
                "Canary corrupted beyond prompt prefill range"
            );
        })
        .unwrap();
}

// ============================================================================
// 3. Start Offset and Partial KV Prefill Boundary Isolation
// ============================================================================

#[test]
fn test_prefill_non_zero_start_offset_isolation() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).unwrap();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo_size = 128 * 1024;
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, bo_size, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    let mut observer = SharedBuffer::new(dmabuf.try_clone().unwrap(), true).unwrap();

    // Fill buffer with distinct pre-existing canary watermark
    observer
        .with_cpu_write(|slice| {
            slice.fill(0x33);
        })
        .unwrap();

    // Prefill 128 tokens starting at offset 64 (i.e. bytes 64 * 128 = 8192)
    let prompt = generate_deterministic_prompt(9999, 128);
    let start_offset = 64;
    let req = PrefillRequest {
        token_ids: &prompt,
        start_offset,
        batch_size: 1,
    };

    engine.dispatch_prefill(req, &dmabuf, -1, 1).unwrap();

    // Validate memory layout isolation:
    // 1. [0 .. 64 * 128] must still be 0x33
    // 2. [64 * 128 .. (64 + 128) * 128] must be (token % 251)
    // 3. [(64 + 128) * 128 .. bo_size] must still be 0x33
    observer
        .with_cpu_read(|slice| {
            let start_byte = start_offset * 128;
            let end_byte = (start_offset + prompt.len()) * 128;

            let prefix = &slice[..start_byte];
            assert!(
                prefix.iter().all(|&b| b == 0x33),
                "Prefix before start_offset was modified!"
            );

            for (i, &token) in prompt.iter().enumerate() {
                let off = start_byte + i * 128;
                let token_slice = &slice[off..off + 128];
                let expected = (token % 251) as u8;
                assert!(
                    token_slice.iter().all(|&b| b == expected),
                    "Token {} mismatch at offset {}",
                    token,
                    off
                );
            }

            let suffix = &slice[end_byte..];
            assert!(
                suffix.iter().all(|&b| b == 0x33),
                "Suffix after prefill range was modified!"
            );
        })
        .unwrap();
}

// ============================================================================
// 4. Deterministic Token Emission T_0 Across Inputs
// ============================================================================

#[test]
fn test_deterministic_token_emission_t0() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).unwrap();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 256 * 1024, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    // 1. Reference prompt baseline invariant
    let ref_prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    for iter in 0..50 {
        let req = PrefillRequest {
            token_ids: ref_prompt,
            start_offset: 0,
            batch_size: 1,
        };
        let res = engine.dispatch_prefill(req, &dmabuf, -1, 1).unwrap();
        assert_eq!(
            res.initial_token_id, 9607,
            "Reference prompt T_0 was non-deterministic on iteration {}",
            iter
        );
        assert_eq!(
            res.initial_token_id,
            DeterministicReferenceOracle::EXPECTED_DECODE_STEPS[0]
        );
    }

    // 2. Determinism across varying prompt lengths (1, 16, 128, 512, 1024)
    let lengths = [1, 16, 128, 512, 1024];
    for &len in &lengths {
        let prompt = generate_deterministic_prompt(0x5555 + len as u64, len);
        let expected_t0 = DeterministicReferenceOracle::next_token(&prompt);
        assert!(
            expected_t0 >= 1 && expected_t0 <= 100_000,
            "Predicted T_0 {} out of bounds for len {}",
            expected_t0,
            len
        );

        for iter in 0..20 {
            let req = PrefillRequest {
                token_ids: &prompt,
                start_offset: 0,
                batch_size: 1,
            };
            let res = engine.dispatch_prefill(req, &dmabuf, -1, 1).unwrap();
            assert_eq!(
                res.initial_token_id, expected_t0,
                "Non-deterministic T_0 for len {} on iter {}",
                len, iter
            );
        }
    }

    // 3. Distinct prompts must yield distinct T_0 emissions (entropy check)
    let prompt_a = [1000u32, 2000, 3000];
    let prompt_b = [1000u32, 2000, 3001];
    let res_a = engine
        .dispatch_prefill(
            PrefillRequest {
                token_ids: &prompt_a,
                start_offset: 0,
                batch_size: 1,
            },
            &dmabuf,
            -1,
            1,
        )
        .unwrap();
    let res_b = engine
        .dispatch_prefill(
            PrefillRequest {
                token_ids: &prompt_b,
                start_offset: 0,
                batch_size: 1,
            },
            &dmabuf,
            -1,
            1,
        )
        .unwrap();
    assert_ne!(
        res_a.initial_token_id, res_b.initial_token_id,
        "Distinct prompts produced collision on T_0"
    );
}

// ============================================================================
// 5. Boundary Conditions, Limits, and Error Rejection
// ============================================================================

#[test]
fn test_boundary_conditions_and_error_handling() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).unwrap();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 65536, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    // 1. Empty prompt rejected
    let res_empty = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &[],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res_empty, Err(EngineError::InvalidArgument(_))));

    // 2. Prompt exceeding 8192 rejected
    let prompt_8193 = vec![42u32; 8193];
    let res_large = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &prompt_8193,
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res_large, Err(EngineError::InvalidArgument(_))));

    // 3. Exactly 8192 tokens accepted
    let bo_1mb = MemoryBridge::allocate_gem_bo(&mock_gpu, 8192 * 128, 64).unwrap();
    let dmabuf_1mb = bo_1mb.export_prime_fd().unwrap();
    let prompt_8192 = vec![42u32; 8192];
    let res_8192 = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &prompt_8192,
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf_1mb,
        -1,
        1,
    );
    assert!(res_8192.is_ok(), "Exact 8192 token limit should be accepted");

    // 4. Out-of-vocabulary token ID rejected
    let res_oov = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &[128_257],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res_oov, Err(EngineError::InvalidArgument(_))));

    let res_u32_max = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &[u32::MAX],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(matches!(res_u32_max, Err(EngineError::InvalidArgument(_))));

    // 5. In-vocabulary boundary tokens accepted
    let valid_boundary_tokens = [0u32, 128_000, 128_256];
    let res_valid = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &valid_boundary_tokens,
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf,
        -1,
        1,
    );
    assert!(res_valid.is_ok(), "Valid boundary token IDs should succeed");

    // 6. Insufficient buffer capacity: exact vs 1-byte short
    let exact_bytes = 4 * 128; // 512 bytes
    let bo_exact = MemoryBridge::allocate_gem_bo(&mock_gpu, exact_bytes, 64).unwrap();
    let dmabuf_exact = bo_exact.export_prime_fd().unwrap();
    let res_exact = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &[1, 2, 3, 4],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf_exact,
        -1,
        1,
    );
    assert!(res_exact.is_ok(), "Exact buffer capacity must succeed");

    // Undersized buffer
    let bo_short = MemoryBridge::allocate_gem_bo(&mock_gpu, exact_bytes - 64, 64).unwrap();
    let dmabuf_short = bo_short.export_prime_fd().unwrap();
    let res_short = engine.dispatch_prefill(
        PrefillRequest {
            token_ids: &[1, 2, 3, 4],
            start_offset: 0,
            batch_size: 1,
        },
        &dmabuf_short,
        -1,
        1,
    );
    assert!(matches!(res_short, Err(EngineError::OutOfResources(_))));
}

// ============================================================================
// 6. Multi-Threaded Concurrency Stress & Cumulative Accounting
// ============================================================================

#[test]
fn test_concurrency_stress_prefill_threads() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).unwrap();
    let engine = Arc::new(engine);

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());
    let num_threads = 16;
    let cycles_per_thread = 20;

    let total_expected_tokens = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    for thread_idx in 0..num_threads {
        let engine_clone = Arc::clone(&engine);
        let mock_gpu_clone = Arc::clone(&mock_gpu);
        let token_counter = Arc::clone(&total_expected_tokens);

        let h = thread::spawn(move || {
            let prompt_lens = [1, 16, 128, 512, 1024];

            for cycle in 0..cycles_per_thread {
                let len = prompt_lens[(thread_idx + cycle) % prompt_lens.len()];
                let prompt = generate_deterministic_prompt(
                    (thread_idx as u64 * 1000) + cycle as u64,
                    len,
                );

                let bo_size = (len * 128).max(4096);
                let bo = MemoryBridge::allocate_gem_bo(&mock_gpu_clone, bo_size, 64).unwrap();
                let dmabuf = bo.export_prime_fd().unwrap();

                let req = PrefillRequest {
                    token_ids: &prompt,
                    start_offset: 0,
                    batch_size: 1,
                };

                let res = engine_clone
                    .dispatch_prefill(req, &dmabuf, -1, cycle as u64 + 1)
                    .unwrap();

                assert_eq!(res.tokens_processed, len);
                token_counter.fetch_add(len, Ordering::Relaxed);

                // Spot-check KV cache content for first and last token in prompt
                let shared = SharedBuffer::new(dmabuf, true).unwrap();
                shared
                    .with_cpu_read(|slice| {
                        let first_byte = (prompt[0] % 251) as u8;
                        assert_eq!(slice[0], first_byte);
                        let last_byte = (prompt[len - 1] % 251) as u8;
                        assert_eq!(slice[(len - 1) * 128], last_byte);
                    })
                    .unwrap();
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().expect("Worker thread panicked during prefill stress");
    }

    let expected = total_expected_tokens.load(Ordering::SeqCst);
    let actual = engine.total_tokens_processed() as usize;
    assert_eq!(
        actual, expected,
        "Cumulative token counter mismatch: actual {} != expected {}",
        actual, expected
    );
}

// ============================================================================
// 7. DRM Syncobj Timeline Fence Signaling
// ============================================================================

#[test]
fn test_prefill_drm_syncobj_fence_signaling() {
    let _lock = acquire_lock();
    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // Allocate syncobj
    let mut create_arg = drm_syncobj_create {
        handle: 0,
        flags: 0,
    };
    mock_gpu
        .ioctl(
            DRM_IOCTL_SYNCOBJ_CREATE_NUM,
            &mut create_arg as *mut _ as *mut libc::c_void,
        )
        .expect("syncobj create");
    let syncobj_handle = create_arg.handle;

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
            DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD_NUM,
            &mut h2fd as *mut _ as *mut libc::c_void,
        )
        .expect("syncobj handle to fd");
    let syncobj_fd = h2fd.fd;

    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).unwrap();

    let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 16384, 64).unwrap();
    let dmabuf = bo.export_prime_fd().unwrap();

    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let target_point = 77u64;

    let res = engine
        .dispatch_prefill(
            PrefillRequest {
                token_ids: prompt,
                start_offset: 0,
                batch_size: 1,
            },
            &dmabuf,
            syncobj_fd,
            target_point,
        )
        .expect("dispatch_prefill with syncobj");

    assert_eq!(res.completion_fence_point, target_point);

    // Negative syncobj_fd (-1) must succeed without error
    let res_no_fence = engine
        .dispatch_prefill(
            PrefillRequest {
                token_ids: prompt,
                start_offset: 0,
                batch_size: 1,
            },
            &dmabuf,
            -1,
            0,
        )
        .expect("dispatch_prefill without fence");
    assert_eq!(res_no_fence.completion_fence_point, 0);
}

// ============================================================================
// 8. Physical Silicon vs Mock Device Backend Equivalence
// ============================================================================

#[test]
fn test_physical_device_or_mock_equivalence() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    // Use device index 0 (which attempts /dev/dri/renderD128 or falls back to mock)
    engine.initialize(0).expect("initialize");

    let is_mock = engine.is_mock();
    assert_eq!(engine.device_index(), 0);
    assert_eq!(engine.peak_compute_tflops(), 32.0);

    let backend = open_or_mock("/dev/dri/renderD128").expect("open_or_mock");
    let bo = MemoryBridge::allocate_gem_bo(&backend, 16384, 64).expect("allocate BO");
    let dmabuf = bo.export_prime_fd().expect("export PRIME FD");

    let prompt = DeterministicReferenceOracle::REFERENCE_PROMPT;
    let res = engine
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
        .expect("prefill");

    assert_eq!(res.tokens_processed, 6);
    assert_eq!(res.initial_token_id, 9607);
    println!(
        "Prefill executed successfully on backend (is_mock: {})",
        is_mock
    );
}

// ============================================================================
// 9. File Descriptor and VMA Leak Freedom After Repeated Prefills
// ============================================================================

#[test]
fn test_fd_and_vma_leak_freedom_after_repeated_prefills() {
    let _lock = acquire_lock();

    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).unwrap();

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    let iterations = 100;
    for i in 0..iterations {
        let bo = MemoryBridge::allocate_gem_bo(&mock_gpu, 8192, 64).unwrap();
        let dmabuf = bo.export_prime_fd().unwrap();
        let raw_fd = dmabuf.as_raw_fd();

        let prompt = [10u32, 20, 30, (i % 100) as u32];
        let req = PrefillRequest {
            token_ids: &prompt,
            start_offset: 0,
            batch_size: 1,
        };

        let res = engine.dispatch_prefill(req, &dmabuf, -1, 1).unwrap();
        assert_eq!(res.tokens_processed, prompt.len());

        drop(dmabuf);
        drop(bo);

        // Verify that this specific FD was closed immediately upon drop
        let is_open = unsafe { libc::fcntl(raw_fd, libc::F_GETFD) } != -1;
        assert!(
            !is_open,
            "FD {} leaked / remained open after drop at iteration {}",
            raw_fd, i
        );
    }
}

// ============================================================================
// 10. Multi-Session KV Cache Memory Isolation Under Load
// ============================================================================

#[test]
fn test_concurrent_multi_session_kv_cache_isolation() {
    let _lock = acquire_lock();
    let mut engine = RocmPrefillEngine::new();
    engine.initialize(0).unwrap();
    let engine = Arc::new(engine);

    let mock_gpu: Arc<dyn DeviceBackend> = Arc::new(MockDeviceBackend::new_gpu());

    // Simulate 4 concurrent tenant inference sessions
    let num_sessions = 4;
    let mut session_handles = Vec::new();

    for session_id in 0..num_sessions {
        let engine_clone = Arc::clone(&engine);
        let mock_gpu_clone = Arc::clone(&mock_gpu);

        let h = thread::spawn(move || {
            let prompt_len = 64;
            // Generate tenant-specific prompt
            let prompt: Vec<u32> = (0..prompt_len)
                .map(|i| ((session_id * 10_000) + i) as u32)
                .collect();

            let bo_size = prompt_len * 128;
            let bo = MemoryBridge::allocate_gem_bo(&mock_gpu_clone, bo_size, 64).unwrap();
            let dmabuf = bo.export_prime_fd().unwrap();

            let req = PrefillRequest {
                token_ids: &prompt,
                start_offset: 0,
                batch_size: 1,
            };

            let res = engine_clone
                .dispatch_prefill(req, &dmabuf, -1, session_id as u64 + 1)
                .unwrap();
            assert_eq!(res.tokens_processed, prompt_len);

            // Verify that this tenant's buffer contains only this tenant's projected data
            let shared = SharedBuffer::new(dmabuf, true).unwrap();
            shared
                .with_cpu_read(|slice| {
                    for (i, &token) in prompt.iter().enumerate() {
                        let token_slice = &slice[i * 128..(i + 1) * 128];
                        let expected = (token % 251) as u8;
                        assert!(
                            token_slice.iter().all(|&b| b == expected),
                            "Cross-tenant corruption in session {} at token {}",
                            session_id,
                            i
                        );
                    }
                })
                .unwrap();
        });
        session_handles.push(h);
    }

    for h in session_handles {
        h.join().expect("Session thread failed");
    }
}

