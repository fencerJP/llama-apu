# Test Infrastructure Specification: Heterogeneous APU Orchestrator Runtime

**Project:** `zero-copy-model-runner`  
**Target Architecture:** AMD Ryzen AI APUs (Strix Point, Krackan Point, Gorgon Point, Strix Halo)  
**Test Suite Type:** Opaque-Box, Requirement-Driven, Progressive 4-Tier E2E Test Harness  
**Authoritative Sources:** `ORIGINAL_REQUEST.md`, `constitution.md`, `spec.md`, `c4_architecture.md`, `plan.md`

---

## 1. Executive Summary & Design Principles

The testing infrastructure for the Heterogeneous APU Model Runner runtime enforces strict non-negotiable architectural invariants:
1. **Absolute Zero-Copy Invariant:** Data sharing across RDNA 3.5 iGPU (prefill) and XDNA 2 NPU (decode) occurs strictly through Linux `dma-buf` file descriptors without host-side `memcpy` or intermediate CPU bounce buffers.
2. **Deterministic Progressive Testability:** Tests execute deterministically in both live APU hardware environments (`/dev/dri/renderD128` + `/dev/accel/accel0`) and containerized/CI environments via high-fidelity mock UAPI driver shims (`memfd_create` + atomic timeline fences) without requiring multi-gigabyte model weights or pre-compiled XCLBINs.
3. **Strict RAII & Resource Safety:** Every file descriptor, GEM buffer handle, and memory-mapped address space is bound to RAII lifecycle guards (`OwnedFd`, `munmap` in `Drop`), guaranteeing zero descriptor leaks or dangling pointers.
4. **Cache Coherency Integrity:** All CPU-mapped inspections of shared buffers strictly enforce 64-byte cache-line alignment and execute explicit `DMA_BUF_IOCTL_SYNC` brackets (`DMA_BUF_SYNC_START` and `DMA_BUF_SYNC_END`).

---

## 2. Directory Layout & Module Organization

All E2E test code, common fixtures, and mock drivers reside strictly within the designated testing paths:

```
tests/
├── common/
│   └── mod.rs                 # Shared test fixtures, silicon probe, mock engines, watermark verifier
└── e2e/
    ├── main.rs                # Root integration test binary runner
    ├── tier1_feature_coverage.rs  # Tier 1: Feature coverage (R1–R5, 25 tests)
    ├── tier2_boundary_corner.rs   # Tier 2: Boundary & corner cases (R1–R5, 25 tests)
    ├── tier3_cross_feature.rs     # Tier 3: Cross-feature combinations (5 tests)
    └── tier4_real_world.rs        # Tier 4: Real-world operational scenarios (5 tests)
```

---

## 3. The 4-Tier Testing Methodology

The test suite implements a rigorous 4-tier testing hierarchy comprising **60 opaque-box test cases**:

### Tier 1: Feature Coverage (25 Tests)
Validates primary behavior (happy paths) for core requirements R1–R5 ($\ge 5$ tests per feature):
- **Feature R1 (Zero-Copy Cross-Accelerator Memory Bridge):**
  - `test_r1_gem_allocation_and_export`: Allocates physical memory buffer, exports via PRIME to `dma-buf`.
  - `test_r1_amdx_import_and_handle`: Imports exported `dma-buf` into AMDXDNA / NPU address space.
  - `test_r1_64byte_cache_line_alignment`: Enforces strict 64-byte cache line alignment on CPU mmap views.
  - `test_r1_cpu_sync_brackets`: Wraps CPU read/write accesses with `DMA_BUF_IOCTL_SYNC` coherency brackets.
  - `test_r1_raii_teardown_and_leak_free`: Validates RAII teardown without dangling pointers or leaked FDs.
- **Feature R2 (Compute-Heavy Prompt Prefill Engine):**
  - `test_r2_prefill_engine_initialization`: Initializes iGPU compute context and queries peak TFLOPs.
  - `test_r2_prefill_dispatch_basic`: Dispatches batched GEMM over prompt tokens, projecting attention to `dma-buf`.
  - `test_r2_prefill_initial_token_emission`: Emits initial sampled token $T_0$ matching reference oracle.
  - `test_r2_prefill_timeline_fence_signaling`: Signals completion timeline fence point upon kernel finish.
  - `test_r2_prefill_batch_size_handling`: Validates varying prompt lengths (16, 64, 128 tokens) and batching.
- **Feature R3 (Memory-Bound Autoregressive Decode Engine):**
  - `test_r3_decode_engine_initialization`: Initializes XDNA AIE2P tile array and queries peak TOPS.
  - `test_r3_decode_attach_kv_cache`: Attaches shared zero-copy `dma-buf` KV cache to decode engine.
  - `test_r3_decode_single_step_dispatch`: Executes single-token autoregressive generation step.
  - `test_r3_decode_in_place_kv_append`: Appends newly generated KV states in-place directly into shared memory.
  - `test_r3_decode_eos_detection`: Detects EOS token ($T=128001$) and sets termination flag.
- **Feature R4 (Explicit Hardware Fence Synchronization):**
  - `test_r4_syncobj_create_and_destroy`: Creates and destroys kernel DRM synchronization objects.
  - `test_r4_syncobj_fd_export_import`: Converts syncobj handle to sync file descriptor and re-imports.
  - `test_r4_timeline_fence_wait_signaled`: Issues timeline point wait on signaled fence.
  - `test_r4_cross_device_fence_handoff`: Coordinates iGPU completion signal directly into NPU decode queue.
  - `test_r4_monotonic_timeline_progression`: Enforces monotonically increasing timeline points.
- **Feature R5 (Test Harness & Silicon Emulation Fallback):**
  - `test_r5_auto_fallback_detection`: Automatically audits `/dev/dri/renderD128` and `/dev/accel/accel0`.
  - `test_r5_mock_memfd_backed_buffer`: Allocates 64-byte aligned shared memory via `memfd_create`.
  - `test_r5_mock_uapi_ioctl_validation`: Verifies ioctl struct sizes against Linux kernel UAPI headers.
  - `test_r5_mock_timeline_fence_state_machine`: Simulates monotonic timeline points with timeout detection.
  - `test_r5_mock_dma_buf_sync_validation`: Validates `dma_buf_sync` flags and rejects invalid combinations.

### Tier 2: Boundary & Corner Cases (25 Tests)
Validates error recovery, edge limits, and defensive boundaries ($\ge 5$ tests per feature):
- **Feature R1 Boundaries:** Zero-length buffers, 64B alignment enforcement, invalid FD rejection, unmapped CPU access guards, extreme buffer sizes (1 byte to 64 MB).
- **Feature R2 Boundaries:** Empty prompt sequence rejection, single-token prompt execution, sequence length exceeding 8192 tokens, vocabulary boundary checks ($V > 128,256$), uninitialized prefill dispatch.
- **Feature R3 Boundaries:** Decode step without attached KV cache, sequence index overflow ($\ge 8192$), negative temperature rejection, uninitialized decode dispatch, zero-capacity KV cache attachment.
- **Feature R4 Boundaries:** Expired fence wait timeout, non-monotonic timeline point signaling rejection, invalid syncobj handle lookup, zero-timeout wait on unsignaled point, unsignaled point in wait-all array.
- **Feature R5 Boundaries:** Corrupted DMA-BUF sync flags, double-destroy of syncobj handles, zero-byte mock allocations, invalid timeline queries, excessive slab allocations.

### Tier 3: Cross-Feature Combinations (5 Tests)
Validates multi-feature interaction and pairwise integration:
1. `test_t3_alloc_prime_export_amdxdna_import_mutation`: Full pipeline: GEM alloc $\rightarrow$ PRIME export $\rightarrow$ AMDXDNA import $\rightarrow$ watermark write $\rightarrow$ in-place mutation $\rightarrow$ verify identity without memcpy.
2. `test_t3_prefill_direct_write_and_fence_handoff_to_decode`: Prefill direct KV write $\rightarrow$ timeline fence signal $\rightarrow$ decode timeline wait $\rightarrow$ first decode step.
3. `test_t3_multi_step_autoregressive_loop_monotonic_fence`: 5-step autoregressive loop with monotonic timeline points ($P_1 \dots P_6$) and in-place KV cache growth.
4. `test_t3_concurrent_sessions_shared_engine_tenant_isolation`: Concurrent sessions A and B sharing engines with independent `dma-buf` caches verifying memory isolation.
5. `test_t3_memory_recycling_and_tenant_zero_initialization`: Recycling KV slab memory between tenants with mandatory zero-initialization scrub.

### Tier 4: Real-World Application Scenarios (5 Tests)
Validates complete operational user workflows:
1. `test_t4_llama3_8b_e2e_prefill_and_10step_decode`: Llama-3-8B chat completion ("The capital of France is" $\rightarrow$ 10-step decode to " Paris is known for its art, culture. <|end_of_text|>").
2. `test_t4_concurrent_multitenant_pipelined_inference`: 4 concurrent client sessions pipelined across the orchestrator with independent token streams.
3. `test_t4_graceful_fallback_under_missing_hardware_nodes`: Headless / non-root / containerized execution gracefully engaging software mock drivers.
4. `test_t4_speculative_drafting_pipeline`: Speculative drafting ($K=4$ candidate tokens drafted on NPU $\rightarrow$ batched verification on iGPU over shared `dma-buf`).
5. `test_t4_dynamic_sliding_window_kv_pruning`: Long context exceeding window $\rightarrow$ in-place sliding window eviction without memory reallocation.

---

## 4. Test Execution & Automation Guide

### Run Full E2E Test Suite
```bash
cargo test --test e2e -- --nocapture
```

### Run All Tests Across Workspace
```bash
cargo test --test '*'
```

### Compile Tests Without Running (CI Verification)
```bash
cargo test --test '*' --no-run
```

### Run Specific Test Tier
```bash
# Tier 1: Feature Coverage
cargo test --test e2e -- tier1_feature_coverage

# Tier 2: Boundary & Corner Cases
cargo test --test e2e -- tier2_boundary_corner

# Tier 3: Cross-Feature Combinations
cargo test --test e2e -- tier3_cross_feature

# Tier 4: Real-World Scenarios
cargo test --test e2e -- tier4_real_world
```

### Run Hardware Silicon-Specific Tests
```bash
cargo test --test e2e -- test_r1_amdx_import_and_handle --nocapture
```

---

## 5. Environmental Matrix & Hardware Support

| Component | Physical Silicon Node | Driver / UAPI | Software Mock Fallback Shim |
|---|---|---|---|
| **GPU GEM Buffer** | `/dev/dri/renderD128` | `DRM_IOCTL_AMDGPU_GEM_CREATE` | `memfd_create` + 64B page alignment |
| **PRIME Export** | `/dev/dri/renderD128` | `DRM_IOCTL_PRIME_HANDLE_TO_FD` | `OwnedFd` wrapping `memfd` |
| **NPU BO Import** | `/dev/accel/accel0` | `DRM_IOCTL_PRIME_FD_TO_HANDLE` | In-memory handle lookup table |
| **Fence Sync** | `/dev/dri/renderD128` | `drm_syncobj` Timeline IOCTLs | `MockTimelineSyncobj` atomic point engine |
| **Cache Brackets**| DMA-BUF FD | `DMA_BUF_IOCTL_SYNC` | UAPI flag validator + memory fences |
| **CPU Pinning** | `/sys/devices/system/cpu` | `sched_setaffinity` | `CpuSet` fallback to Core 0 |

---

## 6. Authoritative Reference Ground Truth

Output verification against fixed prompt inputs utilizes deterministic reference oracles:
- **Reference Prompt:** `[128000, 791, 7421, 315, 9607, 374]` ("The capital of France is")
- **Reference Continuation:** `[9607, 374, 9552, 315, 420, 8496, 11, 7176, 13, 128001]` (" Paris is known for its art, culture. <|end_of_text|>")
- **KPI Gates:**
  - TTFT: $< 55$ ms acceptance threshold.
  - ITL: $\le 28$ ms/token target on Strix Point.
  - Host Memcpy: Exactly $0.00$ GB/s.
