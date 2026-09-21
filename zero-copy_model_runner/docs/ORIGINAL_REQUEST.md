# Original User Request

## 2026-09-09T04:33:55Z

Build a production-grade, user-space heterogeneous LLM inference runtime orchestrator for AMD Ryzen AI APUs, implementing zero-copy memory handoff and static pipeline execution across RDNA 3.5 iGPU (prefill) and XDNA 2 NPU (decode) on Linux.

Working directory: /home/fencer/.openclaw/workspace/projects/zero-copy_model_runner
Integrity mode: demo

Reference specifications:
- Invariants & Principles: `constitution.md`
- Technical Specification & Silicon Matrix: `spec.md`
- C4 Architecture Model: `c4_architecture.md`
- Milestone Implementation Plan: `plan.md`
- Technical WBS Tasks: `tasks.md`

## Requirements

### R1. Zero-Copy Cross-Accelerator Memory Bridge (Milestone 1)
Implement unified physical memory buffer management via Linux `dma-buf` and DRM GEM without host-side copies. The system must allocate shared memory backing GEM buffer objects in AMDGPU, export them to standard `dma-buf` file descriptors, and import them directly into AMDXDNA / XRT buffer objects with strict 64-byte cache-line alignment and explicit CPU cache synchronization brackets (`DMA_BUF_IOCTL_SYNC`).

### R2. Compute-Heavy Prompt Prefill Engine (Milestone 2)
Implement the prompt prefill engine on the RDNA 3.5 iGPU using ROCm/HIP. The engine must accept tokenized prompt sequences, execute batched GEMM forward passes, and project attention Key and Value representations directly into the pre-allocated shared `dma-buf` KV cache without copying through host RAM.

### R3. Memory-Bound Autoregressive Decode Engine (Milestone 2)
Implement the autoregressive decode engine on the XDNA 2 NPU using the native XRT API. The engine must bind to the shared `dma-buf` KV cache populated during prefill, step through sequential token generation across AIE2P tiles, and append newly generated KV states in-place.

### R4. Explicit Hardware Fence Synchronization
Synchronize cross-device handoffs (iGPU prefill completion to NPU decode start) exclusively using Linux DRM synchronization objects (`drm_syncobj`) and timeline fences. The orchestrator must not busy-spin on the host CPU.

### R5. Test Harness & Silicon Emulation Fallback
Provide comprehensive test suites and verification harnesses for the pipeline. If physical AMD APU device nodes (`/dev/dri/renderD128`, `/dev/accel/accel0`) are unavailable or restricted in the execution environment, tests must automatically utilize valid mock UAPI driver shims that strictly validate ioctl structures, fence transitions, and buffer integrity without crashing.

## Acceptance Criteria

### Memory & Zero-Copy Invariants
- [ ] Automated verification script confirms zero host-side data copying (`memcpy`) during the handoff of attention matrices between prefill and decode engines.
- [ ] Exported `dma-buf` file descriptors are cleanly managed with RAII; no leaked file descriptors or unmapped pointers after test teardown.
- [ ] `DMA_BUF_IOCTL_SYNC` brackets are correctly invoked around any CPU-mapped buffer inspections.

### Pipeline Execution & Accuracy
- [ ] End-to-end forward pass test successfully runs a prefill phase followed by multi-step autoregressive decode iterations.
- [ ] Output logits or generated tokens match expected deterministic reference outputs for fixed prompt inputs.
- [ ] Cross-accelerator handoff triggers via `drm_syncobj` timeline fence completion without host CPU polling.

### Build & Test Suite Quality
- [ ] `cargo test` and all integration tests compile cleanly with zero errors and execute successfully.
- [ ] Fallback test stubs validate ioctl arguments against kernel UAPI definitions when run in virtualized or non-root environments.
