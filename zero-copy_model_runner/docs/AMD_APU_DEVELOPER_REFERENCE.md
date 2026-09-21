# AMD Ryzen AI APU Platform Developer & Architectural Reference Guide

A comprehensive architectural and systems reference for software engineers and systems architects developing zero-copy heterogeneous AI inference runtimes, hardware kernels, and `.xclbin` dataflow graphs for **AMD Ryzen AI APUs** (AMD Strix Point, Gorgon Point, Krackan Point, and Strix Halo).

---

## 1. Silicon Architecture & Physical Subsystems

AMD Ryzen AI APUs integrate three distinct compute engines onto a single monolithic die (or multi-die package in Strix Halo), sharing a high-bandwidth physical **Unified Memory Architecture (UMA)**.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           AMD Ryzen AI APU (LPDDR5X UMA)                      │
├────────────────────────┬─────────────────────────────┬────────────────────────┤
│     Zen 5 / Zen 5c     │        RDNA 3.5 iGPU        │      XDNA 2 NPU        │
│       Host CPU         │       (Compute Engine)      │    (Dataflow Engine)   │
│  • 12C/24T (4C + 8Cc)  │  • Up to 16/40 CUs (gfx1150)│  • 32-Tile AIE2P Array │
│  • AVX-512 SIMD        │  • High-throughput GEMM     │  • 50–55 TOPS BF16/INT8│
│  • Tokenize & Sample   │  • Batched Prefill Phase    │  • Streaming GEMV Loop │
└───────────┬────────────┴──────────────┬──────────────┴───────────┬────────────┘
            │                           │                          │
            ▼                           ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                    Linux Prime dma-buf Unified Physical DRAM                  │
│               (Zero-Copy Shared KV-Cache & Model Tensor Allocations)          │
└───────────────────────────────────────────────────────────────────────────────┘
```

### A. Engine Breakdown & Silicon Roles

1. **Host CPU (AMD Zen 5 & Zen 5c Hybrid Cores)**:
   - **Topology**: Heterogeneous mix of high-frequency *Zen 5 Classic* cores (dedicated 16MB L3) and dense *Zen 5c Compact* cores (shared L3, lower power).
   - **Role**: Fast subword tokenization, prompt parsing, sampling (greedy, Top-K/Top-P, temperature), and non-blocking asynchronous event loop orchestration.
   - **Vector ISA**: Native **AVX-512** (512-bit vector registers with `VNNI` and `BF16` instructions) for high-efficiency embedding transformations.

2. **Compute-Bound Engine (RDNA 3.5 iGPU - `gfx1150`)**:
   - **Configuration**: 16 Compute Units (Strix/Gorgon Point) or 40 Compute Units (Strix Halo).
   - **Role**: **Batched GEMM Prompt Prefill**. Highly parallel matrix multiplications with high arithmetic intensity ($\text{FLOPs/byte} \gg 10$).
   - **Runtime & Driver**: AMDGPU kernel driver (`/dev/dri/renderD128`) interfacing via ROCm/HIP.

3. **Memory-Bound Engine (XDNA 2 NPU - AIE2P Tile Array)**:
   - **Configuration**: 32 spatial AI Engine 2 Plus (AIE2P) tiles (Strix Point/Halo) or 16 tiles (Krackan Point).
   - **Compute Capacity**: 50 to 55 TOPS (BlockFP16 / INT8).
   - **Local Memory**: ~64 KB dedicated data SRAM + 16 KB instruction memory per tile (~2–3 MB total on-chip distributed SRAM) providing multi-terabyte/sec aggregate tile interconnect bandwidth.
   - **Role**: **Autoregressive Single-Token Decode (GEMV)**. Sequential token generation is memory-bandwidth bound ($\text{FLOPs/byte} \approx 1$). Streaming weights directly into spatial tiles minimizes energy consumption.
   - **Runtime & Driver**: Linux Compute Accelerator driver (`amdxdna.ko` / `/dev/accel/accel0`) interfacing via the Xilinx Runtime (XRT) API.

### B. Silicon Family Matrix

| Silicon Family | iGPU Prefill | NPU Decode | Peak NPU TOPS | UMA Memory Bus | Peak Bandwidth |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Strix Point** (HX 370 / 365) | 16 CUs RDNA 3.5 | 32-tile AIE2P | 50 TOPS | 128-bit LPDDR5X-7500 | 136 GB/s |
| **Gorgon Point** (HX 470) | 16 CUs RDNA 3.5 | 32-tile AIE2P | 55 TOPS | 128-bit LPDDR5X-8000 | 136 GB/s |
| **Krackan Point** (Ryzen AI 7) | 8 CUs RDNA 3.5 | 16-tile AIE2P | 32 TOPS | 128-bit LPDDR5X-7500 | 120 GB/s |
| **Strix Halo** (MAX+ 395) | 40 CUs RDNA 3.5 | 32-tile AIE2P | 55 TOPS | 256-bit LPDDR5X-8533 | 273 GB/s |

---

## 2. Designing & Synthesizing XDNA 2 Microcode (`.xclbin`)

An **XCLBIN** (Xilinx Container Linear Binary) is a packaged binary hardware container containing the compiled spatial dataflow graph, DMA transaction routing, and Embedded Runtime (ERT) microcontroller instruction stream for the AIE2P array.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        XCLBIN Container Layout                         │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Header & Metadata Block (UUID, Target Silicon: XDNA 2 AIE2P, DPU)  │
├────────────────────────────────────────────────────────────────────────┤
│ 2. AIE Control & Tile Configuration Stream                             │
│    • Core compute kernels (VLIW / Vector instructions)                 │
│    • Local tile SRAM ping-pong buffer layout (64-byte aligned)         │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Spatial DMA Switchbox Interconnect & Shim Routing                   │
│    • Non-blocking streaming channels between LPDDR5X DRAM and SRAM     │
├────────────────────────────────────────────────────────────────────────┤
│ 4. ERT (Embedded Runtime Scheduler) Microcontroller Instructions      │
└────────────────────────────────────────────────────────────────────────┘
```

### A. The XDNA 2 Compilation Toolchain

Creating custom `.xclbin` profiles requires compiling high-level neural operations into spatial dataflow graphs using the open-source MLIR-AIE toolchain:

```
[PyTorch / Triton / IRON Kernel]
               │
               ▼
   [Triton-Shared / Linalg Dialect]
               │
               ▼
   [MLIR-AIE / MLIR-AIR Compiler]  <--- Tiling & Spatial Tile Mapping
               │
               ▼
      [Peano LLVM Clang]           <--- Compiles AIE2P Vector C++ Intrinsics
               │
               ▼
     [xclbinutil / aiebu]          <--- Packages Bitstream & ERT Microcode
               │
               ▼
    [Turnkey Output: model.xclbin]
```

1. **High-Level IR (Triton / IRON)**: Kernels express batched matrix-vector multiplications, activation quantizations, and RMSNorm layers.
2. **MLIR-AIE / MLIR-AIR**: Performs spatial placement, tiling the iteration space across the 32 AIE2P tiles and routing memory streams through the AIE array interconnect.
3. **Peano LLVM**: Open-source LLVM-based backend targeting the AIE2P VLIW vector core instruction set.
4. **`xclbinutil` & `aiebu`**: Bundles ELF binaries, memory topology descriptors, and hardware metadata into the final `.xclbin`.

### B. Essential Rules for AIE2P Kernel Design

1. **Memory Alignment**: All DMA read/write descriptors must be aligned to **64-byte cacheline boundaries**. Unaligned strides cause hardware bus faults.
2. **Ping-Pong Buffer Tiling**: Utilize double-buffering in local tile SRAM (e.g. 32 KB active compute buffer, 32 KB background DMA prefetch buffer) to completely hide LPDDR5X DRAM latency.
3. **Quantization Encoding**: AIE2P tiles natively accelerate **BlockFP16** and **INT8/INT4** matrix operations. Sub-4-bit formats with non-power-of-two bitwidths (`IQ2`, `IQ3`) require software decompression before feeding spatial tiles, causing severe throughput regression.

---

## 3. Linux Kernel UAPI & Zero-Copy Memory Bridges

To achieve maximum inference throughput, memory handoff between the RDNA 3.5 iGPU and the XDNA 2 NPU must occur directly in physical LPDDR5X DRAM without copying through host RAM.

```
       AMDGPU DRM Driver                          AMDXDNA Driver
   (/dev/dri/renderD128)                       (/dev/accel/accel0)
 ┌───────────────────────────┐               ┌───────────────────────────┐
 │  Allocate GEM Buffer BO   │               │   Import dma-buf into     │
 │ (Physically Contiguous)   │               │   NPU IOMMU Page Table    │
 └─────────────┬─────────────┘               └─────────────▲─────────────┘
               │                                           │
               ▼                                           │
     DRM PRIME Export (fd) ────────────────────────────────┘
    (DRM_IOCTL_PRIME_HANDLE_TO_FD)
```

### A. The `/dev/accel/accel0` Character Device

The AMD XDNA NPU is exposed under the Linux Compute Accelerator subsystem (`/dev/accel/accel0`), managed by `drivers/accel/amdxdna/amdxdna.ko`.

* **Context Creation**: `AMDXDNA_IOCTL_CREATE_CONTEXT` establishes hardware ring submissions.
* **Buffer Management**: `AMDXDNA_IOCTL_CREATE_BO` / `AMDXDNA_IOCTL_MAP_BO` manages IOMMU page table bindings.
* **Execution Dispatch**: `AMDXDNA_IOCTL_EXEC_CMD` submits hardware command packets to the ERT scheduler.

### B. Zero-Copy `dma-buf` Memory Sharing (C / C++ Walkthrough)

```cpp
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include <xf86drm.h>
#include <amdgpu_drm.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

// Step 1: Allocate physical buffer on iGPU via AMDGPU DRM GEM
uint32_t gem_handle;
struct drm_amdgpu_gem_create gem_create = {};
gem_create.bo_size = buffer_size_bytes; // 64-byte aligned
gem_create.alignment = 64;
gem_create.domains = AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GTT;
drmIoctl(amdgpu_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &gem_create);
gem_handle = gem_create.out.handle;

// Step 2: Export GEM handle to Linux PRIME dma-buf File Descriptor
int dmabuf_fd = -1;
drmPrimeHandleToFD(amdgpu_fd, gem_handle, DRM_CLOEXEC | DRM_RDWR, &dmabuf_fd);

// Step 3: Import dma-buf FD directly into XDNA NPU (via XRT)
xrt::device npu_device(0); // /dev/accel/accel0
xrt::bo npu_shared_kv(npu_device, dmabuf_fd, buffer_size_bytes, 0);

// Both iGPU and NPU now point to identical physical DRAM addresses!
```

### C. Host Cache Coherency Bracketing (`DMA_BUF_IOCTL_SYNC`)

When host CPU cores need to read or write shared memory (e.g. inspecting logits or initializing embeddings), cache coherency must be explicitly bracketed:

```cpp
struct dma_buf_sync sync = {};

// Begin CPU Read session (flushes pending device write caches)
sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync);

// ... Read logits or update state on CPU ...

// End CPU session
sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync);
```

---

## 4. Hardware Synchronization via `drm_syncobj` Timeline Fences

Traditional AI runtimes poll the CPU (`while (!done) {}`), wasting power and degrading throughput. The AMD APU architecture synchronizes accelerators using **Linux DRM Synchronization Objects (`drm_syncobj`)** with 64-bit monotonically increasing timeline points.

```
Timeline Point (N-1)           Timeline Point (N)            Timeline Point (N+1)
      ───► [iGPU Prefill Done] ──────► [NPU Decode Step] ──────► [Host Sampler]
                 │                              │
                 └────── Hardware dma_fence ────┘
                      (Zero Host CPU Overhead)
```

### Timeline Operations Lifecycle

1. **Creation**: Create a timeline synchronization object via `DRM_IOCTL_SYNCOBJ_CREATE` with `DRM_SYNCOBJ_CREATE_SIGNALED`.
2. **GPU Signal**: Pass the syncobj to ROCm/HIP command dispatch; the RDNA 3.5 hardware Command Processor (CP) signals point $N$ upon kernel completion.
3. **NPU Wait**: Pass the same syncobj to `/dev/accel/accel0` command packet as an execution dependency on point $N$. The NPU hardware begins execution the microsecond the iGPU completes prefill.
4. **Host Non-Blocking Wait**: When the host sampler needs to wait, it invokes `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`:

```cpp
struct drm_syncobj_timeline_wait wait_args = {};
uint32_t handle = syncobj_handle;
uint64_t point = current_timeline_point;

wait_args.handles = (uintptr_t)&handle;
wait_args.points = (uintptr_t)&point;
wait_args.count_handles = 1;
wait_args.timeout_nsec = 5000000000ULL; // 5 second timeout
wait_args.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;

ioctl(drm_fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait_args);
```

---

## 5. End-to-End Heterogeneous Execution Architecture

A production-grade inference engine coordinates compute, memory, and synchronization across the APU pipeline:

```
[User Query]
     │
     ▼
[Stage 1: Tokenization] ──────► Zen 5 Classic Core (AVX-512)
     │                           Pinned via pthread_setaffinity_np()
     ▼
[Stage 2: Batched Prefill] ───► RDNA 3.5 iGPU (gfx1150 via ROCm/HIP)
     │                           Projects KV attention into shared dma-buf
     ▼
[drm_syncobj Fence] ──────────► Hardware-to-hardware sync signal
     │
     ▼
[Stage 3: Streaming Decode] ──► XDNA 2 NPU (32 AIE2P tiles via /dev/accel/accel0)
     │                           Streams model weights; updates KV in-place
     ▼
[Stage 4: Token Sampling] ────► Zen 5c Compact Core (Low-power background thread)
     │                           Greedy / Top-P / Temperature selection
     ▼
[Next Token / Repeat]
```

### Core Affinity Optimization

To prevent CPU cache trashing and minimize latency:
* Pin tokenization and prefill orchestration to **Zen 5 Classic cores** (e.g. Core 0–3) for maximum single-thread clock speed.
* Pin background event loops and polling monitors to **Zen 5c Compact cores** (e.g. Core 4–11), allowing primary cores to enter deep C-states during long token generation sessions.

---

## 6. Summary Checklist for APU Runtime Implementers

- [ ] **Physical Allocation**: Allocate all KV caches via AMDGPU DRM GEM and export to `dma-buf` FDs.
- [ ] **Cacheline Alignment**: Ensure all tensor structures are strictly 64-byte aligned.
- [ ] **Hardware Synchronization**: Replace all host CPU spinloops with `drm_syncobj` timeline points.
- [ ] **XCLBIN Resolution**: Bundle matching `.xclbin` profiles in `/usr/local/share/llama-apu/xclbins`.
- [ ] **Quantization Range**: Implement native support for Q4 through Q16 (`Q4_K_M`, `IQ4_NL`, `Q5_K_M`, `Q8_0`, `BF16`); reject unaligned sub-4-bit quants.
- [ ] **Permissions**: Verify udev rules (`/etc/udev/rules.d/99-amdxdna-apu.rules`) grant access to `/dev/accel/accel0` and `/dev/dri/renderD128`.
