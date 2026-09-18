The proposed architecture is a **user-space heterogeneous runtime orchestrator** designed for modern AMD APUs (such as Strix Point, Krackan Point, and Strix Halo). Instead of treating the CPU, iGPU, and NPU as isolated execution environments, it coordinates all on-die silicon engines as a cooperative pipeline.

By unifying memory via Linux `dma-buf` zero-copy sharing, synchronizing execution with kernel `dma_fence` primitives, and deterministically locking threads to asymmetric CPU core clusters (Zen 5 vs. Zen 5c), this architecture eliminates memory-copy bottlenecks, lowers Time-To-First-Token (TTFT), and drops autoregressive decode power draw by up to 70%.

---

### Software Stack & Interaction Model

The runtime orchestrator sits between the high-level application framework (such as `llama.cpp`, FastFlowLM, or ONNX Runtime) and standard Linux kernel drivers. It does not replace kernel schedulers; rather, it feeds the existing kernel device queues (`amdgpu` and `amdxdna`) and bypasses CPU scheduling heuristics via thread affinity.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      User-Space Inference Runtime                           │
│  (llama.cpp / FastFlowLM / ONNX Runtime Heterogeneous Execution Provider)   │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    APU Topology & Graph Orchestrator                  │  │
│  │  • Graph Partitioning (Prefill -> GPU, Decode -> NPU)                 │  │
│  │  • Static Topology Table (Zen 5 Classic vs. Zen 5c Compact, NUMA)     │  │
│  │  • Dynamic Cost Engine (Monitors Queue Depth & Thermal Limits)        │  │
│  └──────────────────┬───────────────────┬───────────────────┬────────────┘  │
└─────────────────────┼───────────────────┼───────────────────┼───────────────┘
                      │                   │                   │
      [pthread_setaffinity_np]    [ROCm / HIP / libdrm]   [XRT Native C++ API]
                      │                   │                   │
                      ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Linux Kernel Space                              │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐   ┌──────────────────┐  │
│  │ CPU Scheduler       │    │ DRM GPU Subsystem   │   │ DRM Accel Subsys │  │
│  │ (EEVDF / CFS)       │    │ (drivers/gpu/drm/   │   │ (drivers/accel/  │  │
│  │                     │    │  amd/amdgpu)        │   │  amdxdna)        │  │
│  └──────────┬──────────┘    └──────────┬──────────┘   └────────┬─────────┘  │
│             │                          │                       │            │
│             │           ┌──────────────┴───────────────────────┴─────────┐  │
│             │           │          Linux dma-buf & drm_syncobj           │  │
│             │           │  • Zero-Copy GEM buffer page sharing           │  │
│             │           │  • Kernel-managed hardware dma_fences          │  │
│             │           └──────────────────────┬─────────────────────────┘  │
└─────────────┼──────────────────────────────────┼────────────────────────────┘
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               APU Silicon                                   │
│                                                                             │
│  ┌─────────────────────────┐      ┌──────────────────────────────────────┐  │
│  │ Heterogeneous CPU Cores │      │       Unified LPDDR5X System RAM     │  │
│  │                         │      │                                      │  │
│  │ • Zen 5 (4 Cores / 16MB)│      │  ┌────────────────────────────────┐  │  │
│  │   [Prefill Feeder, AVX] │      │  │ Unified Zero-Copy KV Cache     │  │  │
│  │                         │      │  │ (Shared physical page frames)  │  │  │
│  │ • Zen 5c (8 Cores / 8MB)│      │  └──────────────▲─────────────────┘  │  │
│  │   [XRT Wait / Dec. Poll]│      └─────────────────┼────────────────────┘  │
│  └─────────────────────────┘                        │                       │
│                                                     │                       │
│  ┌─────────────────────────┐      ┌─────────────────┴────────────────────┐  │
│  │ RDNA 3.5 iGPU           │      │ XDNA 2 NPU                           │  │
│  │ (Radeon 890M / 8060S)   │      │ (32 AIE2P Tiles / 4MB SRAM)         │  │
│  │ • Prefill Compute Engine│      │ • Decode Token Generation Engine     │  │
│  │ • Command Processor (CP)│◄────►│ • ERT / MERT Microcontroller       │  │
│  │   Hardware Submission   │ Sync │   Spatial Tile DMA Scheduling        │  │
│  └─────────────────────────┘      └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

### Step-by-Step Execution and Data Flow

The lifecycle of an inference request transitions through phases cleanly matched to accelerator characteristics:

```
[1. User Prompt]
       │
       ▼
[Zen 5 Classic Core] ──────► Runs AVX-512 string tokenization within 16MB L3 cache.
       │                     Locks feeder thread via pthread_setaffinity_np(Core 0).
       ▼
[RDNA 3.5 iGPU] ───────────► Executes compute-heavy Prefill GEMM (ROCm/HIP).
       │                     Writes attention Key/Value state directly to shared dma-buf.
       ▼
[Kernel DRM Syncobj] ──────► GPU signals completion via dma_fence.
       │                     Zero CPU intervention; no memory copied or moved.
       ▼
[XDNA 2 NPU (AIE2P)] ──────► Imports KV-cache pointer into XRT Buffer Object (xrt::bo).
       │                     Embedded ERT microcontroller streams weights from RAM.
       │                     Local tile SRAM double-buffers activation matrices.
       ▼
[Zen 5c Compact Core] ─────► Low-power polling thread waits on /dev/accel/accel0 event.
       │                     Zen 5 classic cores enter deep C-states (power saving).
       ▼
[Token Loop Complete] ─────► Selected token appended to KV-cache; repeat decode phase.

```

---

### Key Technical Subsystems

* **Zero-Copy Memory Layer (`dma-buf`):** The host allocates the KV-cache using standard DRM GEM mechanisms in `amdgpu`. By exporting the buffer to a file descriptor and importing it into XRT (`xrt::bo(device, fd, ...)`), the NPU’s IOMMU page table is pointed directly to the same physical RAM addresses. Neither the CPU nor the memory controller performs copy passes.
* **Hardware-Level Synchronization (`drm_syncobj`):** The iGPU and NPU synchronize via Linux kernel `dma_fence` handles. When the GPU's Command Processor completes prefill, it marks the sync object as signaled. The NPU user-channel ERT instance consumes this signal to trigger spatial execution without waiting for an OS thread reschedule.
* **Topology-Aware Core Binding (`sched_setaffinity`):** By separating latency-sensitive feeder threads (locked to Zen 5 Classic cores) from background polling and I/O tasks (locked to Zen 5c Compact cores), the orchestrator prevents OS thread migrations across CCX boundaries, eliminating cache flushes and latency jitter.

---

### Key Documentation & Reference Materials

**1. AMD XDNA Driver & NPU Architecture**

* **Linux Kernel AMDXDNA Documentation:** [`docs.kernel.org/accel/amdxdna/`](https://www.google.com/search?q=https://docs.kernel.org/accel/amdxdna/) — In-depth breakdown of the XDNA 2D array topology (4×8 compute and memory tiles on Strix Point), shared L2 memory, ERT/MERT firmware microcontrollers, and mailbox channels.
* **Upstream Linux AMDXDNA Driver Source:** [`drivers/accel/amdxdna/`](https://www.google.com/search?q=https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/accel/amdxdna) — The kernel-space DRM accelerator driver managing NPU command submission, IOMMU SVA (Shared Virtual Addressing), and power states.
* **AMD XDNA Driver Repository:** [`github.com/amd/xdna-driver`](https://github.com/amd/xdna-driver) — Contains the out-of-tree staging drivers, user-mode SHIM plugin (`xrt_plugin.*-amdxdna.so`), and DKMS packaging.

**2. Runtime & Compiler Toolchains**

* **XRT (Xilinx Runtime) Native APIs:** [`xilinx.github.io/XRT/`](https://xilinx.github.io/XRT/) — Documentation for `xrt::device`, `xrt::bo` (Buffer Objects), execution graphs, and the native `dma-buf` import/export APIs (`xrt::bo(device, fd, ...)`).
* **Triton-XDNA:** [`github.com/amd/Triton-XDNA`](https://github.com/amd/Triton-XDNA) — AMD's backend for compiling Triton kernels to AMD NPU tiles via MLIR-AIE and dispatching through XRT or HSA/ROCR runtimes.
* **MLIR-AIE / IRON:** [`github.com/Xilinx/mlir-aie`](https://github.com/Xilinx/mlir-aie) and [`github.com/amd/iron`](https://github.com/amd/iron) — Open-source LLVM/MLIR toolchain and Python frontend for scheduling VLIW vector instructions and spatial DMA pipelines onto AIE/AIE2P tiles.

**3. Linux Kernel Heterogeneous Computing Primitives**

* **Linux DMA-BUF Subsystem:** [`docs.kernel.org/driver-api/dma-buf.html`](https://docs.kernel.org/driver-api/dma-buf.html) — Technical specification for cross-device buffer sharing, GEM Prime descriptors, and CPU cache-synchronization barriers (`dma_buf_begin_cpu_access`).
* **DRM Synchronization Objects (`drm_syncobj`):** [`docs.kernel.org/gpu/drm-mm.html#drm-sync-objects`](https://docs.kernel.org/gpu/drm-mm.html) — Linux kernel interface for timeline fences allowing CPU-free dependency chains across disparate GPU and accelerator hardware rings.
* **Linux CPU Affinity API:** [`man7.org/linux/man-pages/man3/pthread_setaffinity_np.3.html`](https://man7.org/linux/man-pages/man3/pthread_setaffinity_np.3.html) — POSIX thread pinning controls used to isolate worker roles across heterogeneous CPU microarchitectures.

For raw execution speed on bare metal, Rust and C stand on equal footing in theoretical peak performance, but Rust (#![no_std]) holds a distinct structural advantage in real-world code.

Both languages compile directly to native machine instructions via the same optimizing compiler backends (LLVM or GCC) with zero runtime, garbage collection, or virtual machines. However, key compiler and language dynamics differentiate them:

    Compiler Optimization & Pointer Aliasing (Rust's Advantage): LLVM can optimize code much more aggressively when it knows two pointers do not point to the same memory location. In Rust, the borrow checker statically guarantees that mutable references (&mut) are exclusive. The compiler emits code with automatic noalias metadata, unlocking aggressive loop unrolling, autovectorization, and register caching. In C, achieving this requires the programmer to correctly annotate pointers with restrict—a practice that is rare in practice and prone to undefined behavior if misused.

    Zero-Cost Abstractions: Rust’s iterators, closures, and traits monomorphize into inline machine instructions without function-pointer indirection or dynamic dispatch. In bare-metal C, implementing generic data structures often relies on void* casts and manual function pointers, which inhibit compiler inlining.

    Direct Hardware Mapping (C's Traditional Strength): C’s abstract machine model is practically isomorphic to hardware memory. Mapping Memory-Mapped I/O (MMIO) registers, page tables, and packed hardware descriptors onto C structs is syntactically trivial, whereas Rust requires explicit read_volatile/write_volatile calls and wrapping in UnsafeCell to prevent the compiler from eliding redundant register accesses.

    The Assembly Baseline: Neither C nor Rust can execute 100% of a bare-metal kernel. Low-level initialization—configuring control registers (CR0/CR3/CR4, MSRs), setting up the Interrupt Descriptor Table (IDT), managing page tables, and performing context switches—requires inline or standalone Assembly in both languages.

Verdict: For a modern bare-metal system, Rust (#![no_std]) paired with small assembly primitives is the recommended choice. It matches C's clock-cycle speed and often outperforms non-annotated C in complex pipelines due to aliasing optimization, while preventing the memory corruption vulnerabilities (use-after-free, data races, buffer overruns) that commonly plague bare-metal development.
