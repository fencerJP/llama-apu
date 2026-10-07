# ROCm Release Audit → llama-apu Feature Utilization & Optimization Plans

**Date:** 2026-10-06
**Author:** Lean (audit for Dan)
**Scope:** Last 3 ROCm Core SDK releases vs. `llama-apu/llama.cpp` (`src/ggml-apu-*`, `src/llama-apu-*`)

---

## 0. System baseline (measured)

| Item | Value |
| :--- | :--- |
| Host | `fencer-mini`, Ryzen AI 9 HX 470 w/ Radeon 890M |
| iGPU | **gfx1150** (RDNA 3.5, Strix Point) |
| NPU | RyzenAI-npu4 (XDNA 2 / AIE2P) |
| ROCm installed | **7.2.4** (`/opt/rocm`, `/opt/rocm-7.2.4`), amd-smi 26.2.2 |
| Build | ROCm HIP, `GPU_TARGETS=gfx1150`, Release |
| GGML flags | `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`, `GGML_HIP_RCCL=OFF` |
| BLAS linkage | `roc::rocblas` + `roc::hipblas` only — **no hipBLASLt** (`ggml/src/ggml-hip/CMakeLists.txt:144`) |

> **Key gap #0:** the reviewed releases (10.1 / 10.0 / 7.14.x) are *newer than the installed 7.2.4*. The single largest lever is not a code change — it is the stack upgrade. Every plan below assumes either ROCm 7.14.1 or 10.0/10.1 as a prerequisite, so each plan lists an **upgrade-gate** check.

---

## 1. Last 3 ROCm releases — inference-relevant deltas

### ROCm Core SDK 10.1.0 (`therock-10.1`, 2026-10-05)
- **hipBLASLt / CK:** WMMA instance support on new targets; **FMHA forward (hdim=128) retuned on gfx11/gfx12**; gfx1151/gfx1200 (Navi) SystemDB tuned entries so immediate-mode lookups skip generic heuristics.
- **rocBLAS:** L3 `gemm` `m==1 || n==1` now routes to **gemv kernels** (previously only in `gemm_ex`); **on gfx11 the per-precision heuristics guarding this path are bypassed** (decode-relevant). L2 `gemv` TransA==N small-m/large-n now split across the grid.
- **gfx1153 / Gorgon Point** roofline + counter support (adjacent to gfx1150).
- **rocSOLVER** 2-stage tridiagonal eigensolvers (not inference-relevant).
- **RPP** added to Core SDK (vision, not LLM decode).

### ROCm Core SDK 10.0.0 (`therock-10.0`, 2026-08-26)
- **Composable Kernel 1.2.0:** *"Improved performance of row-column quantized **a8w8 GEMM** through better instruction scheduling in the eight-waves pipeline, wider epilogue stores, and nontemporal C/D memory access."* Plus bias + large-tensor support to the CK Tile quantized GEMM.
- **HIP 10.0.0:**
  - **Stream Ordered Memory Allocator** parity (`hipMemGetDefaultMemPool`).
  - **NUMA-aware VMM** (`hipMemLocationTypeHostNuma`) in `hipMemCreate`.
  - `hipMemcpy2D` small-row/large-row-count path now single shader-based copy (≤256 rows unchanged).
  - `hipMemcpyBatchAsync` split per-device; `hipEventRecord` coalescing via `hipEventDisableTiming`.
  - Stream capture for `hipStreamWaitValue*` / `hipStreamWriteValue*` / `hipBatchMemOp`.
- **RCCL 2.30.7** fused AllGatherV (multi-node; **N/A single device**).
- **ROCm Compute Profiler 3.8.0:** *gfx1150/gfx1151/gfx1152 roofline + benchmarking support; gfx11xx uses **WMMA, replacing MFMA**;* Dual VALU (VOPD) metric reported for gfx115x.
- **hipCUB/rocThrust:** PyTorch <2.11 compatibility removed (framework cohort note).

### ROCm Core SDK 7.14.0 / 7.14.1 (`therock-7.14*`)
- **hipBLASLt 1.4.1:** `hipBLASLt_ext::isSolutionSupported()` used by new **rocBLAS→hipBLASLt integration**.
- **rocBLAS:** `BUILD_WITH_HIPBLASLT_ONLY=ON`; deprecates `ROCBLAS_USE_HIPBLASLT_BATCHED` (*"recent optimizations mean hipBLASLt no longer needs to be disabled for batched operations"*).
- **rocWMMA 2.2.x:** gfx1150 target; moved to `rocm-libraries` monorepo.
- **rocprof-compute:** gfx1150/gfx1152 counter + roofline; Strix Halo (gfx1151) workload folder renamed `strix_halo` → `rdna35_halo`; known issue: gfx1151 `TCP_REQ_sum` zero in single-pass, `$max_mclk` unpopulated (use `amd-smi` + `--specs-correction`).
- **hipBLASLt history (relevant):** gfx1150/gfx1151 enabled since 0.12.x; **block scaling via `HIPBLASLT_MATMUL_DESC_A/B_SCALE_MODE` / `MATRIX_SCALE_VEC32_UE8M0`**, FP8/BF8 swizzle orders, **MX FP4/FP6/FP8 microscaling** dtype support, `BF16/INT8 WMMA GEMM for Navi3x/Navi4x`, LSE output + attention fusion.

---

## 2. Utilization audit

Legend: ✅ used · ⚠️ partial / indirect · ❌ not used

| # | ROCm feature | Where it would land | Status | Evidence |
| :-- | :--- | :--- | :--- | :--- |
| F1 | **hipBLASLt WMMA GEMM (gfx11xx)** | GPU prefill GEMM (`ggml-hip`) | ❌ | Only `roc::rocblas`/`roc::hipblas` linked; no hipBLASLt target. `ROCBLAS_USE_HIPBLASLT` not enabled. |
| F2 | **CK a8w8 row-column quantized GEMM** | int8 prefill | ❌ | Custom quant path is TQ2_0→`apu_tace16_tile` for the **NPU**; GPU quantized GEMM uses ggml MMQ, not CK a8w8/MX. |
| F3 | **rocBLAS gemv fast paths (m==1, gfx11 bypass)** | decode matvec | ⚠️ | ggml-hip calls hipBLAS/rocBLAS for some matmuls, but installed 7.2.4 predates the 10.1 path. |
| F4 | **hipBLASLt/MIOpen+ CK FMHA (hdim=128) & LSE fusion** | attention | ⚠️ | `ggml-apu-kv.cpp` forces flash-attn on for Q4_0 KV; but attention runs ggml's own `fattn` (`amd_wmma_available`), not ROCm FMHA. |
| F5 | **HIP stream-ordered mempool + NUMA VMM** | KV/scratch/WEIGHTS alloc | ❌ | Alloc via custom `apu_gem_buffer` (GEM/dma-buf) + `/proc/meminfo` governor; no HIP mempool. |
| F6 | **HIP memcpy2D / batch-copy opt** | weight/activation staging | ❌ | Staging done in `ggml-apu-convert.cpp` / loader; no `hipMemcpyBatchAsync`. |
| F7 | **rocWMMA gfx1150 tiles** | tensor cores | ❌ | ggml uses its own `mma.cuh` WMMA intrinsics; rocWMMA not linked. |
| F8 | **rocprof-compute gfx115x WMMA/Roofline** | measurement | ❌ | Not used anywhere; would enable trustworthy A/B attribution. |
| F9 | **RCCL AllGatherV fused** | multi-GPU | N/A | Single device; `GGML_HIP_RCCL=OFF` correct. |
| F10 | **MIOpen gfx1151 SystemDB** | conv (vision) | N/A | LLM decode has no conv. |

**Custom-APU substrate (already present, mostly NPU-side):** GEM/PRIME dma-buf bridge (`ggml-apu-bridge.cpp`), DRM timeline syncobj dispatch (`llama-apu-dispatch.cpp`), Q4_0 KV (`ggml-apu-kv.cpp`), MoE router SRAM pinning (`ggml-apu-moe.cpp`), TQ2_0 + T-ACE lowering (`ggml-apu-lowquant.cpp`), XCLBIN synthesis (`ggml-apu-xclbin.cpp`), speculative coordinator (`ggml-apu-spec.cpp`), Sarathi scheduler (phase-9). These are **not** superseded by the ROCm features above — they target the NPU/heterogeneous path. The gap is entirely on the **iGPU (ROCm) side**, which is still a plain ggml-hip build.

---

## 3. Optimization plans + A/B tests

### Common A/B protocol (applies to every plan)
Use the existing harness so results are comparable across plans.
- **Runner:** `tools/llama-bench/llama-bench` + `scripts/compare-llama-bench.py`.
- **Workload matrix:** one dense (`Qwen3-8B`-class) + one MoE (`Qwen3-Coder-Next`-class, from CHANGELOG 0.7.2). Fixed `-c 4096`, `-ngl 99`, `-fa on`.
- **Metrics:** `pp512` (prefill tok/s), `tg128` (decode tok/s), TTFT (ms), quant/attn kernel time, peak RSS + dma-buf bytes.
- **Isolation:** 3× warmup, 5× measured, report median + p95; `nice`/governor pinned; kill background load.
- **Profiling:** `rocprof-compute profile` on gfx1150 (needs F8/ROCm ≥7.14), `--specs-correction` per known gfx1151 caveat.
- **Acceptance:** ≥3% `tg128` or ≥5% `pp512` gain with ≤1% quality delta on a fixed perplexity set; no RSS regression >5%.
- **Control:** identical command with the feature off / previous stack.

---

### P1 — hipBLASLt WMMA backend for iGPU prefill (F1)  ⭐ highest expected impact
**Hypothesis:** gfx1150 is RDNA 3.5 (WMMA, *no* MFMA). ROCm 7.14/10.0 hipBLASLt has WMMA-tuned kernels for gfx1150/1151; default rocBLAS/Tensile paths predate that tuning.
**Steps**
1. Prereq: upgrade to ROCm ≥7.14.1 (see P0 below).
2. Add `find_package(hipblaslt)` + link `roc::hipblaslt` in `ggml/src/ggml-hip/CMakeLists.txt`; add a `GGML_HIP_USE_HIPBLASLT` option.
3. Route large prefill matmuls through hipBLASLt with `hipblasLtMatmulAlgo` cached per shape (use `isSolutionSupported()` to gate by GPU/problem type).
4. Keep rocBLAS fallback for tiny/decode shapes (see P3).
5. Tune via `HIPBLASLT_TUNING_*` offline for the model's actual GEMM shapes.
**A/B:** `-DGGML_HIP_USE_HIPBLASLT=ON` vs OFF. Expect gain concentrated in `pp512` (batch GEMM). **Pass:** ≥5% `pp512` no decode regression.

---

### P2 — a8w8 / MX quantized GEMM on iGPU (F2)
**Hypothesis:** CK 1.2.0's row-column **a8w8** GEMM (eight-wave schedule, wider epilogue, nontemporal C/D) beats current int8 path; MX FP4/FP8 block scaling is available on gfx1150.
**Steps**
1. Confirm CK 1.2.0 a8w8 kernels ship in the chosen ROCm build and target gfx1150 (`ck` samples / `hipblaslt` ext op `HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`).
2. Add an int8 (a8w8) and optionally MX-FP8 path alongside ggml MMQ, selected per-layer by shape.
3. Coordinate with the **existing TQ2_0/NPU** path: TQ2_0 stays on NPU; a8w8 is the **iGPU** prefill complement. Document routing so GPU≠NPU tensors don't double-quantize.
**A/B:** int8-a8w8 vs `Q4_0`/`Q5_K` MMQ baseline on the same weights (convert both). **Pass:** ≥10% prefill over MMQ at equal or better perplexity. *Caveat:* adds a second quant format — gate behind a flag until proven.

---

### P3 — rocBLAS decode gemv path (F3)
**Hypothesis:** 10.1's L3 `gemm` m==1/n==1 → gemv routing (gfx11 heuristics bypassed) accelerates token-by-token decode.
**Steps**
1. After upgrade, ensure decode matmuls hit rocBLAS ≥10.1 (not hipBLASLt) for `m==1` shapes.
2. If ggml bypasses BLAS for decode (MMVQ), add a guarded rocBLAS gemv route for the largest decode matmuls and compare against MMVQ.
3. Benchmark `ROCBLAS_USE_HIPBLASLT=0/1` to let rocBLAS pick.
**A/B:** gemv route vs current MMVQ decode. **Pass:** ≥3% `tg128`, no prefill loss.

---

### P4 — FMHA attention alignment (F4)
**Hypothesis:** ROCm 10.1 retuned FMHA hdim=128 on gfx11/gfx12, and hipBLASLt exposes LSE/attention fusion — potentially better than ggml `fattn` at long context and GQA.
**Steps**
1. Profile current FA kernel (`amd_wmma_available` path) with rocprof-compute (P6).
2. Evaluate whether ROCm FMHA is reachable for ggml (likely via a custom op or via hipBLASLt attention-fusion ext API) for `hdim=128`, GQA heads used by the target models.
3. If reachable, add an opt-in `GGML_APU_ATTENTION=rocm` path; otherwise document as unavailable and keep ggml FA.
**A/B:** ROCm FMHA vs ggml FA at `-c 4096` and `-c 32768`. **Pass:** ≥5% TTFT or decode gain at long ctx, identical output perplexity.

---

### P5 — HIP memory pools + NUMA VMM (F5/F6)
**Hypothesis:** `hipMemGetDefaultMemPool` reduces allocator churn on the unified-memory path; NUMA host VMM + improved `hipMemcpy2D`/batch copy speed weight/scratch staging.
**Steps**
1. Wrap the bridge/kv scratch allocator to prefer HIP stream-ordered pool where the buffer will be written by GPU (keep GEM/dma-buf for NPU-visible tensors).
2. Route batched weight staging through `hipMemcpyBatchAsync`.
3. Re-check the `GGML_HIP_NO_VMM=ON` setting — NUMA VMM (10.0) may now be worth enabling on UMA.
**A/B:** pool+batch-copy vs current allocator, same model. **Pass:** ≥2% `tg128`, lower RSS variance, no leak over 1h soak.

---

### P6 — Measurement substrate (F8) — enable first
**Steps:** install rocprof-compute ≥3.8.0, accept the Strix Halo/`rdna35_halo` folder rename, calibrate with `amd-smi` max mem clock + `--specs-correction`, capture pre-feature baselines for P1–P5.
**Deliverable:** one roofline + kernel-time baseline per target model, archived next to bench results so every later A/B has a control.

---

### P0 — Stack upgrade (prerequisite for P1–P6)
**Decision needed:** ROCm **7.14.1** (Ryzen-AI-focused line, gfx1151 SystemDBs) vs **10.0/10.1** (data-center line, gfx1150/1151 also listed). Both list gfx1150/1151; pick based on which has validated Ryzen AI / XDNA2 + firmware alignment. ⚠️ Firmware/driver alignment is mandatory (ROCm requires a coordinated firmware+driver+userspace stack) — do not upgrade the userspace RPMs alone.
**Gate:** `rocminfo` shows gfx1150, `amd-smi version` matches, llama.cpp builds and passes current tests before any feature work.

---

## 4. Suggested sequencing

```
P0 upgrade ─┬─ P6 measurement baseline ─┬─ P1 hipBLASLt/WMMA  (biggest win)
            │                          ├─ P3 rocBLAS gemv
            │                          ├─ P5 HIP mempool/VMM
            │                          ├─ P4 FMHA
            └──────────────────────────┴─ P2 a8w8/MX (needs quant-format decision)
```

Ship order by risk/reward: **P1 → P3 → P5 → P4 → P2**. Each lands behind its own CMake/env flag so A/B is a rebuild toggle, not a branch.

---

## 5. Open questions for Dan
1. **ROCm line:** 7.14.1 vs 10.0/10.1 for this APU? (Affects all plans.)
2. **Quant strategy:** keep TQ2_0/NPU exclusive, or add iGPU int8-a8w8 (P2) — two formats to maintain?
3. **Workload targets:** which models define "done" for the A/B acceptance set?