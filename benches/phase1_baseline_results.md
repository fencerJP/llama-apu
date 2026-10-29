# Phase 1 Baseline Benchmark Results

**Date:** 2026-10-30  
**Branch:** `feat/phase1-spec-kv-rdna35` (`e8bf93ccf`)  
**Hardware Platform:** AMD Ryzen AI 9 HX 470 (`gfx1150`), Radeon 890M Graphics, 121 GiB Unified RAM  
**Backends:** ROCm 10.1 (`libggml-hip.so`), DRM Syncobj Timeline (`/dev/dri/renderD128`)

---

## 1. Summary of Milestones Implemented in Phase 1

1. **Dynamic Confidence-Gated Speculative Drafting:**
   - EMA feedback throttling and context ladder in `src/ggml-apu-spec.cpp` & `common/speculative.cpp`.
   - Adaptive DRM timeline sync (`drm_syncobj`) with intra-GPU bypass when execution is single-stream (sync latency: 13–14 µs).
2. **MTP Boundary Ring Rollback:**
   - 32-slot circular state ring (`ring_h`, `ring_pos`, `ring_len`, `ring_head`) decoupled from context sequences in `common/speculative.cpp`.
   - Trailing prompt cache rollback salvage wired in `tools/server/server-context.cpp`.
3. **Asymmetric KV Cache with Boundary Guard ($B=2$):**
   - Configurable via `--kv-cache-type asym` and `--kv-boundary-layers <N>`.
   - Keys: `GGML_TYPE_Q8_0` (preserves attention routing precision).
   - Values: `GGML_TYPE_Q4_0` (halves memory bandwidth requirements for value projections).
   - Boundary layers guarded from quantization (`il < B || il + B >= n_layer` forced to `Q8_0`).

---

## 2. Microbenchmarks: Standard vs Asymmetric KV Cache (`llama-bench`)

*Parameters: prompt 512 tokens, generation 128 tokens, offload `-ngl 99`.*

| Model | Parameters | KV Cache Mode | Key Type | Value Type | Prompt Throughput (t/s) | Generation (t/s) | KV Footprint Savings |
|---|---|---|---|---|---|---|---|
| `ornith-1.0-9b` | 8.95 B | Standard | `f16` | `f16` | 272.07 ± 22.07 | 9.76 ± 0.01 | Baseline (16-bit) |
| `ornith-1.0-9b` | 8.95 B | **Asymmetric** | `q8_0` | `q4_0` | **291.33 ± 1.39** (+7.1%) | 9.72 ± 0.03 | **-62.5%** (6-bit avg) |
| `Qwopus3.6-27B` | 27.32 B | Standard | `f16` | `f16` | 85.23 ± 0.00 | 3.20 ± 0.00 | Baseline (16-bit) |
| `Qwopus3.6-27B` | 27.32 B | **Asymmetric** | `q8_0` | `q4_0` | **87.33 ± 0.00** (+2.5%) | 3.17 ± 0.00 | **-62.5%** (6-bit avg) |
| `Nanbeige4.3-33B` | 33.12 B | Standard | `f16` | `f16` | 81.42 ± 0.00 | 2.76 ± 0.00 | Baseline (16-bit) |
| `Nanbeige4.3-33B` | 33.12 B | **Asymmetric** | `q8_0` | `q4_0` | **81.74 ± 0.00** (+0.4%) | 2.79 ± 0.00 (+1.1%) | **-62.5%** (6-bit avg) |

---

## 3. End-to-End Speculative Decoding & Telemetry (`llama-cli`)

### `Qwopus3.6-27B-Coder-Compat-MTP-Q4_K_M.gguf`

| Configuration | Generation Throughput | Speedup | Acceptance Rate ($\alpha$) | Sync Latency | Zero-Copy Handoff |
|---|---|---|---|---|---|
| Autoregressive Baseline | 3.20 t/s | 1.00x | N/A | N/A | PASS |
| **MTP + Dynamic Confidence Gating + DRM Sync** | **4.90 t/s** | **+53.1%** | **85.71%** | 13.00 µs | PASS (host memcpy: 0) |

**Telemetry Breakdown:**
- **Mode:** `AUTO` (coordinated draft & target verification passes)
- **DRM Timeline Sync:** `ENABLED` on `/dev/dri/renderD128`
- **Draft Batches:** 19
- **Candidate Tokens Drafted:** 35
- **Candidate Tokens Accepted:** 30
- **Iterative Candidate Tokens Rejected:** 5
- **Acceptance Rate:** 85.71%
- **KV Cache Rollback Passes:** 2 (handled cleanly by ring rollback without state corruption)
- **Timeline Sync Passes:** 19 (average latency: 13.00 µs)
- **Fallback Triggered:** `NO`
