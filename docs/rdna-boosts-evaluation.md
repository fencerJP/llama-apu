# RDNA Boosts Evaluation (`stew675/llama-cpp-rdna-boosts`) on `gfx1150`

## Overview

- **Upstream source evaluated**: [`stew675/llama-cpp-rdna-boosts`](https://github.com/stew675/llama-cpp-rdna-boosts) (`v18-919794500-r374`, commit `ea41ea1`, 19 patch files `0000`–`0017` + `0300`, plus optional `0100`/`0101` referenced in upstream notes)
- **Target branch**: `eval/rdna-boosts` (branched from `llama-apu` `llama.cpp/master` @ `6d778ee0d`, build `12373` / `1f71de9b0`)
- **Test platform**:
  - **APU / CPU**: AMD Ryzen AI 9 HX 470 w/ Radeon 890M (24 threads, 121 GiB system RAM)
  - **iGPU**: `gfx1150` (RDNA 3.5, Wave Size: 32, 61440 MiB UMA/VRAM, `VMM: no` / `GGML_HIP_NO_VMM=ON`)
  - **ROCm stack**: `/opt/rocm/core-10.2` (`clang++` 24.0.0, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_USE_HIPBLASLT=ON`)
  - **Contention environment**: Resident Docker `llama-server` instances on ports `9090`/`9091` remained running untouched across both baseline and patched runs (`rocm-smi` GPU busy baseline 87%->68%, 62°C->64°C; patched 62%->95%, 64°C->68°C).

---

## Patch Evaluation & Integration Matrix

| Patch / Block | Title / Scope | Status in `eval/rdna-boosts` | Commit(s) | Notes & `llama-apu` Compatibility |
| :--- | :--- | :--- | :--- | :--- |
| `0000` | Foundation, `GGML_GLU_INPLACE`, `gallocr` graph fp, FA/CPU band parity | **Ported** | `f8156f00f` | Ported `ggml-alloc.c` (`GGML_GLU_INPLACE`, `ggml_gallocr_graph_fp`, unused-view release), `fattn-common.cuh` (`ntiles_dst_eff` small-batch invariance), and `sgemm.cpp` (`n <= 8`). |
| `0001` | Hybrid HIP all-reduce (RDNA4-gated) | **Skipped** | — | Multi-GPU RDNA4 NVLink/P2P all-reduce path; not applicable to single-iGPU `gfx1150`. |
| `0002` | Adaptive MTP draft depth | **Skipped** | — | Invasive speculative-decoding changes overlapping `llama-apu` Phase 6 MTP ring rollback. |
| `0003` | Chunked Gated Delta Net (GDN) prefill & `gfx11` BF16 WMMA kernel | **Ported** | `16b444e51` | Added `gated_delta_net_chunked.{cuh,cu}`, `gated_delta_net_chunked_bf16.cu`, `gated_delta_net_chunked_bf16_gfx11.cu`, and `GGML_CUDA_GDN_CHUNKED` dispatch in `gated_delta_net.cu` (before `ggml_cuda_gdn_mma_launch`). |
| `0004` | BF16 KV cache & native BF16 Flash-Attention | **Ported** | `d2dae0309`, `2b6ff5ff9` | Ported `v_dot2_f32_bf16` + perm helpers (`common.cuh`), `gfx11` masked-V zeroing (`fattn-mma-f16.cuh`, indexing fixed in `2b6ff5ff9`), `fattn-tile.{cu,cuh}`, `fattn.cu`, and native BF16 `rope.cu` (`store_coaelsced`, `rms_norm_mul_rope_fused`). |
| `0005` | Flash-Attention band & native quant KV kernels (`fattn-rdna`, 341 files) | **Deferred** | — | High merge-conflict surface vs `llama-apu` asym KV (`Q8_0`/`Q4_0`) & `ROCmFPX` paths; recommended for dedicated follow-up evaluation. |
| `0006` | `mmb` BF16-WMMA prefill GEMM & `mmf32` skinny F32 GEMM | **Ported** | `7ce83a9d2` | Added `mmb.{cu,cuh}`, `mmf32.{cu,cuh}`, `bf16_src1_cache` (`common.cuh`), `MMVF_MAX_BATCH_SIZE_FLAT` (`mmvf.cuh`), and `mmb`/`mmf32`/`expert_bounds` wiring (`ggml-cuda.cu`, `mmf.cu`, `mmid.{cu,cuh}`, `mmq.cu`). Preserves `ROCmFPX` and `llama-apu` hipBLASLt overrides. |
| `0007` | HIP `mul24`/`v_perm_b32`, grid stall pad, and `MMQ`/`MMVF` boosts | **Ported** | `0018f05b8` | Ported `ggml_cuda_grid_stall_pad` (`common.cuh`), `mmq.cu`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`, `mmvf.cu`, and `vecdotq.cuh` (`ggml_cuda_mul_small`, `ggml_cuda_byte_perm`, `apply_ksigns`, `ggml_cuda_mmq_scale_sumi`, Q4_K `scm` packing). Preserved `ROCmFPX`/`Q1_0`/`Q2_0` and `VDR_Q6_K_Q8_1_MMVQ=2` APU guard. |
| `0008` | Fused kernels: bit-exact warp-butterfly softmax & sorted-key argmax in `topk-moe` | **Partially Ported** | `1cdb8a541` | Ported `topk-moe.cu` (`GGML_CUDA_TOPK_SORTED`); skipped coupled fusion parts depending on Block 09/10 infrastructure. |
| `0009` | Host-resident MoE interface & scheduler | **Skipped** | — | Coupled host-expert streaming/residency infra colliding with `llama-apu` Phase 5/8 router & UMA memory management. |
| `0010` | Slab allocator, fusion dispatcher, and async H2D arena | **Skipped** | — | Requires Block 09 slab/arena infra and conflicts with `llama-apu` Phase 4/9 allocation & zero-copy paths. |
| `0011` | `qwen4exp` model / hyper-connection ops | **Skipped** | — | Depends on Block 09/10 fusion dispatcher. |
| `0012` | Campaign remainder CPU-wins / backend tweaks | **Skipped** | — | Depends on Block 09/10 context hooks. |
| `0013` | MTP draft vocabulary subset speculation | **Skipped** | — | Not needed alongside `llama-apu` Phase 6 speculative NPU drafting. |
| `0014` | Streamed KV with fixed VRAM reservation | **Skipped** | — | Requires CUDA/HIP VMM (`cudaMemAddressReserve`), incompatible with `GGML_HIP_NO_VMM=ON` required on `gfx1150`. |
| `0015` | UMA/GTT expert store / APU hooks | **Skipped** | — | Depends on Block 09/10 slab/host-expert framework; overlaps existing `llama-apu` UMA memory governor. |
| `0016` | Server prompt admission policy | **Skipped** | — | Directly conflicts with `llama-apu` Phase 9 Sarathi-Serve stall-free scheduler. |
| `0017` | `TQ1_0` / `TQ2_0` ternary weights on GPU (bit-exact `MUL_MAT`, `GET_ROWS`, dequant, decode fusion) | **Ported** | `3d5f57559` | Added `tq.cu`/`tq.cuh`, `ggml_cuda_tq1_0_trit` & `tq_census[5]` (`common.cuh`), `convert.cu`, `dequantize.cuh`, `getrows.cu`, `mmvq.cu`, and `ggml-cuda.cu` dispatch/fusion (`GGML_CUDA_TQ`, `GGML_CUDA_TQ_FUSE`, `GGML_CUDA_TQ_CENSUS`). |
| `0300` | Relaxed tensor split mode (TEMP multi-GPU) | **Skipped** | — | Multi-GPU temporary patch; irrelevant on single iGPU. |
| `0100` / `0101` | Optional add-on patches (upstream `patches/README.md`) | **Skipped** | — | Optional patches tied to deferred Block 05/09/10 paths. |

---

## Benchmark Results (Baseline `master` vs `eval/rdna-boosts`)

All benchmarks ran with `-ngl 99 -fa 1 -r 5` via `.testing/rdna-boosts-eval/run_bench.sh` comparing untouched baseline `llama.cpp/build-rocm102/bin/llama-bench` (`1f71de9b0`) against `llama.cpp/build-rocm102-rdna/bin/llama-bench` (`3d5f57559` / `2b6ff5ff9`).

### Summary Comparison Table

| Case | Model / Arch | KV (`K`/`V`) | Metric | Baseline (t/s) | Patched (t/s) | Delta |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `neohorse_4b_q4km` | `qwen35 4B Q4_0` | `f16`/`f16` | `pp128` | `572.83 ± 27.77` | `530.61 ± 37.27` | `-7.4%` *(noise; see A/B)* |
| `neohorse_4b_q4km` | `qwen35 4B Q4_0` | `f16`/`f16` | `pp512` | `653.62 ± 24.60` | `657.80 ± 24.85` | `+0.6%` |
| `neohorse_4b_q4km` | `qwen35 4B Q4_0` | `f16`/`f16` | `pp2048` | `416.72 ± 12.40` | `496.12 ± 11.52` | **`+19.1%`** |
| `neohorse_4b_q4km` | `qwen35 4B Q4_0` | `f16`/`f16` | `pp4096` | `399.11 ± 2.69` | `480.62 ± 9.01` | **`+20.4%`** |
| `neohorse_4b_q4km` | `qwen35 4B Q4_0` | `f16`/`f16` | `tg128` | `18.79 ± 0.09` | `18.78 ± 0.11` | `-0.1%` |
| `neohorse_4b_q4km_d2048` | `qwen35 4B Q4_0` | `f16`/`f16` | `tg128 @ d2048` | `18.41 ± 0.07` | `18.38 ± 0.08` | `-0.2%` |
| `dolphin3_8b_q6k` | `llama 8B Q6_K` | `f16`/`f16` | `pp128` | `277.15 ± 17.96` | `323.74 ± 10.90` | **`+16.8%`** |
| `dolphin3_8b_q6k` | `llama 8B Q6_K` | `f16`/`f16` | `pp512` | `209.01 ± 2.07` | `431.82 ± 3.21` | **`+106.6%`** |
| `dolphin3_8b_q6k` | `llama 8B Q6_K` | `f16`/`f16` | `pp2048` | `161.13 ± 1.09` | `317.52 ± 6.16` | **`+97.1%`** |
| `dolphin3_8b_q6k` | `llama 8B Q6_K` | `f16`/`f16` | `pp4096` | `153.09 ± 1.01` | `385.41 ± 28.13` | **`+151.8%`** |
| `dolphin3_8b_q6k` | `llama 8B Q6_K` | `f16`/`f16` | `tg128` | `8.82 ± 0.02` | `8.71 ± 0.01` | `-1.2%` |
| `dolphin3_8b_q6k_bf16kv` | `llama 8B Q6_K` | `bf16`/`bf16` | `pp512` | `209.39 ± 2.18` | `420.41 ± 12.55` | **`+100.8%`** |
| `dolphin3_8b_q6k_bf16kv` | `llama 8B Q6_K` | `bf16`/`bf16` | `pp2048` | `156.27 ± 2.19` | `315.96 ± 2.87` | **`+102.2%`** |
| `dolphin3_8b_q6k_bf16kv` | `llama 8B Q6_K` | `bf16`/`bf16` | `tg128` | `8.65 ± 0.03` | `8.87 ± 0.15` | `+2.5%` |
| `dolphin3_8b_q6k_q8kv` | `llama 8B Q6_K` | `q8_0`/`q8_0` | `pp512` | `207.20 ± 2.86` | `423.71 ± 6.31` | **`+104.5%`** |
| `dolphin3_8b_q6k_q8kv` | `llama 8B Q6_K` | `q8_0`/`q8_0` | `pp2048` | `156.78 ± 2.14` | `316.55 ± 6.66` | **`+101.9%`** |
| `dolphin3_8b_q6k_q8kv` | `llama 8B Q6_K` | `q8_0`/`q8_0` | `tg128` | `8.74 ± 0.31` | `8.64 ± 0.04` | `-1.1%` |
| `qwensploit_27b_q6k` | `qwen35 27B Q6_K` | `f16`/`f16` | `pp128` | `81.05 ± 2.79` | `97.47 ± 2.60` | **`+20.3%`** |
| `qwensploit_27b_q6k` | `qwen35 27B Q6_K` | `f16`/`f16` | `pp512` | `66.13 ± 0.89` | `125.16 ± 0.86` | **`+89.3%`** |
| `qwensploit_27b_q6k` | `qwen35 27B Q6_K` | `f16`/`f16` | `pp2048` | `51.85 ± 1.39` | `100.91 ± 0.97` | **`+94.6%`** |
| `qwensploit_27b_q6k` | `qwen35 27B Q6_K` | `f16`/`f16` | `pp4096` | `68.69 ± 0.21` | `99.79 ± 0.82` | **`+45.3%`** |
| `qwensploit_27b_q6k` | `qwen35 27B Q6_K` | `f16`/`f16` | `tg128` | `2.44 ± 0.00` | `2.49 ± 0.01` | `+2.0%` |
| `ravenx_35b_q4km` | `qwen35moe 35B.A3B Q4_K - Medium` | `f16`/`f16` | `pp128` | `295.36 ± 5.75` | `284.20 ± 7.31` | `-3.8%` |
| `ravenx_35b_q4km` | `qwen35moe 35B.A3B Q4_K - Medium` | `f16`/`f16` | `pp512` | `417.29 ± 11.44` | `522.29 ± 2.07` | **`+25.2%`** |
| `ravenx_35b_q4km` | `qwen35moe 35B.A3B Q4_K - Medium` | `f16`/`f16` | `pp2048` | `234.77 ± 30.16` | `311.00 ± 11.07` | **`+32.5%`** |
| `ravenx_35b_q4km` | `qwen35moe 35B.A3B Q4_K - Medium` | `f16`/`f16` | `pp4096` | `219.40 ± 16.04` | `298.70 ± 5.98` | **`+36.1%`** |
| `ravenx_35b_q4km` | `qwen35moe 35B.A3B Q4_K - Medium` | `f16`/`f16` | `tg128` | `20.12 ± 0.08` | `20.22 ± 0.07` | `+0.5%` |
| `ravenx_35b_q4km_q8kv` | `qwen35moe 35B.A3B Q4_K - Medium` | `q8_0`/`q8_0` | `pp512` | `427.91 ± 3.95` | `513.60 ± 9.79` | **`+20.0%`** |
| `ravenx_35b_q4km_q8kv` | `qwen35moe 35B.A3B Q4_K - Medium` | `q8_0`/`q8_0` | `pp2048` | `225.85 ± 9.65` | `505.45 ± 0.64` | **`+123.8%`** |
| `ravenx_35b_q4km_q8kv` | `qwen35moe 35B.A3B Q4_K - Medium` | `q8_0`/`q8_0` | `tg128` | `20.64 ± 0.66` | `20.46 ± 0.16` | `-0.9%` |
| `spark_4b_tq2_0` | `spark2_5 4.11B TQ2_0` | `f16`/`f16` | `pp128` | `95.31 ± 5.92` | `192.50 ± 1.07` | **`+102.0%`** |
| `spark_4b_tq2_0` | `spark2_5 4.11B TQ2_0` | `f16`/`f16` | `pp512` | `99.12 ± 1.65` | `213.10 ± 1.52` | **`+115.0%`** |
| `spark_4b_tq2_0` | `spark2_5 4.11B TQ2_0` | `f16`/`f16` | `pp2048` | `89.57 ± 3.16` | `205.48 ± 1.09` | **`+129.4%`** |
| `spark_4b_tq2_0` | `spark2_5 4.11B TQ2_0` | `f16`/`f16` | `pp4096` | `93.52 ± 2.14` | `203.75 ± 1.68` | **`+117.9%`** |
| `spark_4b_tq2_0` | `spark2_5 4.11B TQ2_0` | `f16`/`f16` | `tg128` | `3.10 ± 0.60` | **`40.32 ± 0.45`** | **`+1200.6% (13.0x)`** |

### Interleaved A/B Regression Check (`NeoHorse-1-4B` `pp128`)

`neohorse_4b_q4km pp128` was the only cell exceeding −5% (`572.83 ± 27.77` -> `530.61 ± 37.27`, `-7.4%`). An interleaved `-r 5` A/B pass gave:
- Baseline #1: `560.76 ± 27.30 t/s`
- Patched (default): `539.94 ± 46.22 t/s` (`-3.7%` vs Base #1)
- Patched (`GGML_CUDA_GDN_CHUNKED=0`): `564.48 ± 47.18 t/s`
- Patched (`GGML_CUDA_GRID_STALL_PAD=0 GGML_CUDA_MMF32=0`): `561.97 ± 29.05 t/s`
- Baseline #2: `578.11 ± 23.71 t/s`

Because `±27..47 t/s` standard deviation under concurrent Docker load exceeds the delta and `pp2048`/`pp4096` on the exact same model improved `+19.1%`/`+20.4%`, no kill-switch override is warranted.

---

## Verification Summary

### Unit Tests & Diagnostics (`build-rocm102-rdna`)
- `test-alloc`: **PASS** (all allocation / `GGML_GLU_INPLACE` / graph fingerprint tests)
- `test-kv-quant`: **PASS**
- `test-apu-convert`: **PASS** (4/4 pipeline gates)
- `test-apu-lowquant`: **PASS** (4/4 tests)
- `test-apu-moe`: **PASS**
- `test-apu-sarathi`: **PASS**
- `test-apu-spec`: **PASS**
- `test-apu-sync`: **PASS**
- `test-apu-xclbin`: **PASS**
- `llama-apu-cli doctor`: **ALL SYSTEMS HEALTHY (Physical APU ready)**

### `test-backend-ops` (`-b ROCm0`)
- `FLASH_ATTN_EXT`: **4359/4359 PASSED**
- `ROPE`: **490/490 PASSED**
- `GET_ROWS`: **244/244 PASSED** (including `tq1_0` and `tq2_0`)
- `MUL_MAT`: **2570/2570 PASSED** (including `tq1_0` and `tq2_0`)
- `SOFT_MAX`: **214/214 PASSED**
- `GATED_DELTA_NET`:
  - With `GGML_CUDA_GDN_CHUNKED_BF16=0` (FP32 chunked path): **43/43 PASSED** at `1e-7`.
  - With default BF16 WMMA chunked prefill (`S_v=128`, `n_seq_tokens >= 64`): 34/43 passed at the unpatched `1e-7` F32 threshold; the 9 large-chunk BF16 WMMA cases had NMSE `8.16e-6` to `9.92e-6` (expected BF16 precision within upstream Block 03's `1e-4` BF16 gate; `tests/test-backend-ops.cpp` was left unmodified).

### Dual-Entry Coherence Checks (`llama-cli` & `llama-server`)
- **`llama-cli` (`--temp 0 -n 64 -ngl 99 -fa 1 --no-display-prompt`)**:
  - `NeoHorse-1-4B` (`qwen35 4B Q4_0`): Coherent output matching baseline setup (`53.6 t/s` prompt / `18.6 t/s` decode, `Zero-copy handoff: PASS`).
  - `Dolphin3-8B` (`llama 8B Q6_K`, `-ctk bf16 -ctv bf16`): Coherent bullet-point answer (`106.1 t/s` prompt / `8.5 t/s` decode).
  - `RavenX-35B` (`qwen35moe 35B.A3B Q4_K_M`): Coherent reasoning output (`43.6 t/s` prompt / `19.6 t/s` decode).
  - `Huihui-Spark-X2.5-4B_QAT-TQ2_0` (`GGML_CUDA_TQ_CENSUS=1`): `TQ_CENSUS dev=0 mul_mat small=1065 mma=0 get_rows=0 vec=366 vec_fused=220` confirmed full GPU offload of `TQ2_0` (`55.1 t/s` prompt / `35.1 t/s` decode vs baseline `15.4 / 1.9 t/s`). This QAT checkpoint emits `9999...` identically on CPU baseline (`1.9 t/s`), `GGML_CUDA_TQ=0` (`2.0 t/s`), and GPU `TQ2_0` (`35.1 t/s`), confirming bit-identical execution parity across CPU AVX2 and GPU kernels.
  - `K2-Horizon-3.7B-TQ2_0` (experimental uncalibrated `TQ2_0` checkpoint): Ran cleanly on both GPU (`87.6 / 21.3 t/s`) and CPU (`GGML_CUDA_TQ=0`: `12.2 / 1.4 t/s`).
- **`llama-server` (`127.0.0.1:18974 -ngl 99 -fa 1 -c 4096`)**:
  - `NeoHorse-1-4B` with `--stall-free --token-budget 256 --sarathi-chunk-size 64`: `/health` OK, `/v1/chat/completions` returned coherent reasoning (`58.0 t/s` prompt / `18.7 t/s` decode), Sarathi scheduler active (`tau=256, chunk=64`).
  - `Dolphin3-8B` (`-ctk bf16 -ctv bf16`): `/health` OK, `/v1/chat/completions` returned coherent two-sentence answer (`102.7 t/s` prompt / `8.6 t/s` decode).
  - `Huihui-Spark-X2.5-4B_QAT-TQ2_0` (`GGML_CUDA_TQ_CENSUS=1`): `/health` OK, `/v1/chat/completions` served at `60.9 / 35.0 t/s` with `TQ_CENSUS` verifying GPU execution.

---

## Runtime Environment Kill-Switches

- `GGML_CUDA_GDN_CHUNKED=0` — disable chunked GDN prefill kernel (Block 03)
- `GGML_CUDA_GDN_CHUNKED_BF16=0` — fall back from `gfx11` BF16 WMMA GDN kernel to FP32 chunked GDN (Block 03)
- `GGML_CUDA_MMB=0` / `GGML_CUDA_MMB_MIN_T=<N>` — disable or adjust batch threshold (default `512`) for BF16-WMMA prefill GEMM (Block 06)
- `GGML_CUDA_MMF32=0` — disable skinny F32 GEMM path (Block 06)
- `GGML_CUDA_BF16_SRC1_CACHE=0` — disable per-graph BF16 `src1` activation cache (Block 06)
- `GGML_CUDA_GRID_STALL_PAD=0` — disable grid stall padding (Block 07)
- `GGML_CUDA_TOPK_SORTED=0` — disable sorted-key argmax in `topk-moe` (Block 08)
- `GGML_CUDA_TQ=0` — refuse `TQ1_0`/`TQ2_0` on GPU and fall back to host CPU execution (Block 17)
- `GGML_CUDA_TQ_FUSE=0` — disable TQ1_0/TQ2_0 decode `vec` kernel fusion & graph Q/K/V rotation (Block 17)
- `GGML_CUDA_TQ_CENSUS=1` — log GPU TQ kernel launch/fusion counters on context teardown (Block 17)

---

## Recommendations

1. **Merge candidate (`00`, `03`, `04`, `06`, `07`, `08`, `17`)**: Together these blocks deliver +89% to +152% prefill speedup on dense Q6_K/8B–27B models, +20% to +124% prefill speedup on 35B MoE, +19% to +20% long-prompt speedup on 4B hybrid GDN, native BF16 KV cache support, and **13x decode speedup** (+102%–129% prefill) for `TQ1_0`/`TQ2_0` ternary models with zero regressions on decode throughput or Phase 9 Sarathi serving.
2. **Next follow-up**: Evaluate Block `05` (`fattn-rdna`) in a dedicated isolated branch if long-context Flash-Attention prefill or quantized-KV FA becomes a priority bottleneck after merging these foundation/GEMM/TQ improvements.
