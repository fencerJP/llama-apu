// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 9 Sarathi-Serve Chunked Prefill & Stall-Free Scheduling
// C API per docs/phase-9-sarathi-serve.md §9.5. Iteration-level 4-tier
// scheduler: Tier 1 decodes, Tier 1.5 preempted recovery, Tier 2 in-flight
// prefill chunks, Tier 3 new admissions. Tile alignment 64/128, LIFP
// preemption, EMA budget smoothing.
#pragma once

#include <stdint.h>
#include <stdbool.h>

#define LLAMA_APU_MAX_BATCH_SLOTS 64

// Explicit Slot Lifecycle States
typedef enum {
    LLAMA_APU_SLOT_STATE_IDLE          = 0,
    LLAMA_APU_SLOT_STATE_PREFILL_CHUNK = 1,
    LLAMA_APU_SLOT_STATE_GENERATING    = 2,
    LLAMA_APU_SLOT_STATE_SWAPPED       = 3,
    LLAMA_APU_SLOT_STATE_RECOMPUTE     = 4,
    LLAMA_APU_SLOT_STATE_FINISHED      = 5,
    LLAMA_APU_SLOT_STATE_FAILED        = 6
} llama_apu_slot_state_t;

// Status Codes
#define LLAMA_APU_SUCCESS              0
#define LLAMA_APU_ERR_OOM             -1
#define LLAMA_APU_ERR_TIMEOUT         -2
#define LLAMA_APU_ERR_SYNC_FAIL       -3
#define LLAMA_APU_ERR_INVALID_SLOT    -4
#define LLAMA_APU_ERR_PREEMPTED       -5

// Dual Physical Page Addresses for Decoupled KV (INT4 Weights + FP16 Scales)
typedef struct {
    uint64_t page_w_phys_addr; // 16-byte aligned INT4 weight physical page
    uint64_t page_s_phys_addr; // 16-byte aligned FP16 scale physical page
} llama_apu_npu_block_t;

// Slot Descriptor
typedef struct {
    uint32_t slot_id;
    uint32_t request_id;
    llama_apu_slot_state_t state;
    uint32_t prompt_tokens_total;
    uint32_t prompt_tokens_processed;
    uint32_t generated_tokens_count;
    uint32_t * block_table_gpu;              // GPU virtual block lookup array
    llama_apu_npu_block_t * npu_block_table; // 16-byte aligned NPU physical page pair array
    uint32_t num_blocks;
} llama_apu_slot_t;

// Batch Execution Descriptor
typedef struct {
    uint32_t num_decode_slots;
    uint32_t decode_slot_ids[LLAMA_APU_MAX_BATCH_SLOTS];
    uint32_t num_prefill_chunks;
    uint32_t prefill_slot_ids[LLAMA_APU_MAX_BATCH_SLOTS];
    uint32_t prefill_chunk_sizes[LLAMA_APU_MAX_BATCH_SLOTS]; // in tokens (64/128 tile-aligned)
    uint32_t total_batch_tokens;                             // <= token_budget tau
} llama_apu_batch_t;

// Scheduler Configuration
typedef struct {
    uint32_t token_budget;         // Max tokens per iteration (tau), default 512
    uint32_t chunk_size;           // Target prefill chunk size, default 256 (auto-aligned 64/128)
    uint32_t chunk_alignment;      // Tile alignment (64 or 128), default 64
    uint32_t min_prefill_reserve;  // Reserved prefill tokens against starvation, default 64
    float    tbt_slo_ms;           // Target P99 TBT latency target, default 100.0
    bool     enable_stall_free;    // Enable hybrid batch coalescing
} llama_apu_scheduler_config_t;

// Scheduler telemetry (EMA-smoothed budget + TBT/TFT tracking)
typedef struct {
    uint64_t iterations;
    uint64_t decode_tokens;
    uint64_t prefill_tokens;
    uint64_t preemptions;
    uint64_t swapped_slots;
    uint64_t recompute_slots;
    double   ema_batch_tokens;   // EMA(α=0.1) of total_batch_tokens
    double   tbt_p50_ms;
    double   tbt_p90_ms;
    double   tbt_p99_ms;
    double   ttft_p50_ms;
    double   stall_max_ms;       // max(0, TBT_actual - 2*TBT_isolated_decode)
    uint32_t effective_budget;  // tau after MoE fan-out / SLO tuning
} llama_apu_scheduler_stats_t;

// Opaque scheduler context (defined in .cpp)
typedef struct llama_apu_sched_ctx_t llama_apu_sched_ctx_t;

// Chunk-size helper: clamp + align up to 64/128 boundary
uint32_t llama_apu_align_chunk_size(uint32_t requested, uint32_t alignment);

// Padded execution width for a tail chunk (n_past advances by real tokens only)
uint32_t llama_apu_padded_chunk_width(uint32_t real_tokens, uint32_t alignment);

#ifdef __cplusplus
extern "C" {
#endif

llama_apu_sched_ctx_t * llama_apu_sched_ctx_init(const llama_apu_scheduler_config_t * config);
void llama_apu_sched_ctx_free(llama_apu_sched_ctx_t * ctx);
int  llama_apu_sched_configure(llama_apu_sched_ctx_t * ctx, const llama_apu_scheduler_config_t * config);

// Register/update a slot's scheduling state. Returns LLAMA_APU_SUCCESS or ERR_INVALID_SLOT.
int llama_apu_sched_upsert_slot(llama_apu_sched_ctx_t * ctx, const llama_apu_slot_t * slot);
int llama_apu_sched_remove_slot(llama_apu_sched_ctx_t * ctx, uint32_t slot_id);

// LIFP preemption: evict newest generating slot to SWAPPED (swap=true) or RECOMPUTE.
int llama_apu_sched_preempt_lifp(llama_apu_sched_ctx_t * ctx, bool swap, uint32_t * out_slot_id);

// Constructs next hybrid batch adhering to token budget tau.
int llama_apu_schedule_next_batch(llama_apu_sched_ctx_t * ctx, llama_apu_batch_t * batch_out);

// Dispatches hybrid execution across GPU and NPU with DRM syncobj timeline fences.
// On hosts without /dev/dri + /dev/accel this validates the batch and returns
// SUCCESS as a graceful CPU-fallback (no hardware dispatch attempted).
struct llama_apu_gpu_ctx_t;
struct llama_apu_npu_ctx_t;
int llama_apu_submit_hybrid_batch(
    struct llama_apu_gpu_ctx_t * gpu_ctx,
    struct llama_apu_npu_ctx_t * npu_ctx,
    const llama_apu_batch_t * batch,
    uint64_t signal_point,
    uint64_t wait_point);

// SLO helpers: suggest tau for a TBT target; EMA update with 10% deadband.
uint32_t llama_apu_suggest_budget_for_tbt(float tbt_slo_ms, uint32_t active_experts);
// SLO tuner: request a new effective budget; applied on next record_iteration with 10% deadband.
void llama_apu_sched_request_budget(llama_apu_sched_ctx_t * ctx, uint32_t token_budget);
void llama_apu_sched_record_iteration(llama_apu_sched_ctx_t * ctx, const llama_apu_batch_t * batch,
                                      double tbt_ms, double ttft_ms);

int llama_apu_sched_get_stats(const llama_apu_sched_ctx_t * ctx, llama_apu_scheduler_stats_t * out);
void llama_apu_sched_reset_stats(llama_apu_sched_ctx_t * ctx);

#ifdef __cplusplus
}
#endif
