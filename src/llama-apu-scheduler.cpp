// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 9 Sarathi-Serve scheduler core (4-tier, LIFP, EMA)
#include "llama-apu-scheduler.h"
#include "llama-apu-dispatch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#define LLAMA_APU_SCHED_DEFAULT_TOKEN_BUDGET  512u
#define LLAMA_APU_SCHED_DEFAULT_CHUNK_SIZE    256u
#define LLAMA_APU_SCHED_DEFAULT_ALIGN         64u
#define LLAMA_APU_SCHED_DEFAULT_RESERVE       64u
#define LLAMA_APU_SCHED_DEFAULT_TBT_SLO       100.0f
#define LLAMA_APU_SCHED_EMA_ALPHA             0.1
#define LLAMA_APU_SCHED_DEADBAND_FRAC         0.10
#define LLAMA_APU_SCHED_MAX_SAMPLES           1024u

uint32_t llama_apu_align_chunk_size(uint32_t requested, uint32_t alignment) {
    uint32_t a = (alignment == 128u) ? 128u : 64u;
    if (requested == 0) return 0;
    return ((requested + a - 1u) / a) * a;
}

uint32_t llama_apu_padded_chunk_width(uint32_t real_tokens, uint32_t alignment) {
    return llama_apu_align_chunk_size(real_tokens, alignment);
}

struct llama_apu_sched_entry {
    llama_apu_slot_t slot;
    uint64_t seq = 0;
};

struct llama_apu_sched_ctx_t {
    llama_apu_scheduler_config_t cfg;
    std::unordered_map<uint32_t, llama_apu_sched_entry> slots;
    uint64_t next_seq = 1;
    mutable std::mutex mutex_;
    llama_apu_scheduler_stats_t stats{};
    std::vector<double> tbt_samples;
    std::vector<double> ttft_samples;
    double tbt_isolated_min_ms = 0.0;
    uint32_t pending_budget = 0; // requested via SLO tuner, applied with deadband
    bool pending_budget_valid = false;
};

static uint32_t sched_valid_alignment(uint32_t a) {
    return (a == 128u) ? 128u : 64u;
}

static void sched_apply_config(llama_apu_sched_ctx_t * ctx, const llama_apu_scheduler_config_t * c) {
    if (c->token_budget >= 64u && c->token_budget <= 8192u) {
        ctx->cfg.token_budget = c->token_budget;
    } else if (c->token_budget != 0) {
        ctx->cfg.token_budget = LLAMA_APU_SCHED_DEFAULT_TOKEN_BUDGET;
    }
    ctx->cfg.chunk_alignment = sched_valid_alignment(c->chunk_alignment);
    if (c->chunk_size >= 16u && c->chunk_size <= 4096u) {
        ctx->cfg.chunk_size = c->chunk_size;
    } else if (c->chunk_size != 0) {
        ctx->cfg.chunk_size = LLAMA_APU_SCHED_DEFAULT_CHUNK_SIZE;
    }
    if (c->min_prefill_reserve >= 16u && c->min_prefill_reserve <= 1024u) {
        ctx->cfg.min_prefill_reserve = c->min_prefill_reserve;
    }
    if (c->tbt_slo_ms >= 1.0f && c->tbt_slo_ms <= 10000.0f) {
        ctx->cfg.tbt_slo_ms = c->tbt_slo_ms;
    }
    ctx->cfg.enable_stall_free = c->enable_stall_free;
    if (ctx->stats.effective_budget == 0) {
        ctx->stats.effective_budget = ctx->cfg.token_budget;
    }
}

llama_apu_sched_ctx_t * llama_apu_sched_ctx_init(const llama_apu_scheduler_config_t * config) {
    llama_apu_sched_ctx_t * ctx = new (std::nothrow) llama_apu_sched_ctx_t();
    if (!ctx) return nullptr;
    ctx->cfg.token_budget = LLAMA_APU_SCHED_DEFAULT_TOKEN_BUDGET;
    ctx->cfg.chunk_size = LLAMA_APU_SCHED_DEFAULT_CHUNK_SIZE;
    ctx->cfg.chunk_alignment = LLAMA_APU_SCHED_DEFAULT_ALIGN;
    ctx->cfg.min_prefill_reserve = LLAMA_APU_SCHED_DEFAULT_RESERVE;
    ctx->cfg.tbt_slo_ms = LLAMA_APU_SCHED_DEFAULT_TBT_SLO;
    ctx->cfg.enable_stall_free = true;
    ctx->stats.effective_budget = ctx->cfg.token_budget;
    if (config) {
        sched_apply_config(ctx, config);
        ctx->stats.effective_budget = ctx->cfg.token_budget;
    }
    return ctx;
}

void llama_apu_sched_ctx_free(llama_apu_sched_ctx_t * ctx) {
    delete ctx;
}

int llama_apu_sched_configure(llama_apu_sched_ctx_t * ctx, const llama_apu_scheduler_config_t * config) {
    if (!ctx || !config) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    sched_apply_config(ctx, config);
    return LLAMA_APU_SUCCESS;
}

int llama_apu_sched_upsert_slot(llama_apu_sched_ctx_t * ctx, const llama_apu_slot_t * slot) {
    if (!ctx || !slot) return LLAMA_APU_ERR_INVALID_SLOT;
    if (slot->state > LLAMA_APU_SLOT_STATE_FAILED) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    auto it = ctx->slots.find(slot->slot_id);
    if (it == ctx->slots.end()) {
        if (ctx->slots.size() >= LLAMA_APU_MAX_SCHED_SLOTS) return LLAMA_APU_ERR_OOM;
        llama_apu_sched_entry e;
        e.slot = *slot;
        e.seq = ctx->next_seq++;
        ctx->slots.emplace(slot->slot_id, e);
    } else {
        uint64_t seq = it->second.seq;
        it->second.slot = *slot;
        it->second.seq = seq;
    }
    return LLAMA_APU_SUCCESS;
}

int llama_apu_sched_remove_slot(llama_apu_sched_ctx_t * ctx, uint32_t slot_id) {
    if (!ctx) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    auto it = ctx->slots.find(slot_id);
    if (it == ctx->slots.end()) return LLAMA_APU_ERR_INVALID_SLOT;
    ctx->slots.erase(it);
    return LLAMA_APU_SUCCESS;
}

int llama_apu_sched_preempt_lifp(llama_apu_sched_ctx_t * ctx, bool swap, uint32_t * out_slot_id) {
    if (!ctx) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    llama_apu_sched_entry * newest = nullptr;
    for (auto & kv : ctx->slots) {
        if (kv.second.slot.state == LLAMA_APU_SLOT_STATE_GENERATING) {
            if (!newest || kv.second.seq > newest->seq) newest = &kv.second;
        }
    }
    if (!newest) return LLAMA_APU_ERR_INVALID_SLOT;
    newest->slot.state = swap ? LLAMA_APU_SLOT_STATE_SWAPPED : LLAMA_APU_SLOT_STATE_RECOMPUTE;
    ctx->stats.preemptions++;
    if (swap) ctx->stats.swapped_slots++;
    else ctx->stats.recompute_slots++;
    if (out_slot_id) *out_slot_id = newest->slot.slot_id;
    return LLAMA_APU_SUCCESS;
}

static double sched_percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(q * (v.size() - 1) + 0.5);
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

int llama_apu_schedule_next_batch(llama_apu_sched_ctx_t * ctx, llama_apu_batch_t * batch_out) {
    if (!ctx || !batch_out) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    memset(batch_out, 0, sizeof(*batch_out));

    uint32_t tau = ctx->stats.effective_budget ? ctx->stats.effective_budget : ctx->cfg.token_budget;
    if (tau == 0) tau = LLAMA_APU_SCHED_DEFAULT_TOKEN_BUDGET;
    uint32_t reserve = ctx->cfg.min_prefill_reserve;
    if (reserve >= tau) reserve = 64u;
    uint32_t align = sched_valid_alignment(ctx->cfg.chunk_alignment);
    uint32_t chunk_target = ctx->cfg.chunk_size;
    if (chunk_target == 0 || chunk_target > tau) chunk_target = tau;
    chunk_target = llama_apu_align_chunk_size(chunk_target, align);
    if (chunk_target > tau) chunk_target = llama_apu_align_chunk_size(tau, align);

    // order slots by seq (oldest first) for fairness; LIFP uses reverse
    std::vector<llama_apu_sched_entry *> ordered;
    ordered.reserve(ctx->slots.size());
    for (auto & kv : ctx->slots) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(), [](const llama_apu_sched_entry * a, const llama_apu_sched_entry * b) {
        return a->seq < b->seq;
    });

    uint32_t used = 0;

    // Tier 1: ongoing decodes, capped at N_dec_max = tau - reserve (anti-starvation)
    uint32_t n_dec_max = (tau > reserve) ? (tau - reserve) : 0;
    std::vector<llama_apu_sched_entry *> generating;
    for (auto * e : ordered) {
        if (e->slot.state == LLAMA_APU_SLOT_STATE_GENERATING) generating.push_back(e);
    }
    uint32_t n_dec_take = std::min<uint32_t>((uint32_t)generating.size(), n_dec_max);
    // also cap by MAX slots
    n_dec_take = std::min<uint32_t>(n_dec_take, LLAMA_APU_MAX_BATCH_SLOTS);
    for (uint32_t i = 0; i < n_dec_take; i++) {
        batch_out->decode_slot_ids[batch_out->num_decode_slots++] = generating[i]->slot.slot_id;
        used += 1;
    }

    // Tier 1.5: preempted recovery ahead of new requests
    for (auto * e : ordered) {
        if (used >= tau) break;
        if (e->slot.state == LLAMA_APU_SLOT_STATE_SWAPPED) {
            if (batch_out->num_decode_slots >= LLAMA_APU_MAX_BATCH_SLOTS) break;
            batch_out->decode_slot_ids[batch_out->num_decode_slots++] = e->slot.slot_id;
            used += 1;
        } else if (e->slot.state == LLAMA_APU_SLOT_STATE_RECOMPUTE) {
            if (batch_out->num_prefill_chunks >= LLAMA_APU_MAX_BATCH_SLOTS) break;
            uint32_t total = e->slot.prompt_tokens_total;
            uint32_t done = e->slot.prompt_tokens_processed;
            uint32_t remaining = (total > done) ? (total - done) : 0;
            if (remaining == 0) {
                // nothing to recompute; treat as 1-token resume
                if (used + 1 > tau) break;
                batch_out->decode_slot_ids[batch_out->num_decode_slots++] = e->slot.slot_id;
                used += 1;
            } else {
                uint32_t budget_left = (tau > used) ? (tau - used) : 0;
                if (budget_left == 0) break;
                uint32_t c = std::min<uint32_t>({remaining, chunk_target, budget_left});
                // align down to fit? keep real tokens, padded width computed separately
                // ensure at least 1 real token and do not exceed budget
                if (c == 0) break;
                // if remaining > budget_left, we chunk to budget_left (may be unaligned tail; padding is execution-only)
                batch_out->prefill_slot_ids[batch_out->num_prefill_chunks] = e->slot.slot_id;
                batch_out->prefill_chunk_sizes[batch_out->num_prefill_chunks] = c;
                batch_out->num_prefill_chunks++;
                used += c;
            }
        }
    }

    // Tier 2: in-flight prefill chunks (previously chunked, remaining prompt)
    for (auto * e : ordered) {
        if (used >= tau) break;
        if (e->slot.state != LLAMA_APU_SLOT_STATE_PREFILL_CHUNK) continue;
        // skip if already scheduled as recompute (not possible; states differ)
        uint32_t total = e->slot.prompt_tokens_total;
        uint32_t done = e->slot.prompt_tokens_processed;
        uint32_t remaining = (total > done) ? (total - done) : 0;
        if (remaining == 0) continue;
        if (batch_out->num_prefill_chunks >= LLAMA_APU_MAX_BATCH_SLOTS) break;
        uint32_t budget_left = (tau > used) ? (tau - used) : 0;
        if (budget_left == 0) break;
        uint32_t c = std::min<uint32_t>({remaining, chunk_target, budget_left});
        if (c == 0) break;
        batch_out->prefill_slot_ids[batch_out->num_prefill_chunks] = e->slot.slot_id;
        batch_out->prefill_chunk_sizes[batch_out->num_prefill_chunks] = c;
        batch_out->num_prefill_chunks++;
        used += c;
    }

    // Tier 3: new request admission (IDLE with pending prompt), chunked to fit
    for (auto * e : ordered) {
        if (used >= tau) break;
        if (e->slot.state != LLAMA_APU_SLOT_STATE_IDLE) continue;
        uint32_t total = e->slot.prompt_tokens_total;
        uint32_t done = e->slot.prompt_tokens_processed;
        uint32_t remaining = (total > done) ? (total - done) : 0;
        if (remaining == 0) continue;
        if (batch_out->num_prefill_chunks >= LLAMA_APU_MAX_BATCH_SLOTS) break;
        uint32_t budget_left = (tau > used) ? (tau - used) : 0;
        if (budget_left == 0) break;
        uint32_t c = std::min<uint32_t>({remaining, chunk_target, budget_left});
        if (c == 0) break;
        batch_out->prefill_slot_ids[batch_out->num_prefill_chunks] = e->slot.slot_id;
        batch_out->prefill_chunk_sizes[batch_out->num_prefill_chunks] = c;
        batch_out->num_prefill_chunks++;
        used += c;
    }

    batch_out->total_batch_tokens = used;
    return LLAMA_APU_SUCCESS;
}

int llama_apu_submit_hybrid_batch(
    struct llama_apu_gpu_ctx_t * /*gpu_ctx*/,
    struct llama_apu_npu_ctx_t * /*npu_ctx*/,
    const llama_apu_batch_t * batch,
    uint64_t signal_point,
    uint64_t wait_point) {
    if (!batch) return LLAMA_APU_ERR_INVALID_SLOT;
    if (batch->num_decode_slots > LLAMA_APU_MAX_BATCH_SLOTS) return LLAMA_APU_ERR_INVALID_SLOT;
    if (batch->num_prefill_chunks > LLAMA_APU_MAX_BATCH_SLOTS) return LLAMA_APU_ERR_INVALID_SLOT;
    // §9.2: route through the single-producer dispatch sequencer. On hosts
    // without /dev/dri the shared context degrades to validated CPU fallback.
    if (signal_point == 0) return LLAMA_APU_ERR_INVALID_SLOT;
    return llama_apu_dispatch_submit_shared(batch, signal_point, wait_point);
}

uint32_t llama_apu_suggest_budget_for_tbt(float tbt_slo_ms, uint32_t active_experts) {
    uint32_t tau;
    if (tbt_slo_ms <= 50.0f) tau = 256u;
    else if (tbt_slo_ms <= 100.0f) tau = 512u;
    else if (tbt_slo_ms <= 200.0f) tau = 1024u;
    else tau = 2048u;
    if (active_experts > 32u) {
        uint32_t scaled = (tau * 32u) / active_experts;
        if (scaled < 128u) scaled = 128u;
        tau = scaled;
    }
    if (tau < 64u) tau = 64u;
    if (tau > 2048u) tau = 2048u;
    return tau;
}

void llama_apu_sched_request_budget(llama_apu_sched_ctx_t * ctx, uint32_t token_budget) {
    if (!ctx) return;
    if (token_budget < 64u || token_budget > 8192u) return;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    ctx->pending_budget = token_budget;
    ctx->pending_budget_valid = true;
}

void llama_apu_sched_record_iteration(llama_apu_sched_ctx_t * ctx, const llama_apu_batch_t * batch,
                                      double tbt_ms, double ttft_ms) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    ctx->stats.iterations++;
    if (batch) {
        ctx->stats.decode_tokens += batch->num_decode_slots;
        uint64_t pre = 0;
        for (uint32_t i = 0; i < batch->num_prefill_chunks; i++) pre += batch->prefill_chunk_sizes[i];
        ctx->stats.prefill_tokens += pre;
        double total = (double)batch->total_batch_tokens;
        if (ctx->stats.iterations == 1) ctx->stats.ema_batch_tokens = total;
        else ctx->stats.ema_batch_tokens = (1.0 - LLAMA_APU_SCHED_EMA_ALPHA) * ctx->stats.ema_batch_tokens
                                         + LLAMA_APU_SCHED_EMA_ALPHA * total;
    }
    if (tbt_ms > 0) {
        if (ctx->tbt_isolated_min_ms == 0.0 || tbt_ms < ctx->tbt_isolated_min_ms) {
            ctx->tbt_isolated_min_ms = tbt_ms;
        }
        double stall = tbt_ms - 2.0 * ctx->tbt_isolated_min_ms;
        if (stall < 0) stall = 0;
        if (stall > ctx->stats.stall_max_ms) ctx->stats.stall_max_ms = stall;
        ctx->tbt_samples.push_back(tbt_ms);
        if (ctx->tbt_samples.size() > LLAMA_APU_SCHED_MAX_SAMPLES) {
            ctx->tbt_samples.erase(ctx->tbt_samples.begin(),
                                   ctx->tbt_samples.begin() + (ctx->tbt_samples.size() - LLAMA_APU_SCHED_MAX_SAMPLES));
        }
        ctx->stats.tbt_p50_ms = sched_percentile(ctx->tbt_samples, 0.50);
        ctx->stats.tbt_p90_ms = sched_percentile(ctx->tbt_samples, 0.90);
        ctx->stats.tbt_p99_ms = sched_percentile(ctx->tbt_samples, 0.99);
    }
    if (ttft_ms > 0) {
        ctx->ttft_samples.push_back(ttft_ms);
        if (ctx->ttft_samples.size() > LLAMA_APU_SCHED_MAX_SAMPLES) {
            ctx->ttft_samples.erase(ctx->ttft_samples.begin(),
                                    ctx->ttft_samples.begin() + (ctx->ttft_samples.size() - LLAMA_APU_SCHED_MAX_SAMPLES));
        }
        ctx->stats.ttft_p50_ms = sched_percentile(ctx->ttft_samples, 0.50);
    }
    // Apply pending budget with 10% deadband to avoid oscillation
    if (ctx->pending_budget_valid && ctx->pending_budget != 0) {
        uint32_t cur = ctx->stats.effective_budget ? ctx->stats.effective_budget : ctx->cfg.token_budget;
        uint32_t want = ctx->pending_budget;
        double frac = (want > cur) ? ((double)(want - cur) / (double)cur) : ((double)(cur - want) / (double)cur);
        if (frac >= LLAMA_APU_SCHED_DEADBAND_FRAC) {
            ctx->stats.effective_budget = want;
        }
        ctx->pending_budget_valid = false;
    }
    if (ctx->stats.effective_budget == 0) ctx->stats.effective_budget = ctx->cfg.token_budget;
}

int llama_apu_sched_get_stats(const llama_apu_sched_ctx_t * ctx, llama_apu_scheduler_stats_t * out) {
    if (!ctx || !out) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    *out = ctx->stats;
    return LLAMA_APU_SUCCESS;
}

void llama_apu_sched_reset_stats(llama_apu_sched_ctx_t * ctx) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    uint32_t eff = ctx->cfg.token_budget;
    ctx->stats = llama_apu_scheduler_stats_t{};
    ctx->stats.effective_budget = eff;
    ctx->tbt_samples.clear();
    ctx->ttft_samples.clear();
    ctx->tbt_isolated_min_ms = 0.0;
    ctx->pending_budget_valid = false;
}
