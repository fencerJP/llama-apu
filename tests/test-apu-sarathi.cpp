// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 9 Sarathi-Serve scheduler unit tests
#include "llama-apu-scheduler.h"

#include <cassert>
#include <cmath>
#include <cstdio>

static llama_apu_slot_t mk_slot(uint32_t id, llama_apu_slot_state_t st, uint32_t total, uint32_t done) {
    llama_apu_slot_t s{};
    s.slot_id = id;
    s.request_id = id;
    s.state = st;
    s.prompt_tokens_total = total;
    s.prompt_tokens_processed = done;
    return s;
}

static void test_align() {
    printf("[align] running...\n");
    assert(llama_apu_align_chunk_size(70, 64) == 128);
    assert(llama_apu_align_chunk_size(128, 64) == 128);
    assert(llama_apu_align_chunk_size(129, 128) == 256);
    assert(llama_apu_align_chunk_size(256, 64) == 256);
    assert(llama_apu_padded_chunk_width(70, 64) == 128);
    printf("  [PASS]\n");
}

static void test_tier_order() {
    printf("[tier_order] running...\n");
    llama_apu_scheduler_config_t cfg{};
    cfg.token_budget = 512;
    cfg.chunk_size = 256;
    cfg.chunk_alignment = 64;
    cfg.min_prefill_reserve = 64;
    cfg.tbt_slo_ms = 100.0f;
    cfg.enable_stall_free = true;
    auto * ctx = llama_apu_sched_ctx_init(&cfg);
    assert(ctx);

    // 2 decodes + 1 swapped + 1 in-flight prefill (300 remaining) + 1 new (500)
    auto d0 = mk_slot(0, LLAMA_APU_SLOT_STATE_GENERATING, 10, 10);
    auto d1 = mk_slot(1, LLAMA_APU_SLOT_STATE_GENERATING, 10, 10);
    auto sw = mk_slot(2, LLAMA_APU_SLOT_STATE_SWAPPED, 100, 100);
    auto pf = mk_slot(3, LLAMA_APU_SLOT_STATE_PREFILL_CHUNK, 500, 200);
    auto nw = mk_slot(4, LLAMA_APU_SLOT_STATE_IDLE, 500, 0);
    assert(llama_apu_sched_upsert_slot(ctx, &d0) == LLAMA_APU_SUCCESS);
    assert(llama_apu_sched_upsert_slot(ctx, &d1) == LLAMA_APU_SUCCESS);
    assert(llama_apu_sched_upsert_slot(ctx, &sw) == LLAMA_APU_SUCCESS);
    assert(llama_apu_sched_upsert_slot(ctx, &pf) == LLAMA_APU_SUCCESS);
    assert(llama_apu_sched_upsert_slot(ctx, &nw) == LLAMA_APU_SUCCESS);

    llama_apu_batch_t b{};
    assert(llama_apu_schedule_next_batch(ctx, &b) == LLAMA_APU_SUCCESS);
    // Tier1: 2 decodes; Tier1.5: swapped takes 1; Tier2: prefill 256 of 300; Tier3: remainder fits 253
    assert(b.num_decode_slots == 3);
    assert(b.decode_slot_ids[0] == 0 && b.decode_slot_ids[1] == 1 && b.decode_slot_ids[2] == 2);
    assert(b.num_prefill_chunks == 2);
    assert(b.prefill_slot_ids[0] == 3 && b.prefill_chunk_sizes[0] == 256);
    assert(b.prefill_slot_ids[1] == 4 && b.prefill_chunk_sizes[1] == (512 - 3 - 256));
    assert(b.total_batch_tokens <= 512);
    llama_apu_sched_ctx_free(ctx);
    printf("  [PASS]\n");
}

static void test_antisarvation() {
    printf("[antistarvation] running...\n");
    llama_apu_scheduler_config_t cfg{};
    cfg.token_budget = 128; cfg.chunk_size = 256; cfg.chunk_alignment = 64;
    cfg.min_prefill_reserve = 64; cfg.enable_stall_free = true;
    auto * ctx = llama_apu_sched_ctx_init(&cfg);
    for (uint32_t i = 0; i < 100; i++) {
        auto d = mk_slot(i, LLAMA_APU_SLOT_STATE_GENERATING, 10, 10);
        llama_apu_sched_upsert_slot(ctx, &d);
    }
    auto nw = mk_slot(100, LLAMA_APU_SLOT_STATE_IDLE, 200, 0);
    llama_apu_sched_upsert_slot(ctx, &nw);
    llama_apu_batch_t b{};
    llama_apu_schedule_next_batch(ctx, &b);
    // N_dec_max = 128-64 = 64, so 64 budget left for prefill
    assert(b.num_decode_slots == 64);
    assert(b.num_prefill_chunks == 1);
    assert(b.total_batch_tokens == 128);
    llama_apu_sched_ctx_free(ctx);
    printf("  [PASS]\n");
}

static void test_lifp() {
    printf("[lifp] running...\n");
    llama_apu_scheduler_config_t cfg{};
    cfg.token_budget = 512; cfg.chunk_size = 256; cfg.chunk_alignment = 64;
    cfg.min_prefill_reserve = 64; cfg.enable_stall_free = true;
    auto * ctx = llama_apu_sched_ctx_init(&cfg);
    auto d0 = mk_slot(0, LLAMA_APU_SLOT_STATE_GENERATING, 10, 10);
    auto d1 = mk_slot(1, LLAMA_APU_SLOT_STATE_GENERATING, 10, 10);
    llama_apu_sched_upsert_slot(ctx, &d0);
    llama_apu_sched_upsert_slot(ctx, &d1);
    uint32_t victim = 999;
    assert(llama_apu_sched_preempt_lifp(ctx, true, &victim) == LLAMA_APU_SUCCESS);
    assert(victim == 1); // newest
    llama_apu_batch_t b{};
    llama_apu_schedule_next_batch(ctx, &b);
    // d0 generating + victim swapped recovery
    assert(b.num_decode_slots == 2);
    llama_apu_scheduler_stats_t st{};
    llama_apu_sched_get_stats(ctx, &st);
    assert(st.preemptions == 1 && st.swapped_slots == 1);
    llama_apu_sched_ctx_free(ctx);
    printf("  [PASS]\n");
}

static void test_slo_ema() {
    printf("[slo_ema] running...\n");
    assert(llama_apu_suggest_budget_for_tbt(40.0f, 8) == 256);
    assert(llama_apu_suggest_budget_for_tbt(100.0f, 8) == 512);
    assert(llama_apu_suggest_budget_for_tbt(100.0f, 64) == 256); // MoE fan-out scales down
    llama_apu_scheduler_config_t cfg{};
    cfg.token_budget = 512; cfg.chunk_size = 256; cfg.chunk_alignment = 64;
    cfg.min_prefill_reserve = 64; cfg.enable_stall_free = true;
    auto * ctx = llama_apu_sched_ctx_init(&cfg);
    auto d0 = mk_slot(0, LLAMA_APU_SLOT_STATE_GENERATING, 10, 10);
    llama_apu_sched_upsert_slot(ctx, &d0);
    llama_apu_batch_t b{};
    llama_apu_schedule_next_batch(ctx, &b);
    llama_apu_sched_record_iteration(ctx, &b, 20.0, 50.0);
    llama_apu_sched_record_iteration(ctx, &b, 30.0, 60.0);
    llama_apu_scheduler_stats_t st{};
    llama_apu_sched_get_stats(ctx, &st);
    assert(st.iterations == 2);
    assert(st.tbt_p50_ms > 0 && st.stall_max_ms >= 0);
    // deadband: small change ignored, large change applied
    llama_apu_sched_request_budget(ctx, 520); // <10% -> ignored
    llama_apu_sched_record_iteration(ctx, &b, 20.0, 0);
    llama_apu_sched_get_stats(ctx, &st);
    assert(st.effective_budget == 512);
    llama_apu_sched_request_budget(ctx, 1024); // >10% -> applied
    llama_apu_sched_record_iteration(ctx, &b, 20.0, 0);
    llama_apu_sched_get_stats(ctx, &st);
    assert(st.effective_budget == 1024);
    llama_apu_sched_ctx_free(ctx);
    printf("  [PASS]\n");
}

int main() {
    printf("====================================================\n");
    printf("  llama-apu: Phase 9 Sarathi-Serve Scheduler Tests\n");
    printf("====================================================\n");
    test_align();
    test_tier_order();
    test_antisarvation();
    test_lifp();
    test_slo_ema();
    printf("\nAll Phase 9 scheduler unit tests passed!\n");
    return 0;
}
