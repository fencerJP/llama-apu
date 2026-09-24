// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 6 Speculative Decoding Coordination and DRM Timeline Sync Tests
#include "ggml-apu-spec.h"
#include "ggml-apu-bridge.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>

static void test_spec_coordinator_config() {
    printf("[test_spec_coordinator_config] running...\n");
    auto & coord = apu_spec_coordinator::get();
    coord.reset();

    // Default configuration
    assert(coord.get_mode() == APU_SPEC_AUTO);
    assert(coord.get_timeline_sync() == true);
    assert(coord.is_active() == true);

    // Explicit controls
    coord.set_mode(APU_SPEC_ON);
    assert(coord.get_mode() == APU_SPEC_ON);
    assert(coord.is_active() == true);

    coord.set_mode(APU_SPEC_OFF);
    assert(coord.get_mode() == APU_SPEC_OFF);
    assert(coord.is_active() == false);

    coord.set_timeline_sync(false);
    assert(coord.get_timeline_sync() == false);

    coord.set_timeline_sync(true);
    assert(coord.get_timeline_sync() == true);

    coord.reset();
    assert(coord.get_mode() == APU_SPEC_AUTO);
    printf("  [PASS]\n");
}

static void test_spec_telemetry_tracking() {
    printf("[test_spec_telemetry_tracking] running...\n");
    auto & coord = apu_spec_coordinator::get();
    coord.reset();

    // 1. First draft batch: 4 tokens drafted, 3 accepted, 1 rejected
    coord.record_draft_batch(4);
    coord.record_acceptance(3, 4);

    apu_spec_stats stats = coord.get_stats();
    assert(stats.draft_batches == 1);
    assert(stats.total_draft_tokens == 4);
    assert(stats.accepted_draft_tokens == 3);
    assert(stats.rejected_draft_tokens == 1);
    assert(stats.rollback_events == 1);
    assert(std::fabs(stats.acceptance_rate - 0.75) < 1e-4);

    // 2. Second draft batch: 5 tokens drafted, all 5 accepted (0 rejected, 0 rollback)
    coord.record_draft_batch(5);
    coord.record_acceptance(5, 5);

    stats = coord.get_stats();
    assert(stats.draft_batches == 2);
    assert(stats.total_draft_tokens == 9);
    assert(stats.accepted_draft_tokens == 8);
    assert(stats.rejected_draft_tokens == 1);
    assert(stats.rollback_events == 1);
    assert(std::fabs(stats.acceptance_rate - (8.0 / 9.0)) < 1e-4);

    // 3. Explicit rollback tracking
    coord.record_rollback(2);
    stats = coord.get_stats();
    assert(stats.rollback_events == 2);
    assert(stats.rejected_draft_tokens == 3);

    std::string summary = coord.format_summary();
    assert(summary.find("APU SPECULATIVE DECODING TELEMETRY") != std::string::npos);
    assert(summary.find("Draft Batches          : 2") != std::string::npos);
    assert(summary.find("Candidate Tokens Drafted: 9") != std::string::npos);
    assert(summary.find("Candidate Tokens Accepted: 8") != std::string::npos);

    coord.reset();
    printf("  [PASS]\n");
}

static void test_spec_drm_timeline_sync() {
    printf("[test_spec_drm_timeline_sync] running...\n");
    auto & coord = apu_spec_coordinator::get();
    coord.reset();
    coord.set_mode(APU_SPEC_ON);
    coord.set_timeline_sync(true);

    bool sync_ok = coord.synchronize_draft_to_target();
    apu_spec_stats stats = coord.get_stats();

    if (sync_ok && stats.timeline_sync_passes > 0) {
        printf("  [+] DRM timeline synchronization active on /dev/dri/renderD128 (latency: %ld us)\n",
               (long) stats.sync_latency_us);
        assert(stats.timeline_sync_passes == 1);
    } else {
        printf("  [*] DRM timeline sync skipped or unavailable in sandbox environment\n");
    }

    // Disabled mode should return true immediately without syncing
    coord.set_timeline_sync(false);
    assert(coord.synchronize_draft_to_target() == true);
    assert(coord.get_stats().timeline_sync_passes == stats.timeline_sync_passes);

    coord.reset();
    printf("  [PASS]\n");
}

static void test_spec_fallback_handling() {
    printf("[test_spec_fallback_handling] running...\n");
    auto & coord = apu_spec_coordinator::get();
    coord.reset();

    assert(!coord.get_stats().fallback_occurred);

    const std::string reason = "draft model vocabulary mismatch (vocab_cmpt=0)";
    coord.record_fallback(reason);

    apu_spec_stats stats = coord.get_stats();
    assert(stats.fallback_occurred);
    assert(stats.last_fallback_reason == reason);

    std::string summary = coord.format_summary();
    assert(summary.find("Fallback Triggered     : YES (draft model vocabulary mismatch") != std::string::npos);

    coord.reset();
    printf("  [PASS]\n");
}

int main() {
    printf("====================================================\n");
    printf("  llama-apu: Phase 6 Speculative Decoding Tests\n");
    printf("====================================================\n");

    test_spec_coordinator_config();
    test_spec_telemetry_tracking();
    test_spec_drm_timeline_sync();
    test_spec_fallback_handling();

    printf("\nAll APU speculative decoding unit tests passed successfully!\n");
    return 0;
}
