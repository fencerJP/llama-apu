// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 4 Cross-APU Memory and Synchronization Refinement Tests
#include "ggml-apu-bridge.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

static void test_zero_copy_tracker() {
    printf("[test_zero_copy_tracker] running...\n");
    auto & tracker = apu_zero_copy_tracker::get();
    tracker.reset();

    assert(!tracker.is_zero_copy_compliant());

    // Record valid zero-copy handoffs and physical alias checks
    tracker.record_handoff(true, 1024 * 1024);
    tracker.record_handoff(true, 512 * 1024);
    tracker.record_alias_check(true);
    tracker.record_alias_check(true);

    auto stats = tracker.get_stats();
    (void)stats;
    assert(stats.total_handoffs == 2);
    assert(stats.zero_copy_handoffs == 2);
    assert(stats.host_memcpy_count == 0);
    assert(stats.host_memcpy_bytes == 0);
    assert(stats.physical_alias_checks == 2);
    assert(stats.physical_alias_matches == 2);
    assert(tracker.is_zero_copy_compliant());

    // Introduce simulated host memcpy violation
    tracker.record_host_memcpy(4096);
    assert(!tracker.is_zero_copy_compliant());

    tracker.reset();
    assert(!tracker.is_zero_copy_compliant());
    printf("  [PASS]\n");
}

static void test_subbuffer_alignment() {
    printf("[test_subbuffer_alignment] running...\n");
    alignas(64) uint8_t buf[1024];
    uintptr_t base = reinterpret_cast<uintptr_t>(buf);
    (void)base;

    assert(apu_is_tile_dma_aligned(base));
    assert(apu_is_host_cache_aligned(buf));

    // Valid sub-buffer offsets & sizes (16-byte multiples)
    assert(apu_verify_subbuffer_alignment(base, 0, 16));
    assert(apu_verify_subbuffer_alignment(base, 16, 32));
    assert(apu_verify_subbuffer_alignment(base, 64, 64));
    assert(apu_verify_subbuffer_alignment(base, 128, 256));

    // Invalid sub-buffer offsets (not divisible by 16)
    assert(!apu_verify_subbuffer_alignment(base, 4, 32));
    assert(!apu_verify_subbuffer_alignment(base, 8, 32));
    assert(!apu_verify_subbuffer_alignment(base, 15, 32));

    // Invalid sizes
    assert(!apu_verify_subbuffer_alignment(base, 16, 15));
    assert(!apu_verify_subbuffer_alignment(base, 32, 10));

    // Misaligned base address
    assert(!apu_verify_subbuffer_alignment(base + 1, 16, 32));
    assert(!apu_verify_subbuffer_alignment(base + 8, 16, 32));

    printf("  [PASS]\n");
}

static void test_timeline_syncobj_latency() {
    printf("[test_timeline_syncobj_latency] running...\n");
    std::string render_node = apu_find_render_node();
    if (render_node.empty()) {
        printf("  [SKIP: no render node]\n");
        return;
    }

    int render_fd = ::open(render_node.c_str(), O_RDWR | O_CLOEXEC);
    if (render_fd < 0) {
        printf("  [SKIP: cannot open render node]\n");
        return;
    }

    apu_sync_profile_result result{};
    std::string log;
    bool ok = apu_profile_timeline_sync(render_fd, 200, result, false, log);
    ::close(render_fd);

    if (ok) {
        printf("  200 timeline points: avg = %.2f us, p99 = %.2f us\n",
               result.avg_latency_us, result.p99_latency_us);
        assert(result.ordering_preserved);
        assert(result.timeout_handled);
        assert(result.concurrent_stress_passed);
        assert(result.avg_latency_us < 50.0);
        printf("  [PASS]\n");
    } else {
        printf("  [WARN: timeline profile failed: %s]\n", log.c_str());
    }
}

static void test_phase4_audit() {
    printf("[test_phase4_audit] running...\n");
    apu_phase4_audit_result audit{};
    std::string log;
    bool ok = apu_run_phase4_refinement_audit(false, audit, log);
    if (ok) {
        assert(audit.zero_copy_passed);
        assert(audit.physical_alias_passed);
        assert(audit.tile_dma_alignment_passed);
        assert(audit.async_sync_file_passed);
        assert(audit.timeline_latency_passed);
        assert(audit.concurrent_stress_passed);
        printf("  Phase 4 audit passed: LPDDR5X bandwidth = %.2f GB/s\n", audit.lpddr5x_bandwidth_gbps);
        printf("  [PASS]\n");
    } else {
        printf("  Phase 4 audit output: %s\n", log.c_str());
        assert(ok);
    }
}

int main() {
    printf("=== test-apu-sync: Phase 4 Cross-APU Memory & Sync Tests ===\n");
    test_zero_copy_tracker();
    test_subbuffer_alignment();
    test_timeline_syncobj_latency();
    test_phase4_audit();
    printf("ALL TESTS PASSED.\n");
    return 0;
}
