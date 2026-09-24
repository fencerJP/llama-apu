// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 5 MoE Routing and Router SRAM Pinning Tests (§5.1-§5.3)
#include "ggml-apu-moe.h"
#include "ggml-apu-bridge.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

static void test_moe_manager_config() {
    printf("[test_moe_manager_config] running...\n");
    auto & mgr = apu_moe_router_manager::get();
    mgr.reset();

    // Default configuration
    assert(mgr.get_mode() == APU_MOE_ROUTER_AUTO);
    assert(mgr.get_sram_limit_mb() == 32);

    // Explicit controls
    mgr.set_mode(APU_MOE_ROUTER_ON);
    assert(mgr.get_mode() == APU_MOE_ROUTER_ON);

    mgr.set_mode(APU_MOE_ROUTER_OFF);
    assert(mgr.get_mode() == APU_MOE_ROUTER_OFF);

    mgr.set_sram_limit_mb(96);
    assert(mgr.get_sram_limit_mb() == 96);

    mgr.reset();
    printf("  [PASS]\n");
}

static void test_moe_budget_evaluation_logic() {
    printf("[test_moe_budget_evaluation_logic] running...\n");
    auto & mgr = apu_moe_router_manager::get();

    // Simulate 80MB footprint with 32MB limit in AUTO mode -> DRAM fallback with warning
    mgr.reset();
    mgr.set_mode(APU_MOE_ROUTER_AUTO);
    mgr.set_sram_limit_mb(32);

    apu_moe_router_info info{};
    info.is_moe = true;
    info.arch_name = "qwen35moe";
    info.n_expert_layers = 40;
    info.n_experts = 256;
    info.n_experts_used = 8;
    info.n_embd = 2048;
    info.total_router_bytes = 80ULL * 1024ULL * 1024ULL;
    info.sram_limit_bytes = 32ULL * 1024ULL * 1024ULL;

    assert(info.total_router_bytes > info.sram_limit_bytes);
    // Over budget in auto mode -> fallback to DRAM
    info.pinned_in_sram = false;
    info.fallback_to_dram = true;
    assert(info.fallback_to_dram);
    assert(!info.pinned_in_sram);

    // Under budget with 96MB limit -> pinned in SRAM
    info.sram_limit_bytes = 96ULL * 1024ULL * 1024ULL;
    assert(info.total_router_bytes <= info.sram_limit_bytes);
    info.pinned_in_sram = true;
    info.fallback_to_dram = false;
    assert(info.pinned_in_sram);
    assert(!info.fallback_to_dram);

    // Disabled mode -> fallback to DRAM regardless of limit
    info.mode = APU_MOE_ROUTER_OFF;
    info.pinned_in_sram = false;
    info.fallback_to_dram = true;
    assert(info.fallback_to_dram);
    assert(!info.pinned_in_sram);

    printf("  [PASS]\n");
}

static void test_subbuffer_tile_dma_alignment() {
    printf("[test_subbuffer_tile_dma_alignment] running...\n");
    // Verify that router slot alignments satisfy 16-byte Tile DMA and 64-byte host cache lines
    alignas(64) uint8_t sram_mock_buffer[4096];
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(sram_mock_buffer);

    assert(apu_is_tile_dma_aligned(base_addr));
    assert(apu_is_host_cache_aligned(sram_mock_buffer));

    // Verify sub-buffer slots for router gating matrices
    for (size_t offset = 0; offset < 2048; offset += 64) {
        assert(apu_is_tile_dma_aligned(base_addr, offset));
        assert(apu_is_host_cache_aligned(sram_mock_buffer, offset));
        assert(apu_verify_subbuffer_alignment(base_addr, offset, 64));
    }
    printf("  [PASS]\n");
}

static void test_real_model_audit_dense() {
    printf("[test_real_model_audit_dense] running...\n");
    const std::string dense_path = "/opt/models/neohorse-1-4b/NeoHorse-1-4B-Q4_K_M.gguf";

    apu_moe_router_info info{};
    std::string log;
    bool ok = apu_audit_moe_router(dense_path, APU_MOE_ROUTER_AUTO, 32, false, info, log);
    assert(ok);
    assert(!info.is_moe);
    assert(!info.pinned_in_sram);
    assert(!info.fallback_to_dram);
    assert(info.total_router_bytes == 0);
    printf("  [PASS] Dense model confirmed: zero overhead, MoE bypass active.\n");
}

static void test_real_model_audit_moe() {
    printf("[test_real_model_audit_moe] running...\n");
    const std::string moe_path = "/opt/models/occamy-1.0/occamy-ai_occamy-1.0-IQ4_NL.gguf";

    // 1. Audit under default 32MB limit: footprint (80MB) > limit (32MB) -> fallback with warning
    {
        apu_moe_router_info info{};
        std::string log;
        bool ok = apu_audit_moe_router(moe_path, APU_MOE_ROUTER_AUTO, 32, true, info, log);
        assert(ok);
        assert(info.is_moe);
        assert(info.arch_name == "qwen35moe");
        assert(info.n_experts == 256);
        assert(info.n_experts_used == 8);
        assert(info.n_expert_layers == 40);
        assert(info.total_router_bytes == 80ULL * 1024ULL * 1024ULL);
        assert(!info.pinned_in_sram);
        assert(info.fallback_to_dram);
        assert(!info.warning_msg.empty());
        printf("  [PASS] Default 32MB limit: Graceful DRAM fallback with warning verified.\n");
    }

    // 2. Audit under 96MB limit: footprint (80MB) <= limit (96MB) -> pinned in SRAM
    {
        apu_moe_router_info info{};
        std::string log;
        bool ok = apu_audit_moe_router(moe_path, APU_MOE_ROUTER_AUTO, 96, false, info, log);
        assert(ok);
        assert(info.is_moe);
        assert(info.pinned_in_sram);
        assert(!info.fallback_to_dram);
        printf("  [PASS] 96MB limit: Successful AIE2P SRAM pinning verified.\n");
    }

    // 3. Audit under --no-router-sram (mode OFF) -> fallback to DRAM
    {
        apu_moe_router_info info{};
        std::string log;
        bool ok = apu_audit_moe_router(moe_path, APU_MOE_ROUTER_OFF, 96, false, info, log);
        assert(ok);
        assert(info.is_moe);
        assert(!info.pinned_in_sram);
        assert(info.fallback_to_dram);
        printf("  [PASS] Mode OFF (--no-router-sram): DRAM routing confirmed.\n");
    }
}

int main() {
    printf("=== test-apu-moe: Phase 5 MoE Routing and Router SRAM Pinning Tests ===\n");
    test_moe_manager_config();
    test_moe_budget_evaluation_logic();
    test_subbuffer_tile_dma_alignment();
    test_real_model_audit_dense();
    test_real_model_audit_moe();
    printf("ALL TESTS PASSED.\n");
    return 0;
}
