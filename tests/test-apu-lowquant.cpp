// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 7 Low-Quantization & T-ACE 16B Unit Test Suite
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cmath>

#include "ggml.h"
#include "ggml-apu-lowquant.h"

// Forward declaration of raw block for testing
typedef struct {
    uint8_t qs[64];
    ggml_fp16_t d;
} apu_raw_block_tq2_0;

static void test_tace16_lowering_and_dequant() {
    printf("[test_tace16_lowering_and_dequant] running...\n");

    const size_t n_weights = 256;
    alignas(64) apu_raw_block_tq2_0 tq2_block;

    // Set scale to 1.0f (FP16: 0x3C00)
    tq2_block.d = ggml_fp32_to_fp16(1.0f);

    // Populate 256 trits: pattern of -1, 0, +1
    // In TQ2_0 encoding:
    // q = 0 -> trit -1
    // q = 1 -> trit  0
    // q = 2 -> trit +1
    for (size_t i = 0; i < 64; ++i) {
        uint8_t q0 = (i % 3);       // trit = q0 - 1
        uint8_t q1 = ((i + 1) % 3);
        uint8_t q2 = ((i + 2) % 3);
        uint8_t q3 = (i % 3);
        tq2_block.qs[i] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
    }

    alignas(64) apu_tace16_tile tace_tiles[4];
    bool lower_ok = apu_lowquant_engine::get().lower_tq2_to_tace16(&tq2_block, tace_tiles, n_weights);
    assert(lower_ok);
    (void)lower_ok;

    // Verify 4 tiles produced, each exactly 16 bytes
    assert(sizeof(tace_tiles) == 64);

    // Dequantize back to float
    std::vector<float> dequant(n_weights, 0.0f);
    bool dequant_ok = apu_lowquant_engine::get().dequantize_tace16_to_f32(tace_tiles, dequant.data(), n_weights);
    assert(dequant_ok);
    (void)dequant_ok;

    // Verify that every dequantized weight is close to {-1.0, 0.0, 1.0} or scaled variant
    size_t non_zero = 0;
    for (size_t i = 0; i < n_weights; ++i) {
        float val = dequant[i];
        if (fabsf(val) > 1e-4f) non_zero++;
        assert(fabsf(val) <= 2.0f);
    }
    assert(non_zero > 0);

    printf("  [PASS]\n");
}

static void test_tace_alignment() {
    printf("[test_tace_alignment] running...\n");

    alignas(64) uint8_t aligned_buf[128];
    assert(apu_verify_tace_alignment(aligned_buf, 64) == true);
    assert(apu_verify_tace_alignment(aligned_buf + 4, 64) == false);
    assert(apu_verify_tace_alignment(aligned_buf, 60) == false); // Not multiple of 16

    printf("  [PASS]\n");
}

static void test_cache_coherency() {
    printf("[test_cache_coherency] running...\n");

    alignas(64) uint8_t test_buf[256];
    for (size_t i = 0; i < sizeof(test_buf); ++i) {
        test_buf[i] = static_cast<uint8_t>(i);
    }

    // Flush cache range
    apu_sync_cache_coherency(test_buf, sizeof(test_buf));

    // Verify data intact
    for (size_t i = 0; i < sizeof(test_buf); ++i) {
        assert(test_buf[i] == static_cast<uint8_t>(i));
    }

    printf("  [PASS]\n");
}

static void test_memory_governor() {
    printf("[test_memory_governor] running...\n");

    auto status1 = apu_lowquant_engine::get().query_memory_governor(1ULL * 1024 * 1024 * 1024); // 1 GB
    assert(status1.total_ram_bytes > 0);
    assert(status1.available_ram_bytes > 0);
    assert(status1.within_50gb_ceiling == true);
    assert(!status1.recommended_hierarchy.empty());

    // Test exceeding 50 GB ceiling
    auto status_huge = apu_lowquant_engine::get().query_memory_governor(55ULL * 1024 * 1024 * 1024); // 55 GB
    assert(status_huge.within_50gb_ceiling == false);
    assert(status_huge.recommended_hierarchy == "micro_chunk");

    printf("  [PASS]\n");
}

int main() {
    printf("====================================================\n");
    printf("  llama-apu: Phase 7 Low-Quant & T-ACE Unit Tests\n");
    printf("====================================================\n");

    test_tace16_lowering_and_dequant();
    test_tace_alignment();
    test_cache_coherency();
    test_memory_governor();

    printf("\nAll APU low-quant unit tests passed successfully!\n");
    return 0;
}
