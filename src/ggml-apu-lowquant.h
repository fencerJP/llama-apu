// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 7 Low-Quantization & TQ2_0 Standard Support Substrate (v3.0)
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

#include "ggml.h"
#include "llama.h"

// Alignment requirements for AIE2P (XDNA 2) NPU
#ifndef APU_TILE_DMA_ALIGNMENT_BYTES
#define APU_TILE_DMA_ALIGNMENT_BYTES 16
#endif

#ifndef APU_HOST_CACHE_ALIGNMENT_BYTES
#define APU_HOST_CACHE_ALIGNMENT_BYTES 64
#endif

#ifndef APU_PAGE_SIZE_BYTES
#define APU_PAGE_SIZE_BYTES 4096
#endif

// 50 GB system memory ceiling for out-of-core operations on unified APUs
constexpr uint64_t APU_MAX_SYSTEM_MEMORY_CEILING_BYTES = 50ULL * 1024 * 1024 * 1024;

#ifdef __cplusplus
extern "C" {
#endif

// T-ACE 16-byte co-packed hardware tile for AIE2P vector PE execution (64 weights)
// Layout:
// - bytes [0..11]: 60 ternary trits packed in base-3 (5 trits per byte: 3^5 = 243 < 256)
// - byte  12:      4 ternary trits packed in base-3 (3^4 = 81 < 256)
// - bytes [13..15]: 3-byte scale metadata:
//   - byte 13: 8-bit base scale exponent S (unbiased power-of-two exponent)
//   - bytes 14..15: 16x 1-bit subgroup shift offsets s_n (1 bit per 4 weights, 16 groups = 64 weights)
#pragma pack(push, 1)
typedef struct {
    uint8_t packed_trits_main[12]; // 60 trits
    uint8_t packed_trits_tail;     // 4 trits
    uint8_t base_exponent;         // S
    uint16_t subgroup_shifts;      // 16x 1-bit shift offsets
} apu_tace16_tile;
#pragma pack(pop)

static_assert(sizeof(apu_tace16_tile) == 16, "apu_tace16_tile must be exactly 16 bytes for Tile DMA");

// C ABI Interface
bool apu_verify_tace_alignment(const void * ptr, size_t size);
void apu_sync_cache_coherency(const void * host_ptr, size_t size);

#ifdef __cplusplus
}

// C++ API and Engine
struct apu_memory_governor_status {
    uint64_t total_ram_bytes = 0;
    uint64_t available_ram_bytes = 0;
    uint64_t peak_allocated_bytes = 0;
    bool     within_50gb_ceiling = true;
    std::string recommended_hierarchy; // "layer", "expert", "micro_chunk"
};

class apu_lowquant_engine {
public:
    static apu_lowquant_engine & get();

    // Transform native TQ2_0 blocks (256 weights, 66 bytes) to T-ACE 16B tiles (64 weights, 16 bytes each -> 64 bytes per 256 weights)
    bool lower_tq2_to_tace16(
        const void * tq2_src,
        void * tace_dst,
        size_t n_weights
    );

    // Dequantize T-ACE 16B tiles back to float for verification / testing
    bool dequantize_tace16_to_f32(
        const void * tace_src,
        float * f32_dst,
        size_t n_weights
    );

    // Memory Governor: check current system headroom and enforce 50GB ceiling
    apu_memory_governor_status query_memory_governor(size_t working_set_bytes = 0) const;

    // Cache coherency flush for zero-copy DMA-BUF buffers before NPU submission
    void sync_cache_range(const void * ptr, size_t size);

private:
    apu_lowquant_engine() = default;
};

#endif
