// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 7 Low-Quantization & TQ2_0 Standard Support Substrate (v3.0)
#include "ggml-apu-lowquant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Block definition matching ggml-common.h for TQ2_0
typedef struct {
    uint8_t qs[64];  // 256 2-bit trits
    ggml_fp16_t d;   // FP16 block scale
} apu_raw_block_tq2_0;

static_assert(sizeof(apu_raw_block_tq2_0) == 66, "apu_raw_block_tq2_0 must be 66 bytes");

bool apu_verify_tace_alignment(const void * ptr, size_t size) {
    if (!ptr) return false;
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr % APU_TILE_DMA_ALIGNMENT_BYTES) != 0) return false;
    if ((size % APU_TILE_DMA_ALIGNMENT_BYTES) != 0) return false;
    return true;
}

void apu_sync_cache_coherency(const void * host_ptr, size_t size) {
    apu_lowquant_engine::get().sync_cache_range(host_ptr, size);
}

apu_lowquant_engine & apu_lowquant_engine::get() {
    static apu_lowquant_engine instance;
    return instance;
}

void apu_lowquant_engine::sync_cache_range(const void * ptr, size_t size) {
    if (!ptr || size == 0) return;

#if defined(__x86_64__) || defined(_M_X64)
    const uintptr_t start = reinterpret_cast<uintptr_t>(ptr) & ~(uintptr_t)(APU_HOST_CACHE_ALIGNMENT_BYTES - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(ptr) + size + APU_HOST_CACHE_ALIGNMENT_BYTES - 1) & ~(uintptr_t)(APU_HOST_CACHE_ALIGNMENT_BYTES - 1);

    for (uintptr_t p = start; p < end; p += APU_HOST_CACHE_ALIGNMENT_BYTES) {
        _mm_clflush(reinterpret_cast<const void *>(p));
    }
    _mm_sfence();
#endif
}

// Convert trit {-1, 0, +1} to base-3 digit: 0 -> 0, +1 -> 1, -1 -> 2
static inline uint8_t trit_to_base3(int8_t t) {
    if (t == 1)  return 1;
    if (t == -1) return 2;
    return 0;
}

static inline int8_t base3_to_trit(uint8_t b) {
    if (b == 1) return 1;
    if (b == 2) return -1;
    return 0;
}

bool apu_lowquant_engine::lower_tq2_to_tace16(
    const void * tq2_src,
    void * tace_dst,
    size_t n_weights
) {
    if (!tq2_src || !tace_dst) return false;
    if (n_weights == 0 || (n_weights % 256) != 0) return false;

    const size_t n_blocks = n_weights / 256;
    const apu_raw_block_tq2_0 * src_blocks = reinterpret_cast<const apu_raw_block_tq2_0 *>(tq2_src);
    apu_tace16_tile * dst_tiles = reinterpret_cast<apu_tace16_tile *>(tace_dst);

    for (size_t b = 0; b < n_blocks; ++b) {
        const apu_raw_block_tq2_0 & sb = src_blocks[b];
        const float d = ggml_fp16_to_fp32(sb.d);

        // Dequantize 256 2-bit entries into trits
        int8_t trits[256];
        size_t idx = 0;
        for (size_t j = 0; j < sizeof(sb.qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = (sb.qs[j + m] >> (l * 2)) & 3;
                    trits[idx++] = (int8_t)q - 1; // 0->-1, 1->0, 2->1
                }
            }
        }

        // Each 256-weight block maps into 4 T-ACE 16B tiles (64 weights each)
        for (size_t tile_idx = 0; tile_idx < 4; ++tile_idx) {
            apu_tace16_tile & tile = dst_tiles[b * 4 + tile_idx];
            const int8_t * tile_trits = &trits[tile_idx * 64];

            // Pack 60 trits into 12 bytes (5 trits per byte: 3^5 = 243)
            for (size_t byte_idx = 0; byte_idx < 12; ++byte_idx) {
                uint8_t val = 0;
                uint8_t mul = 1;
                for (size_t k = 0; k < 5; ++k) {
                    uint8_t dig = trit_to_base3(tile_trits[byte_idx * 5 + k]);
                    val += dig * mul;
                    mul *= 3;
                }
                tile.packed_trits_main[byte_idx] = val;
            }

            // Pack remaining 4 trits into 1 byte (3^4 = 81)
            {
                uint8_t val = 0;
                uint8_t mul = 1;
                for (size_t k = 0; k < 4; ++k) {
                    uint8_t dig = trit_to_base3(tile_trits[60 + k]);
                    val += dig * mul;
                    mul *= 3;
                }
                tile.packed_trits_tail = val;
            }

            // Compute power-of-two base scale exponent S
            int exp = 0;
            float abs_d = fabsf(d);
            if (abs_d > 1e-12f) {
                frexpf(abs_d, &exp); // abs_d = frac * 2^exp
            }
            // Clamp exponent to uint8 range with bias 128
            int biased_exp = std::max(0, std::min(255, exp + 128));
            tile.base_exponent = static_cast<uint8_t>(biased_exp);

            // Compute 16x 1-bit subgroup shifts (4 weights per subgroup)
            uint16_t shifts = 0;
            for (size_t sg = 0; sg < 16; ++sg) {
                // If all weights in subgroup are zero or small, shift by 1 to refine precision
                int non_zero = 0;
                for (size_t elem = 0; elem < 4; ++elem) {
                    if (tile_trits[sg * 4 + elem] != 0) non_zero++;
                }
                if (non_zero <= 1) {
                    shifts |= (1 << sg);
                }
            }
            tile.subgroup_shifts = shifts;
        }
    }

    return true;
}

bool apu_lowquant_engine::dequantize_tace16_to_f32(
    const void * tace_src,
    float * f32_dst,
    size_t n_weights
) {
    if (!tace_src || !f32_dst) return false;
    if (n_weights == 0 || (n_weights % 64) != 0) return false;

    const size_t n_tiles = n_weights / 64;
    const apu_tace16_tile * tiles = reinterpret_cast<const apu_tace16_tile *>(tace_src);

    for (size_t t = 0; t < n_tiles; ++t) {
        const apu_tace16_tile & tile = tiles[t];
        float * dst = &f32_dst[t * 64];

        int exp = static_cast<int>(tile.base_exponent) - 128;
        float base_scale = ldexpf(1.0f, exp);

        // Unpack 60 trits
        for (size_t byte_idx = 0; byte_idx < 12; ++byte_idx) {
            uint8_t val = tile.packed_trits_main[byte_idx];
            for (size_t k = 0; k < 5; ++k) {
                uint8_t dig = val % 3;
                val /= 3;
                size_t elem_idx = byte_idx * 5 + k;
                size_t sg = elem_idx / 4;
                float sg_scale = (tile.subgroup_shifts & (1 << sg)) ? (base_scale * 0.5f) : base_scale;
                dst[elem_idx] = static_cast<float>(base3_to_trit(dig)) * sg_scale;
            }
        }

        // Unpack 4 tail trits
        {
            uint8_t val = tile.packed_trits_tail;
            for (size_t k = 0; k < 4; ++k) {
                uint8_t dig = val % 3;
                val /= 3;
                size_t elem_idx = 60 + k;
                size_t sg = elem_idx / 4;
                float sg_scale = (tile.subgroup_shifts & (1 << sg)) ? (base_scale * 0.5f) : base_scale;
                dst[elem_idx] = static_cast<float>(base3_to_trit(dig)) * sg_scale;
            }
        }
    }

    return true;
}

apu_memory_governor_status apu_lowquant_engine::query_memory_governor(size_t working_set_bytes) const {
    apu_memory_governor_status status{};

    FILE * f = fopen("/proc/meminfo", "r");
    if (f) {
        char key[64];
        uint64_t val = 0;
        char unit[16];
        while (fscanf(f, "%63s %llu %15s", key, (unsigned long long *)&val, unit) == 3) {
            if (!strcmp(key, "MemTotal:")) {
                status.total_ram_bytes = val * 1024ULL;
            } else if (!strcmp(key, "MemAvailable:")) {
                status.available_ram_bytes = val * 1024ULL;
            }
        }
        fclose(f);
    }

    status.peak_allocated_bytes = working_set_bytes;
    status.within_50gb_ceiling = (working_set_bytes <= APU_MAX_SYSTEM_MEMORY_CEILING_BYTES);

    // Tiered distillation hierarchy recommendation based on available headroom
    if (status.available_ram_bytes >= working_set_bytes * 3) {
        status.recommended_hierarchy = "layer";
    } else if (status.available_ram_bytes >= working_set_bytes) {
        status.recommended_hierarchy = "expert";
    } else {
        status.recommended_hierarchy = "micro_chunk";
    }

    return status;
}
