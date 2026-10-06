// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 3 Dynamic KV Cache Quantization Substrate
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "ggml.h"
#include "llama.h"
#include "llama-apu-scheduler.h"   // llama_apu_npu_block_t (§9.3 dual-view)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    APU_KV_MODE_AUTO = 0,
    APU_KV_MODE_Q4_0 = 1,
    APU_KV_MODE_FP16 = 2,
} apu_kv_mode_t;

typedef struct {
    uint32_t ctx_tokens;
    uint64_t fp16_bytes;
    uint64_t q4_0_bytes;
    float    reduction_ratio; // e.g. ~3.5x reduction
} apu_kv_telemetry_t;

// C ABI Interface
void apu_backend_enable_kv_quant(struct llama_context_params * params, apu_kv_mode_t mode);
bool apu_verify_kv_alignment(const void * ptr, size_t size, size_t chunk_stride);
apu_kv_telemetry_t apu_compute_kv_telemetry(const struct llama_model * model, uint32_t n_ctx);

#ifdef __cplusplus
}

// C++ API
bool apu_evaluate_kv_quant_compatibility(
    const struct llama_model * model,
    apu_kv_mode_t requested_mode,
    apu_kv_mode_t & resolved_mode,
    std::string & reason
);

apu_kv_mode_t apu_kv_mode_from_string(const std::string & s);
const char *  apu_kv_mode_to_string(apu_kv_mode_t mode);

// ---------------------------------------------------------------------------
// Phase 9 §9.3: Decoupled Q4_0 KV layout (W_kv / S_kv) + refcounted CoW pairs
//
// Packed block_q4_0 is 18 bytes (2B FP16 delta + 16B nibbles) which violates
// the 16-byte AIE2P Tile DMA beat. The decoupled layout stores:
//   W_kv: pure nibble stream, 16 bytes per 32-element block (16B aligned)
//   S_kv: parallel FP16 scale stream, 2 bytes per block (16B aligned)
// All split/merge operations are bit-exact against the packed reference.
// ---------------------------------------------------------------------------

// Block math: n_elements must be a multiple of QK4_0 (32).
uint32_t apu_kv_decoupled_nblocks(uint32_t n_elements);
size_t   apu_kv_decoupled_w_bytes(uint32_t n_elements); // nblocks * 16
size_t   apu_kv_decoupled_s_bytes(uint32_t n_elements); // nblocks * 2

// Split packed block_q4_0[] into decoupled W_kv/S_kv. Outputs must be
// 16-byte aligned (w_bytes/s_bytes sized). Returns false on size mismatch.
bool apu_kv_decoupled_split(const void * packed, uint32_t n_elements,
                            void * w_kv, void * s_kv);
// Merge decoupled streams back into packed block_q4_0[] (bit-exact inverse).
bool apu_kv_decoupled_merge(const void * w_kv, const void * s_kv,
                            uint32_t n_elements, void * packed_out);
// Roundtrip parity: split+merge must reproduce the packed input byte-for-byte.
bool apu_kv_decoupled_parity(const void * packed, uint32_t n_elements);

// Reference dequantization of one decoupled block -> 32 floats.
// Must be bit-identical to packed Q4_0 dequant: x_i = (q_i - 8) * d.
void apu_kv_decoupled_dequant(const void * w_block, const void * s_scale,
                              float * out32);

// Dual-view NPU block table builder (§9.3): validates every (w, s) address
// pair against 16-byte Tile DMA + 64-byte cacheline rules and fills the
// scatter-gather list. Returns false (leaving entries zeroed) if any pair is
// misaligned.
bool apu_kv_fill_npu_block_table(llama_apu_npu_block_t * out,
                                 uint32_t num_blocks,
                                 const void * const * w_ptrs,
                                 const void * const * s_ptrs);

// Refcounted decoupled page pair for Copy-On-Write of shared prompt KV pages
// (§9.1). Both the INT4 weight buffer and the FP16 scale buffer are cloned
// together so GPU/NPU views never diverge.
struct apu_kv_cow_pair;
struct apu_kv_cow_pair * apu_kv_cow_alloc(uint32_t n_elements);   // ref = 1
struct apu_kv_cow_pair * apu_kv_cow_share(struct apu_kv_cow_pair * p);   // ref++
// If shared (ref > 1): clone W+S into a fresh pair, release caller's ref on
// the original, return the clone (ref = 1). If exclusive: return p unchanged.
struct apu_kv_cow_pair * apu_kv_cow_prepare_write(struct apu_kv_cow_pair * p);
void apu_kv_cow_release(struct apu_kv_cow_pair * p);   // ref--; free at 0
uint32_t apu_kv_cow_refcount(const struct apu_kv_cow_pair * p);
const void * apu_kv_cow_w(const struct apu_kv_cow_pair * p);
const void * apu_kv_cow_s(const struct apu_kv_cow_pair * p);
const llama_apu_npu_block_t * apu_kv_cow_npu_block(const struct apu_kv_cow_pair * p);

#endif
