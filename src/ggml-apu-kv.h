// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 3 Dynamic KV Cache Quantization Substrate
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "ggml.h"
#include "llama.h"

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

#endif
