// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 3 Dynamic KV Cache Quantization Implementation
#include "ggml-apu-kv.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <sstream>

extern "C" {

void apu_backend_enable_kv_quant(struct llama_context_params * params, apu_kv_mode_t mode) {
    if (!params) return;

    switch (mode) {
        case APU_KV_MODE_FP16:
            params->type_k = GGML_TYPE_F16;
            params->type_v = GGML_TYPE_F16;
            break;
        case APU_KV_MODE_Q4_0:
            params->type_k = GGML_TYPE_Q4_0;
            params->type_v = GGML_TYPE_Q4_0;
            // Flash attention is required for quantized V cache
            if (params->flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ||
                params->flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
                params->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            }
            break;
        case APU_KV_MODE_AUTO:
        default:
            // Auto defaults to Q4_0 when hardware permits
            params->type_k = GGML_TYPE_Q4_0;
            params->type_v = GGML_TYPE_Q4_0;
            if (params->flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ||
                params->flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
                params->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            }
            break;
    }
}

bool apu_verify_kv_alignment(const void * ptr, size_t size, size_t chunk_stride) {
    if (!ptr) return false;

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    // 1. Strict 16-byte (128-bit) AIE2P Tile DMA chunk alignment
    if (addr % 16 != 0) {
        return false;
    }
    if (chunk_stride % 16 != 0) {
        return false;
    }

    // 2. 64-byte host CPU cacheline alignment for base allocations >= 64 bytes
    if (size >= 64 && (addr % 64 != 0)) {
        return false;
    }

    return true;
}

apu_kv_telemetry_t apu_compute_kv_telemetry(const struct llama_model * model, uint32_t n_ctx) {
    apu_kv_telemetry_t tel = { n_ctx, 0, 0, 1.0f };
    if (!model) return tel;

    const auto & hparams = model->hparams;
    uint32_t n_layer = hparams.n_layer();

    uint64_t total_elements_k = 0;
    uint64_t total_elements_v = 0;

    for (uint32_t il = 0; il < n_layer; ++il) {
        uint32_t head_k = hparams.n_embd_head_k(il);
        uint32_t head_v = hparams.n_embd_head_v(il);
        uint32_t n_head_kv = hparams.n_head_kv(il);

        total_elements_k += (uint64_t)head_k * n_head_kv * n_ctx;
        total_elements_v += (uint64_t)head_v * n_head_kv * n_ctx;
    }

    // FP16: 2 bytes per element
    tel.fp16_bytes = (total_elements_k + total_elements_v) * 2;

    // Q4_0: 18 bytes per 32 elements (16 bytes quantized nibbles + 2 bytes FP16 scale)
    // 18 / 32 = 0.5625 bytes per element
    uint64_t q4_blocks_k = (total_elements_k + 31) / 32;
    uint64_t q4_blocks_v = (total_elements_v + 31) / 32;
    tel.q4_0_bytes = (q4_blocks_k + q4_blocks_v) * 18;

    if (tel.q4_0_bytes > 0) {
        tel.reduction_ratio = (float)tel.fp16_bytes / (float)tel.q4_0_bytes;
    }

    return tel;
}

} // extern "C"

bool apu_evaluate_kv_quant_compatibility(
    const struct llama_model * model,
    apu_kv_mode_t requested_mode,
    apu_kv_mode_t & resolved_mode,
    std::string & reason
) {
    if (!model) {
        resolved_mode = requested_mode;
        reason = "Null model; keeping requested mode";
        return false;
    }

    const auto & hparams = model->hparams;
    const uint32_t n_layer = hparams.n_layer();

    // Check 1: User explicit FP16 override
    if (requested_mode == APU_KV_MODE_FP16) {
        resolved_mode = APU_KV_MODE_FP16;
        reason = "User requested uncompressed FP16 baseline";
        return true;
    }

    // Check 2: Compound perplexity heuristic: sub-2-bit / 1-bit models
    // If the base model weights are already extremely quantized (sub-2-bit),
    // compounding low-bit weights with low-bit KV causes rapid perplexity collapse.
    if (model->ftype() == LLAMA_FTYPE_MOSTLY_Q1_0 || model->ftype() == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
        resolved_mode = APU_KV_MODE_FP16;
        reason = "Heuristic safeguard: deep sub-2-bit model detected; forcing FP16 KV to prevent compound perplexity collapse";
        return true;
    }

    // Check 3: Block size divisibility for Q4_0 (QK4_0 = 32)
    const uint32_t blck_size = 32;
    for (uint32_t il = 0; il < n_layer; ++il) {
        uint32_t head_k = hparams.n_embd_head_k(il);
        uint32_t head_v = hparams.n_embd_head_v(il);

        if (head_k % blck_size != 0) {
            resolved_mode = APU_KV_MODE_FP16;
            std::ostringstream ss;
            ss << "Head dimension K (" << head_k << ") at layer " << il
               << " not divisible by Q4_0 block size (32); falling back to FP16";
            reason = ss.str();
            return false;
        }

        if (head_v % blck_size != 0) {
            resolved_mode = APU_KV_MODE_FP16;
            std::ostringstream ss;
            ss << "Head dimension V (" << head_v << ") at layer " << il
               << " not divisible by Q4_0 block size (32); falling back to FP16";
            reason = ss.str();
            return false;
        }
    }

    // Optimal APU route: Q4_0 dynamic quantization approved
    resolved_mode = APU_KV_MODE_Q4_0;
    reason = "Optimal APU route: Q4_0 dynamic KV quantization enabled (~3.56x DRAM reduction, 16B Tile DMA aligned)";
    return true;
}

apu_kv_mode_t apu_kv_mode_from_string(const std::string & s) {
    if (s == "q4_0" || s == "Q4_0" || s == "q4") {
        return APU_KV_MODE_Q4_0;
    }
    if (s == "fp16" || s == "f16" || s == "FP16" || s == "F16") {
        return APU_KV_MODE_FP16;
    }
    return APU_KV_MODE_AUTO;
}

const char * apu_kv_mode_to_string(apu_kv_mode_t mode) {
    switch (mode) {
        case APU_KV_MODE_Q4_0: return "q4_0";
        case APU_KV_MODE_FP16: return "fp16";
        case APU_KV_MODE_AUTO:
        default:               return "auto";
    }
}
