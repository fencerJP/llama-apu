// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 3 Dynamic KV Cache Quantization Implementation
#include "ggml-apu-kv.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cstring>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <new>
#include <vector>

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

// ---------------------------------------------------------------------------
// Phase 9 §9.3: Decoupled Q4_0 KV layout (W_kv / S_kv)
//
// Packed block_q4_0 (ggml-common.h): { ggml_half d; uint8_t qs[16]; } = 18B.
// The 18B stride violates the 16B AIE2P Tile DMA beat, so the nibble and
// scale streams are split into two independently 16-byte-aligned buffers.
// ---------------------------------------------------------------------------

static constexpr uint32_t APU_Q4_0_BITS      = 4;
static constexpr uint32_t APU_Q4_0_BLK       = 32;   // QK4_0
static constexpr uint32_t APU_Q4_0_W_BYTES   = 16;   // 32 * 4bit
static constexpr uint32_t APU_Q4_0_S_BYTES   = 2;    // ggml_half
static constexpr uint32_t APU_Q4_0_PACKED    = 18;   // 2 + 16

uint32_t apu_kv_decoupled_nblocks(uint32_t n_elements) {
    if (n_elements == 0 || n_elements % APU_Q4_0_BLK != 0) return 0;
    return n_elements / APU_Q4_0_BLK;
}

size_t apu_kv_decoupled_w_bytes(uint32_t n_elements) {
    return (size_t) apu_kv_decoupled_nblocks(n_elements) * APU_Q4_0_W_BYTES;
}

size_t apu_kv_decoupled_s_bytes(uint32_t n_elements) {
    return (size_t) apu_kv_decoupled_nblocks(n_elements) * APU_Q4_0_S_BYTES;
}

bool apu_kv_decoupled_split(const void * packed, uint32_t n_elements,
                            void * w_kv, void * s_kv) {
    const uint32_t nb = apu_kv_decoupled_nblocks(n_elements);
    if (!nb || !packed || !w_kv || !s_kv) return false;
    if ((uintptr_t) w_kv % 16 != 0 || (uintptr_t) s_kv % 16 != 0) return false;

    const uint8_t * src = (const uint8_t *) packed;
    uint8_t * w = (uint8_t *) w_kv;
    uint8_t * s = (uint8_t *) s_kv;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t * blk = src + (size_t) b * APU_Q4_0_PACKED;
        memcpy(s + (size_t) b * APU_Q4_0_S_BYTES, blk,                APU_Q4_0_S_BYTES); // d
        memcpy(w + (size_t) b * APU_Q4_0_W_BYTES, blk + APU_Q4_0_S_BYTES, APU_Q4_0_W_BYTES); // qs
    }
    return true;
}

bool apu_kv_decoupled_merge(const void * w_kv, const void * s_kv,
                            uint32_t n_elements, void * packed_out) {
    const uint32_t nb = apu_kv_decoupled_nblocks(n_elements);
    if (!nb || !w_kv || !s_kv || !packed_out) return false;
    if ((uintptr_t) w_kv % 16 != 0 || (uintptr_t) s_kv % 16 != 0) return false;

    const uint8_t * w = (const uint8_t *) w_kv;
    const uint8_t * s = (const uint8_t *) s_kv;
    uint8_t * dst = (uint8_t *) packed_out;
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t * blk = dst + (size_t) b * APU_Q4_0_PACKED;
        memcpy(blk,                s + (size_t) b * APU_Q4_0_S_BYTES, APU_Q4_0_S_BYTES);
        memcpy(blk + APU_Q4_0_S_BYTES, w + (size_t) b * APU_Q4_0_W_BYTES, APU_Q4_0_W_BYTES);
    }
    return true;
}

bool apu_kv_decoupled_parity(const void * packed, uint32_t n_elements) {
    const uint32_t nb = apu_kv_decoupled_nblocks(n_elements);
    if (!nb || !packed) return false;

    const size_t w_bytes = (size_t) nb * APU_Q4_0_W_BYTES;
    const size_t s_bytes = (size_t) nb * APU_Q4_0_S_BYTES;
    const size_t packed_bytes = (size_t) nb * APU_Q4_0_PACKED;

    // 16-byte aligned scratch (stack buffers are 16B aligned via alignas)
    alignas(16) uint8_t w[16];       // grow via heap for large n
    alignas(16) uint8_t s[16];
    std::vector<uint8_t> wbuf(w_bytes ? w_bytes : 16, 0);
    std::vector<uint8_t> sbuf(s_bytes ? s_bytes : 16, 0);
    std::vector<uint8_t> roundtrip(packed_bytes, 0);
    (void) w; (void) s;
    // vector data of size >= 16 is 16B aligned for these small allocations
    // (malloc guarantees max_align_t = 16B on x86-64 glibc).
    if (!apu_kv_decoupled_split(packed, n_elements, wbuf.data(), sbuf.data())) return false;
    if (!apu_kv_decoupled_merge(wbuf.data(), sbuf.data(), n_elements, roundtrip.data())) return false;
    return memcmp(packed, roundtrip.data(), packed_bytes) == 0;
}

void apu_kv_decoupled_dequant(const void * w_block, const void * s_scale,
                              float * out32) {
    if (!w_block || !s_scale || !out32) return;
    ggml_fp16_t d16;
    memcpy(&d16, s_scale, sizeof(d16));
    const float d = ggml_fp16_to_fp32(d16);
    const uint8_t * qs = (const uint8_t *) w_block;
    for (uint32_t i = 0; i < APU_Q4_0_BLK; ++i) {
        const uint8_t byte = qs[i / 2];
        const uint8_t q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        out32[i] = ((int32_t) q - 8) * d;   // Q4_0 reference: x = (q - 8) * d
    }
}

bool apu_kv_fill_npu_block_table(llama_apu_npu_block_t * out,
                                 uint32_t num_blocks,
                                 const void * const * w_ptrs,
                                 const void * const * s_ptrs) {
    if (!out || !num_blocks || !w_ptrs || !s_ptrs) return false;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        const uintptr_t w = (uintptr_t) w_ptrs[i];
        const uintptr_t s = (uintptr_t) s_ptrs[i];
        // §9.3: 16B Tile DMA beat on both streams + 64B host cacheline
        if (!w || !s) return false;
        if (w % 16 != 0 || s % 16 != 0) return false;
        if (w % 64 != 0 || s % 64 != 0) return false;
        out[i].page_w_phys_addr = (uint64_t) w;
        out[i].page_s_phys_addr = (uint64_t) s;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Refcounted decoupled CoW page pair (§9.1): clones W_kv and S_kv together so
// GPU (block_table_gpu) and NPU (npu_block_table) views never diverge.
// ---------------------------------------------------------------------------

struct apu_kv_cow_pair {
    uint32_t n_elements = 0;
    void *   w = nullptr;   // 64B-aligned host allocation (W_kv)
    void *   s = nullptr;   // 64B-aligned host allocation (S_kv)
    llama_apu_npu_block_t npu{};  // host "physical" view for dual-view tables
    std::atomic<uint32_t> ref{1};
};

static std::mutex g_cow_mutex;   // guards share/prepare_write/release races

struct apu_kv_cow_pair * apu_kv_cow_alloc(uint32_t n_elements) {
    const uint32_t nb = apu_kv_decoupled_nblocks(n_elements);
    if (!nb) return nullptr;
    auto * p = new (std::nothrow) apu_kv_cow_pair();
    if (!p) return nullptr;
    p->n_elements = n_elements;
    const size_t w_bytes = (size_t) nb * APU_Q4_0_W_BYTES;
    const size_t s_bytes = (size_t) nb * APU_Q4_0_S_BYTES;
    // posix_memalign: guaranteed 64B host cacheline + 16B Tile DMA alignment
    if (posix_memalign(&p->w, 64, w_bytes) != 0 ||
        posix_memalign(&p->s, 64, s_bytes) != 0) {
        free(p->w); free(p->s); delete p; return nullptr;
    }
    memset(p->w, 0, w_bytes);
    memset(p->s, 0, s_bytes);
    p->npu.page_w_phys_addr = (uint64_t)(uintptr_t) p->w;
    p->npu.page_s_phys_addr = (uint64_t)(uintptr_t) p->s;
    return p;
}

struct apu_kv_cow_pair * apu_kv_cow_share(struct apu_kv_cow_pair * p) {
    if (!p) return nullptr;
    std::lock_guard<std::mutex> lock(g_cow_mutex);
    p->ref.fetch_add(1, std::memory_order_acq_rel);
    return p;
}

struct apu_kv_cow_pair * apu_kv_cow_prepare_write(struct apu_kv_cow_pair * p) {
    if (!p) return nullptr;
    std::lock_guard<std::mutex> lock(g_cow_mutex);
    if (p->ref.load(std::memory_order_acquire) == 1) {
        return p;   // exclusive owner: no copy needed
    }
    // Shared page: clone BOTH streams atomically (§9.1 dual W/S clone)
    auto * c = apu_kv_cow_alloc(p->n_elements);
    if (!c) return nullptr;   // OOM: caller keeps shared ref, must not write
    memcpy(c->w, p->w, apu_kv_decoupled_w_bytes(p->n_elements));
    memcpy(c->s, p->s, apu_kv_decoupled_s_bytes(p->n_elements));
    p->ref.fetch_sub(1, std::memory_order_acq_rel);  // release caller's share
    return c;   // ref == 1 (fresh)
}

void apu_kv_cow_release(struct apu_kv_cow_pair * p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(g_cow_mutex);
    if (p->ref.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        free(p->w);
        free(p->s);
        delete p;
    }
}

uint32_t apu_kv_cow_refcount(const struct apu_kv_cow_pair * p) {
    return p ? p->ref.load(std::memory_order_acquire) : 0;
}

const void * apu_kv_cow_w(const struct apu_kv_cow_pair * p) { return p ? p->w : nullptr; }
const void * apu_kv_cow_s(const struct apu_kv_cow_pair * p) { return p ? p->s : nullptr; }
const llama_apu_npu_block_t * apu_kv_cow_npu_block(const struct apu_kv_cow_pair * p) {
    return p ? &p->npu : nullptr;
}
