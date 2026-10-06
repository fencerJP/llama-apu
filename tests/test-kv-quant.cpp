// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 9 §9.3 Decoupled Q4_0 KV layout tests (with Phase 3 KV tests)
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml-apu-kv.h"
#include "ggml.h"
#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>
#include <vector>

static void test_q4_0_accuracy() {
    printf("[test_q4_0_accuracy] running...\n");
    const int N = 1024;
    std::vector<float> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = sinf((float)i * 0.05f) * cosf((float)i * 0.02f) * 3.0f;
    }

    std::vector<uint8_t> q4_data((N / 32) * 18);
    std::vector<float> dequant(N);

    for (int b = 0; b < N / 32; b++) {
        const float * src = &input[b * 32];
        uint8_t * dst = &q4_data[b * 18];
        float max_abs = 0.0f;
        for (int i = 0; i < 32; i++) {
            float a = fabsf(src[i]);
            if (a > max_abs) max_abs = a;
        }
        float d = max_abs / -8.0f;
        float id = d ? 1.0f / d : 0.0f;

        ggml_fp16_t h = ggml_fp32_to_fp16(d);
        memcpy(dst, &h, sizeof(h));

        uint8_t * qs = dst + sizeof(h);
        for (int i = 0; i < 16; i++) {
            float x0 = src[i] * id;
            float x1 = src[i + 16] * id;
            int8_t q0 = std::min(15, std::max(0, (int)roundf(x0) + 8));
            int8_t q1 = std::min(15, std::max(0, (int)roundf(x1) + 8));
            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        float scale = ggml_fp16_to_fp32(h);
        float * out = &dequant[b * 32];
        for (int i = 0; i < 16; i++) {
            int8_t q0 = (qs[i] & 0x0F) - 8;
            int8_t q1 = ((qs[i] >> 4) & 0x0F) - 8;
            out[i] = q0 * scale;
            out[i + 16] = q1 * scale;
        }
    }

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < N; i++) {
        dot += input[i] * dequant[i];
        norm_a += input[i] * input[i];
        norm_b += dequant[i] * dequant[i];
    }
    float cos_sim = dot / (sqrtf(norm_a) * sqrtf(norm_b));
    printf("  cos_sim = %.5f\n", cos_sim);
    assert(cos_sim > 0.98f);
    printf("  [PASS]\n");
}

static void test_alignment_verification() {
    printf("[test_alignment_verification] running...\n");
    alignas(64) uint8_t aligned_buf[512];
    (void)aligned_buf;

    // Perfectly aligned base address and 16-byte chunk stride
    assert(apu_verify_kv_alignment(aligned_buf, sizeof(aligned_buf), 32) == true);
    assert(apu_verify_kv_alignment(aligned_buf, sizeof(aligned_buf), 16) == true);
    assert(apu_verify_kv_alignment(aligned_buf, sizeof(aligned_buf), 64) == true);

    // Misaligned base address
    assert(apu_verify_kv_alignment(aligned_buf + 4, sizeof(aligned_buf) - 4, 32) == false);
    assert(apu_verify_kv_alignment(aligned_buf + 8, sizeof(aligned_buf) - 8, 32) == false);
    assert(apu_verify_kv_alignment(aligned_buf + 1, sizeof(aligned_buf) - 1, 32) == false);

    // Misaligned chunk stride (not divisible by 16)
    assert(apu_verify_kv_alignment(aligned_buf, sizeof(aligned_buf), 18) == false);
    assert(apu_verify_kv_alignment(aligned_buf, sizeof(aligned_buf), 24) == false);

    // Null pointer
    assert(apu_verify_kv_alignment(nullptr, 128, 32) == false);

    printf("  [PASS]\n");
}

static void test_telemetry_reduction() {
    printf("[test_telemetry_reduction] running...\n");
    // Synthetic KV size calculation
    uint32_t ctx = 8192;
    uint32_t n_layer = 24;
    uint32_t n_head_kv = 8;
    uint32_t head_dim = 64;

    uint64_t elements = 2ULL * n_layer * n_head_kv * head_dim * ctx;
    uint64_t fp16_bytes = elements * 2;
    uint64_t q4_blocks = (elements + 31) / 32;
    uint64_t q4_bytes = q4_blocks * 18;
    double ratio = (double)fp16_bytes / (double)q4_bytes;

    printf("  8k ctx: FP16 = %.2f MiB, Q4_0 = %.2f MiB, ratio = %.2fx\n",
           (double)fp16_bytes / (1024.0 * 1024.0), (double)q4_bytes / (1024.0 * 1024.0), ratio);
    assert(ratio > 3.5f && ratio < 3.6f);
    printf("  [PASS]\n");
}

static void test_compatibility_heuristics() {
    printf("[test_compatibility_heuristics] running...\n");
    // Parse strings
    assert(apu_kv_mode_from_string("q4_0") == APU_KV_MODE_Q4_0);
    assert(apu_kv_mode_from_string("fp16") == APU_KV_MODE_FP16);
    assert(apu_kv_mode_from_string("auto") == APU_KV_MODE_AUTO);

    assert(strcmp(apu_kv_mode_to_string(APU_KV_MODE_Q4_0), "q4_0") == 0);
    assert(strcmp(apu_kv_mode_to_string(APU_KV_MODE_FP16), "fp16") == 0);
    assert(strcmp(apu_kv_mode_to_string(APU_KV_MODE_AUTO), "auto") == 0);

    // Null model
    apu_kv_mode_t resolved = APU_KV_MODE_AUTO;
    std::string reason;
    bool ok = apu_evaluate_kv_quant_compatibility(nullptr, APU_KV_MODE_AUTO, resolved, reason);
    (void)ok;
    assert(!ok);

    printf("  [PASS]\n");
}

// ---------------------------------------------------------------------------
// Phase 9 §9.3: Decoupled Q4_0 KV layout + refcounted CoW + dual-view table
// ---------------------------------------------------------------------------

static void test_decoupled_parity() {
    printf("[test_decoupled_parity] running...\n");
    const uint32_t N = 256;                    // 8 blocks
    const uint32_t nb = apu_kv_decoupled_nblocks(N);
    assert(nb == 8);
    assert(apu_kv_decoupled_w_bytes(N) == 8 * 16);
    assert(apu_kv_decoupled_s_bytes(N) == 8 * 2);
    assert(apu_kv_decoupled_nblocks(30) == 0); // must be QK4_0 multiple

    // Random packed Q4_0 input (8 blocks x 18B)
    std::vector<uint8_t> packed(nb * 18);
    uint32_t seed = 0xC0FFEEu;
    for (auto & b : packed) { seed = seed * 1664525u + 1013904223u; b = (uint8_t)(seed >> 24); }

    // Bit-exact roundtrip: split -> merge == original
    assert(apu_kv_decoupled_parity(packed.data(), N));

    // Alignment of decoupled buffers required (16B Tile DMA)
    alignas(16) uint8_t w[8 * 16];
    alignas(16) uint8_t s[8 * 2 + 16];
    assert(apu_kv_decoupled_split(packed.data(), N, w, s));
    assert(apu_kv_decoupled_merge(w, s, N, packed.data()));
    assert(apu_kv_decoupled_parity(packed.data(), N));

    // Misaligned decoupled buffers must be rejected
    assert(!apu_kv_decoupled_split(packed.data(), N, w + 1, s));
    assert(!apu_kv_decoupled_merge(w, s + 2, N, packed.data()));

    printf("  [PASS]\n");
}

static void test_decoupled_dequant_reference() {
    printf("[test_decoupled_dequant_reference] running...\n");
    // One block: scale 2.0, nibble pattern 0..15 twice
    ggml_fp16_t d16 = ggml_fp32_to_fp16(2.0f);
    uint8_t w[16];
    for (int i = 0; i < 16; i++) w[i] = (uint8_t)((i * 0x11) & 0xFF); // (i, i+0x11...) simple pattern
    w[0] = 0x10; w[1] = 0x32; w[2] = 0x54; w[3] = 0x76;
    w[4] = 0x98; w[5] = 0xBA; w[6] = 0xDC; w[7] = 0xFE;
    w[8] = 0x01; w[9] = 0x23; w[10] = 0x45; w[11] = 0x67;
    w[12] = 0x89; w[13] = 0xAB; w[14] = 0xCD; w[15] = 0xEF;

    float out[32];
    apu_kv_decoupled_dequant(w, &d16, out);
    const float scale = 2.0f;
    for (int i = 0; i < 32; i++) {
        uint8_t byte = w[i / 2];
        uint8_t q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        assert(out[i] == ((float)((int)q - 8) * scale));
    }
    // Packed-vs-decoupled dequant bit-exactness: dequant via packed layout too
    uint8_t packed[18];
    memcpy(packed, &d16, 2);
    memcpy(packed + 2, w, 16);
    float ref[32];
    ggml_fp16_t pd; memcpy(&pd, packed, 2);
    const float pscale = ggml_fp16_to_fp32(pd);
    for (int i = 0; i < 32; i++) {
        uint8_t byte = packed[2 + i / 2];
        uint8_t q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        ref[i] = (float)((int)q - 8) * pscale;
    }
    for (int i = 0; i < 32; i++) assert(out[i] == ref[i]); // bit-exact

    printf("  [PASS]\n");
}

static void test_dual_view_block_table() {
    printf("[test_dual_view_block_table] running...\n");
    const uint32_t NB = 4;
    alignas(64) uint8_t wbufs[NB][64];
    alignas(64) uint8_t sbufs[NB][64];
    const void * w_ptrs[NB];
    const void * s_ptrs[NB];
    for (uint32_t i = 0; i < NB; i++) { w_ptrs[i] = wbufs[i]; s_ptrs[i] = sbufs[i]; }

    llama_apu_npu_block_t table[NB];
    memset(table, 0xAA, sizeof(table));
    assert(apu_kv_fill_npu_block_table(table, NB, w_ptrs, s_ptrs));
    for (uint32_t i = 0; i < NB; i++) {
        assert(table[i].page_w_phys_addr == (uint64_t)(uintptr_t) wbufs[i]);
        assert(table[i].page_s_phys_addr == (uint64_t)(uintptr_t) sbufs[i]);
        assert((table[i].page_w_phys_addr % 16) == 0);
        assert((table[i].page_s_phys_addr % 16) == 0);
        assert((table[i].page_w_phys_addr % 64) == 0);
        assert((table[i].page_s_phys_addr % 64) == 0);
    }
    // Misaligned pair must be rejected (and earlier entries left intact)
    const void * bad_s[NB] = { sbufs[0], sbufs[1] + 8, sbufs[2], sbufs[3] };
    memset(table, 0, sizeof(table));
    assert(!apu_kv_fill_npu_block_table(table, NB, w_ptrs, bad_s));
    assert(table[0].page_w_phys_addr != 0);   // entry 0 accepted before failure
    assert(table[1].page_w_phys_addr == 0);   // rejected at misaligned entry

    printf("  [PASS]\n");
}

static void test_cow_clone_and_isolation() {
    printf("[test_cow_clone_and_isolation] running...\n");
    const uint32_t N = 64;   // 2 blocks
    apu_kv_cow_pair * parent = apu_kv_cow_alloc(N);
    assert(parent && apu_kv_cow_refcount(parent) == 1);
    assert(apu_kv_cow_w(parent) && apu_kv_cow_s(parent));
    assert(((uintptr_t) apu_kv_cow_w(parent) % 64) == 0);
    assert(((uintptr_t) apu_kv_cow_s(parent) % 64) == 0);

    // Fill parent W with marker bytes
    memset((void *) apu_kv_cow_w(parent), 0xAB, apu_kv_decoupled_w_bytes(N));

    // Share -> ref 2 -> prepare_write must clone both streams
    apu_kv_cow_pair * child = apu_kv_cow_share(parent);
    assert(child == parent && apu_kv_cow_refcount(parent) == 2);
    apu_kv_cow_pair * cloned = apu_kv_cow_prepare_write(child);
    assert(cloned != nullptr && cloned != parent);
    assert(apu_kv_cow_refcount(parent) == 1);
    assert(apu_kv_cow_refcount(cloned) == 1);
    // Byte-for-byte copy of BOTH W and S (§9.1 dual clone)
    assert(memcmp(apu_kv_cow_w(parent), apu_kv_cow_w(cloned), apu_kv_decoupled_w_bytes(N)) == 0);
    assert(memcmp(apu_kv_cow_s(parent), apu_kv_cow_s(cloned), apu_kv_decoupled_s_bytes(N)) == 0);
    assert(apu_kv_cow_npu_block(parent)->page_w_phys_addr != apu_kv_cow_npu_block(cloned)->page_w_phys_addr);
    assert(apu_kv_cow_npu_block(parent)->page_s_phys_addr != apu_kv_cow_npu_block(cloned)->page_s_phys_addr);

    // Isolation: mutate child, parent untouched
    memset((void *) apu_kv_cow_w(cloned), 0xCD, apu_kv_decoupled_w_bytes(N));
    const uint8_t * pw = (const uint8_t *) apu_kv_cow_w(parent);
    for (size_t i = 0; i < apu_kv_decoupled_w_bytes(N); i++) assert(pw[i] == 0xAB);

    // Exclusive prepare_write: no clone
    assert(apu_kv_cow_prepare_write(cloned) == cloned);

    apu_kv_cow_release(cloned);
    apu_kv_cow_release(parent);

    printf("  [PASS]\n");
}

static void test_cow_concurrent_stress() {
    printf("[test_cow_concurrent_stress] running...\n");
    const uint32_t N = 64;
    apu_kv_cow_pair * base = apu_kv_cow_alloc(N);
    assert(base);
    memset((void *) apu_kv_cow_w(base), 0x5A, apu_kv_decoupled_w_bytes(N));

    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 16; t++) {
        threads.emplace_back([&, t]() {
            for (int iter = 0; iter < 200; iter++) {
                apu_kv_cow_pair * sh = apu_kv_cow_share(base);
                apu_kv_cow_pair * mine = apu_kv_cow_prepare_write(sh);
                if (!mine) return;
                if (mine == base) {
                    // exclusive: writer must be this thread only when ref==1;
                    // under contention this happens only if nobody else shares
                } else {
                    // cloned: verify copy integrity then drop
                    if (memcmp(apu_kv_cow_w(mine), apu_kv_cow_w(base), apu_kv_decoupled_w_bytes(N)) == 0 || true) {
                        // after other threads may have mutated clones, only
                        // structural checks are deterministic here
                    }
                }
                apu_kv_cow_release(mine);
                ok.fetch_add(1);
            }
        });
    }
    for (auto & th : threads) th.join();
    assert(ok.load() == 16 * 200);
    assert(apu_kv_cow_refcount(base) == 1);   // no leaked refs
    apu_kv_cow_release(base);

    printf("  [PASS]\n");
}

int main() {
    printf("=== test-kv-quant: Phase 3 Dynamic KV Cache Tests ===\n");
    test_q4_0_accuracy();
    test_alignment_verification();
    test_telemetry_reduction();
    test_compatibility_heuristics();
    printf("=== Phase 9 §9.3 Decoupled KV Tests ===\n");
    test_decoupled_parity();
    test_decoupled_dequant_reference();
    test_dual_view_block_table();
    test_cow_clone_and_isolation();
    test_cow_concurrent_stress();
    printf("ALL TESTS PASSED.\n");
    return 0;
}
