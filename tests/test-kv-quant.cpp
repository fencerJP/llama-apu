// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 3 Dynamic KV Cache Quantization Unit Tests
#include "ggml-apu-kv.h"
#include "ggml.h"
#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
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

int main() {
    printf("=== test-kv-quant: Phase 3 Dynamic KV Cache Tests ===\n");
    test_q4_0_accuracy();
    test_alignment_verification();
    test_telemetry_reduction();
    test_compatibility_heuristics();
    printf("ALL TESTS PASSED.\n");
    return 0;
}
