/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_apu_c_api.c
 * @brief Standalone C program verifying the public C ABI for apu-backend.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

#include "../include/apu_backend.h"

int main(void) {
    printf("========================================================\n");
    printf("  Testing apu-backend C ABI Integration with llama.cpp\n");
    printf("========================================================\n");

    const char* mock_model_path = "/tmp/c_abi_validation_model.gguf";

    // Create a mock GGUF model file
    FILE* fp = fopen(mock_model_path, "wb");
    assert(fp != NULL);
    const char magic[4] = {'G', 'G', 'U', 'F'};
    uint32_t version = 3;
    uint64_t tensor_count = 10;
    uint64_t kv_count = 5;
    fwrite(magic, 1, 4, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&tensor_count, sizeof(tensor_count), 1, fp);
    fwrite(&kv_count, sizeof(kv_count), 1, fp);
    uint8_t dummy_weights[2048];
    memset(dummy_weights, 0x55, sizeof(dummy_weights));
    fwrite(dummy_weights, 1, sizeof(dummy_weights), fp);
    fclose(fp);

    printf("[1/6] Loading model via apu_backend_load_model()...\n");
    ApuBackendContext* ctx = NULL;
    int rc = apu_backend_load_model(mock_model_path, NULL, &ctx);
    printf("      apu_backend_load_model returned %d, ctx=%p\n", rc, (void*)ctx);
    assert(rc == 0);
    assert(ctx != NULL);

    printf("[2/6] Querying model hyperparameters via apu_backend_get_hyperparams()...\n");
    ApuModelHyperparams params;
    memset(&params, 0, sizeof(params));
    rc = apu_backend_get_hyperparams(ctx, &params);
    printf("      apu_backend_get_hyperparams returned %d\n", rc);
    printf("      Model: hidden_dim=%u, layers=%u, heads=%u, vocab=%u\n",
           params.hidden_dim, params.num_layers, params.num_heads, params.vocab_size);
    assert(rc == 0);
    assert(params.hidden_dim > 0);
    assert(params.num_layers > 0);

    printf("[3/6] Allocating shared dma-buf KV cache via apu_backend_allocate_shared_kv()...\n");
    int dmabuf_fd = -1;
    rc = apu_backend_allocate_shared_kv(ctx, 4 * 1024 * 1024, &dmabuf_fd);
    printf("      apu_backend_allocate_shared_kv returned %d, dmabuf_fd=%d\n", rc, dmabuf_fd);
    assert(rc == 0);
    assert(dmabuf_fd >= 0);

    printf("[4/6] Dispatching prompt prefill pass on RDNA 3.5 iGPU...\n");
    uint32_t prompt_tokens[] = {1, 256, 1024, 4096, 16384};
    size_t num_tokens = sizeof(prompt_tokens) / sizeof(prompt_tokens[0]);
    uint32_t initial_token = 0;
    rc = apu_backend_dispatch_prefill(ctx, prompt_tokens, num_tokens, -1, 1, &initial_token);
    printf("      apu_backend_dispatch_prefill returned %d, initial_token=%u\n", rc, initial_token);
    assert(rc == 0);
    assert(initial_token > 0);

    printf("[5/6] Dispatching decode step on XDNA 2 NPU...\n");
    uint32_t output_token = 0;
    bool is_eos = false;
    rc = apu_backend_dispatch_decode_step(
        ctx,
        initial_token,
        num_tokens,
        0.0f,
        -1,
        1,
        2,
        &output_token,
        &is_eos
    );
    printf("      apu_backend_dispatch_decode_step returned %d, output_token=%u, is_eos=%d\n",
           rc, output_token, is_eos);
    assert(rc == 0);
    assert(output_token > 0);

    printf("[6/6] Freeing context via apu_backend_free()...\n");
    apu_backend_free(ctx);
    printf("      Context cleanly freed.\n");

    remove(mock_model_path);
    remove("/tmp/c_abi_validation_model.q4nx");

    printf("========================================================\n");
    printf("  C ABI Integration Test: PASSED CLEANLY (100%%)\n");
    printf("========================================================\n");
    return 0;
}
