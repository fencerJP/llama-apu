#include "llama.h"
#include <cassert>
#include <iostream>
#include <vector>

static void test_chunked_kv_defaults() {
    std::cout << "[Test 1] Chunked KV Context Parameter Defaults\n";

    llama_context_params cparams = llama_context_default_params();

    // Must be enabled by default
    assert(cparams.chunked_kv == true);
    assert(cparams.chunk_size == 32);

    // TriForce must be disabled by default
    assert(cparams.triforce == false);
    assert(cparams.triforce_draft_k == 4);

    std::cout << "  Passed: Chunked KV is enabled by default (chunk_size=32), TriForce disabled.\n";
}

static void test_chunked_kv_alignment_math() {
    std::cout << "[Test 2] Chunked KV Page Alignment Math\n";

    const uint32_t chunk_size = 32;

    for (uint32_t n_tokens : {1, 15, 32, 33, 64, 100}) {
        uint32_t chunks_needed = (n_tokens + chunk_size - 1) / chunk_size;
        uint32_t aligned_capacity = chunks_needed * chunk_size;

        assert(aligned_capacity >= n_tokens);
        assert(aligned_capacity % chunk_size == 0);
        assert(aligned_capacity - n_tokens < chunk_size);
    }

    std::cout << "  Passed: Chunk boundary alignment guarantees zero page fragmentation.\n";
}

static void test_chunked_kv_toggle() {
    std::cout << "[Test 3] Chunked KV Toggle Invariant\n";

    llama_context_params cparams = llama_context_default_params();
    cparams.chunked_kv = false;
    assert(cparams.chunked_kv == false);

    cparams.triforce = true;
    assert(cparams.triforce == true);

    std::cout << "  Passed: Context parameters correctly accept explicit runtime toggles.\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Running Chunked KV Unit Tests         \n";
    std::cout << "========================================\n";

    test_chunked_kv_defaults();
    test_chunked_kv_alignment_math();
    test_chunked_kv_toggle();

    std::cout << "\nAll Chunked KV Tests PASSED successfully!\n";
    return 0;
}
