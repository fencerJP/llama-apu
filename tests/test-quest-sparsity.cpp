#include "llama.h"
#include "../src/llama-kv-cache-quest.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

static void test_quest_bounds_and_scoring() {
    std::cout << "[Test 1] Quest Bounding Box and Scoring Math\n";

    const uint32_t n_layers   = 2;
    const uint32_t n_heads_kv = 2;
    const uint32_t head_dim   = 4;

    llama_quest_params qparams;
    qparams.sparsity  = 0.5f; // 50% sparsity
    qparams.min_pages = 2;
    qparams.page_size = 4;    // 4 tokens per page

    llama_quest_tracker tracker(n_layers, n_heads_kv, head_dim, qparams);
    assert(tracker.is_enabled());

    // Populate 4 pages (16 tokens) for layer 0, head 0
    // Page 0 (tokens 0..3): all 1.0f
    for (uint32_t t = 0; t < 4; ++t) {
        float k[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        tracker.update_token(0, 0, t, k);
    }
    // Page 1 (tokens 4..7): all 0.1f (low importance)
    for (uint32_t t = 4; t < 8; ++t) {
        float k[4] = {0.1f, 0.1f, 0.1f, 0.1f};
        tracker.update_token(0, 0, t, k);
    }
    // Page 2 (tokens 8..11): all 5.0f (high importance)
    for (uint32_t t = 8; t < 12; ++t) {
        float k[4] = {5.0f, 5.0f, 5.0f, 5.0f};
        tracker.update_token(0, 0, t, k);
    }
    // Page 3 (tokens 12..15): all 2.0f (recent page)
    for (uint32_t t = 12; t < 16; ++t) {
        float k[4] = {2.0f, 2.0f, 2.0f, 2.0f};
        tracker.update_token(0, 0, t, k);
    }

    // Query vector: [1.0, 1.0, 1.0, 1.0] for 2 query heads
    float q[8] = {1.0f, 1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<bool> active = tracker.compute_active_pages(0, 16, q, 2, 2);
    assert(active.size() == 4);

    // Page 0 must be active (sink token)
    assert(active[0] == true);
    // Page 3 and Page 2 must be active (recent sliding window)
    assert(active[3] == true);
    assert(active[2] == true);

    std::cout << "  Passed: Bounding box updates and active page selection verified.\n";
}

static void test_quest_mask_application() {
    std::cout << "[Test 2] Quest Sparsity Mask Application\n";

    llama_quest_params qparams;
    qparams.sparsity  = 0.5f;
    qparams.min_pages = 2;
    qparams.page_size = 4;

    llama_quest_tracker tracker(1, 1, 4, qparams);

    // 16 elements in KQ mask initialized to 0.0f
    std::vector<float> mask(16, 0.0f);
    std::vector<bool> active = {true, false, true, true}; // Page 1 pruned

    tracker.apply_mask(mask.data(), 16, active, -INFINITY);

    // Page 0 (tokens 0..3) must remain 0.0f
    for (int i = 0; i < 4; ++i) {
        assert(mask[i] == 0.0f);
    }

    // Page 1 (tokens 4..7) must be -INFINITY (pruned)
    for (int i = 4; i < 8; ++i) {
        assert(std::isinf(mask[i]) && mask[i] < 0.0f);
    }

    // Page 2 and 3 must remain 0.0f
    for (int i = 8; i < 16; ++i) {
        assert(mask[i] == 0.0f);
    }

    std::cout << "  Passed: Sparsity mask correctly applied with -inf on pruned pages.\n";
}

static void test_quest_disabled_passthrough() {
    std::cout << "[Test 3] Quest Disabled Passthrough Invariant\n";

    llama_quest_params qparams;
    qparams.sparsity = 0.0f; // Disabled

    llama_quest_tracker tracker(1, 1, 4, qparams);
    assert(!tracker.is_enabled());

    std::vector<bool> active = tracker.compute_active_pages(0, 64, nullptr, 0, 0);
    assert(active.size() == 4);
    for (bool b : active) {
        assert(b);
    }

    std::cout << "  Passed: 100% full dense retention preserved when sparsity is disabled.\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Running Quest KV Sparsity Unit Tests  \n";
    std::cout << "========================================\n";

    test_quest_bounds_and_scoring();
    test_quest_mask_application();
    test_quest_disabled_passthrough();

    std::cout << "\nAll Quest KV Sparsity Tests PASSED successfully!\n";
    return 0;
}
