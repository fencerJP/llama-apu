#pragma once

#include "llama.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

//
// Quest: Query-Aware Page-Level KV Cache Sparsity
//
// Maintains per-page min/max key bounding vectors to estimate upper-bound
// attention scores for fast dynamic page skipping during generation.
//

struct llama_quest_params {
    float    sparsity  = 0.0f; // Sparsity ratio in [0.0, 1.0) (0.0 = disabled)
    uint32_t min_pages = 16;   // Minimum number of pages to retain
    uint32_t page_size = 16;   // Tokens per page
};

struct llama_quest_page_bounds {
    std::vector<float> k_min;
    std::vector<float> k_max;
};

class llama_quest_tracker {
public:
    llama_quest_tracker() = default;
    llama_quest_tracker(uint32_t n_layers, uint32_t n_heads_kv, uint32_t head_dim, llama_quest_params params);

    void reset();

    // Update or insert key vector for a given layer, head, token position
    void update_token(uint32_t layer, uint32_t head_kv, uint32_t token_pos, const float * k_vec);

    // Compute active pages for a given sequence length and query representation
    // Returns a vector of booleans of length n_pages (true = active, false = pruned)
    std::vector<bool> compute_active_pages(
        uint32_t layer,
        uint32_t n_kv,
        const float * q_vec,
        uint32_t n_heads_q,
        uint32_t n_heads_kv
    ) const;

    // Mask out pruned pages in KQ mask
    template<typename T>
    void apply_mask(
        T * kq_mask,
        int64_t n_kv,
        const std::vector<bool> & active_pages,
        T mask_drop
    ) const {
        if (active_pages.empty() || !is_enabled()) {
            return;
        }

        const uint32_t n_pages = (n_kv + params.page_size - 1) / params.page_size;
        for (uint32_t p = 0; p < n_pages && p < active_pages.size(); ++p) {
            if (!active_pages[p]) {
                uint32_t t_start = p * params.page_size;
                uint32_t t_end = std::min((uint32_t)n_kv, t_start + params.page_size);
                for (uint32_t t = t_start; t < t_end; ++t) {
                    kq_mask[t] = mask_drop;
                }
            }
        }
    }

    bool is_enabled() const {
        return params.sparsity > 0.0f;
    }

    const llama_quest_params & get_params() const {
        return params;
    }

private:
    uint32_t n_layers   = 0;
    uint32_t n_heads_kv = 0;
    uint32_t head_dim   = 0;
    llama_quest_params params;

    // [n_layers][n_heads_kv][n_pages]
    std::vector<std::vector<std::vector<llama_quest_page_bounds>>> bounds;
};
