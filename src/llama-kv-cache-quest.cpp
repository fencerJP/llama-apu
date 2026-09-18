// SPDX-License-Identifier: Apache-2.0

#include "llama-kv-cache-quest.h"
#include <numeric>
#include <limits>

llama_quest_tracker::llama_quest_tracker(
    uint32_t n_layers,
    uint32_t n_heads_kv,
    uint32_t head_dim,
    llama_quest_params params
) : n_layers(n_layers),
    n_heads_kv(n_heads_kv),
    head_dim(head_dim),
    params(params) {
    if (params.page_size == 0) {
        this->params.page_size = 16;
    }
    reset();
}

void llama_quest_tracker::reset() {
    bounds.clear();
    if (n_layers > 0 && n_heads_kv > 0) {
        bounds.resize(n_layers);
        for (uint32_t il = 0; il < n_layers; ++il) {
            bounds[il].resize(n_heads_kv);
        }
    }
}

void llama_quest_tracker::update_token(
    uint32_t layer,
    uint32_t head_kv,
    uint32_t token_pos,
    const float * k_vec
) {
    if (!is_enabled() || !k_vec) {
        return;
    }

    if (layer >= bounds.size() || head_kv >= bounds[layer].size()) {
        return;
    }

    const uint32_t page_idx = token_pos / params.page_size;
    auto & head_pages = bounds[layer][head_kv];

    if (page_idx >= head_pages.size()) {
        head_pages.resize(page_idx + 1);
    }

    auto & page = head_pages[page_idx];
    if (page.k_min.empty() || page.k_min.size() != head_dim) {
        page.k_min.assign(k_vec, k_vec + head_dim);
        page.k_max.assign(k_vec, k_vec + head_dim);
    } else {
        for (uint32_t d = 0; d < head_dim; ++d) {
            if (k_vec[d] < page.k_min[d]) page.k_min[d] = k_vec[d];
            if (k_vec[d] > page.k_max[d]) page.k_max[d] = k_vec[d];
        }
    }
}

std::vector<bool> llama_quest_tracker::compute_active_pages(
    uint32_t layer,
    uint32_t n_kv,
    const float * q_vec,
    uint32_t n_heads_q,
    uint32_t n_heads_kv
) const {
    if (n_kv == 0 || params.page_size == 0) {
        return {};
    }

    const uint32_t n_pages = (n_kv + params.page_size - 1) / params.page_size;
    std::vector<bool> active(n_pages, true);

    if (!is_enabled() || n_pages <= params.min_pages) {
        return active;
    }

    // Number of pages to retain
    uint32_t keep_k = std::max(params.min_pages, (uint32_t)((1.0f - params.sparsity) * (float)n_pages));
    if (keep_k >= n_pages) {
        return active;
    }

    std::fill(active.begin(), active.end(), false);

    // Attention sink protection: always retain the initial page (page 0)
    active[0] = true;

    // Sliding window recency protection: always retain the last 2 pages
    if (n_pages > 1) {
        active[n_pages - 1] = true;
    }
    if (n_pages > 2) {
        active[n_pages - 2] = true;
    }

    // Count already protected pages
    uint32_t retained = 0;
    for (uint32_t p = 0; p < n_pages; ++p) {
        if (active[p]) retained++;
    }

    if (retained >= keep_k) {
        return active;
    }

    uint32_t budget_left = keep_k - retained;

    // Score intermediate candidate pages: p in [1, n_pages - (n_pages > 2 ? 2 : 1)]
    uint32_t end_candidate = (n_pages > 2) ? (n_pages - 2) : 1;
    std::vector<std::pair<float, uint32_t>> candidates; // (score, page_idx)

    bool has_bounds = (layer < bounds.size() && !bounds[layer].empty());

    for (uint32_t p = 1; p < end_candidate; ++p) {
        float total_page_score = 0.0f;

        if (has_bounds && q_vec && n_heads_q > 0 && n_heads_kv > 0) {
            const uint32_t gqa_ratio = n_heads_q / n_heads_kv;
            for (uint32_t h_kv = 0; h_kv < n_heads_kv && h_kv < bounds[layer].size(); ++h_kv) {
                if (p >= bounds[layer][h_kv].size()) continue;
                const auto & pb = bounds[layer][h_kv][p];
                if (pb.k_min.size() != head_dim || pb.k_max.size() != head_dim) continue;

                // Match with corresponding query heads
                for (uint32_t g = 0; g < gqa_ratio; ++g) {
                    uint32_t h_q = h_kv * gqa_ratio + g;
                    if (h_q >= n_heads_q) break;
                    const float * qh = q_vec + h_q * head_dim;

                    float head_score = 0.0f;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        float v1 = qh[d] * pb.k_min[d];
                        float v2 = qh[d] * pb.k_max[d];
                        head_score += std::max(v1, v2);
                    }
                    total_page_score += head_score;
                }
            }
        } else {
            // Default heuristic score: bias towards more recent pages
            total_page_score = (float)p;
        }

        candidates.push_back({total_page_score, p});
    }

    // Sort descending by score
    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    for (uint32_t i = 0; i < budget_left && i < candidates.size(); ++i) {
        active[candidates[i].second] = true;
    }

    return active;
}
