#pragma once

#include "common.h"
#include "json.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum common_model_category {
    COMMON_MODEL_CATEGORY_UNKNOWN = 0,
    COMMON_MODEL_CATEGORY_DENSE,
    COMMON_MODEL_CATEGORY_MOE,
    COMMON_MODEL_CATEGORY_DECISION,
};

const char * common_model_category_name(common_model_category category);

struct common_model_profile {
    std::string path;
    std::string name;
    std::string arch;
    common_model_category category      = COMMON_MODEL_CATEGORY_UNKNOWN;
    common_decision_type  decision_type = COMMON_DECISION_TYPE_NONE;
    bool     decision_is_encoder        = false;
    uint64_t file_size_bytes            = 0;
    uint32_t n_layers                   = 0;
    uint32_t n_expert                   = 0;
    uint32_t n_expert_used              = 0;
    uint32_t head_dim_k                 = 0;
    uint32_t head_dim_v                 = 0;
    uint32_t n_head_kv                  = 0;
    uint32_t full_attn_interval         = 0;
    uint32_t n_ctx_train                = 0;
    uint64_t expert_bytes_total         = 0;
    uint64_t non_expert_bytes           = 0;
    uint64_t min_expert_slice_bytes     = 0;
    uint64_t avg_expert_slice_bytes     = 0;
    bool     has_ple_engram             = false;
    bool     is_hybrid_recurrent        = false;
    bool     valid                      = false;
    std::string skip_reason;
};

struct common_opt_trial_config {
    std::string label;
    std::string hypothesis;
    std::string model_path;
    common_model_category category = COMMON_MODEL_CATEGORY_UNKNOWN;
    bool        decision_is_encoder = false;

    int32_t     n_ctx           = 2048;
    int32_t     n_batch         = 2048;
    int32_t     n_ubatch        = 512;
    int32_t     n_gpu_layers    = -1;
    int32_t     n_threads       = -1;
    bool        cpu_moe         = false;
    bool        fit_params      = true;
    bool        kv_unified      = false;
    bool        no_op_offload   = false;
    bool        use_mmap        = true;

    std::string flash_attn      = "auto";
    std::string kv_type_k       = "f16";
    std::string kv_type_v       = "f16";

    // "none", "upstream", or "custom"
    std::string moe_backend         = "none";
    size_t      moe_cache_mib       = 0;
    float       moe_cache_cpu_share = 0.0f;

    std::map<std::string, std::string> env_vars;
};

struct common_opt_trial_result {
    bool        ok              = false;
    bool        coherent        = false;
    int         exit_code       = -1;
    double      load_ms         = 0.0;
    double      pp_tps          = 0.0;
    double      tg_tps          = 0.0;
    double      pp_warm_tps     = 0.0;
    double      tg_warm_tps     = 0.0;
    double      eval_tps        = 0.0;
    double      eval_ms         = 0.0;
    double      score           = 0.0;
    size_t      vram_used_mib   = 0;
    size_t      vram_free_mib   = 0;
    std::string cache_summary;
    std::string error;
};

struct common_opt_model_summary {
    common_model_profile    profile;
    common_opt_trial_result baseline;
    common_opt_trial_result best;
    common_opt_trial_config best_config;
    int                     best_round  = 0;
    int                     rounds_run  = 0;
    std::string             winning_strategy;
};

// Probe a GGUF file header without allocating weights to inspect architecture, size, and category.
common_model_profile common_probe_model_profile(const std::string & path);

// Scan directories for GGUF files and probe all candidate models.
std::vector<common_model_profile> common_discover_models(
        const std::vector<std::string> & search_dirs,
        bool verbose = true);

// Select up to max_per_category models for each category (MoE, Dense, Decision),
// prioritizing distinct architectures and size/VRAM-fit diversity.
std::vector<common_model_profile> common_select_optimization_models(
        const std::vector<common_model_profile> & candidates,
        size_t total_vram_bytes,
        size_t max_per_category = 3);

// Synthesize universalized rules from tested example summaries into JSON schema.
common_json common_derive_universal_settings(
        const std::vector<common_opt_model_summary> & summaries,
        const std::string & gpu_name,
        size_t gpu_vram_total_bytes,
        int cpu_threads);

// Run the full --optimization-test discovery, 3-round adaptive tuning loop, and JSON generation.
int common_run_optimization_test(const common_params & params, const std::string & self_exe);

// Internal worker entrypoint invoked in an isolated subprocess per trial (--optimization-test-worker).
int common_run_optimization_worker(const std::string & trial_json_path);

// Read optimized-settings.json and dynamically adapt params to the loaded model and live VRAM.
bool common_optimize_settings_apply(common_params & params);
