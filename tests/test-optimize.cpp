#include "optimize.h"

#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1); \
        } \
    } while (0)

namespace fs = std::filesystem;
using json = common_json;

static std::string write_synthetic_gguf(
        const std::string & path,
        const std::string & arch,
        uint32_t block_count,
        uint32_t head_dim,
        uint32_t n_expert,
        uint32_t n_expert_used,
        const std::string & decision_type,
        uint64_t virtual_size_bytes) {
    struct gguf_context * ctx = gguf_init_empty();
    CHECK(ctx != nullptr);

    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    gguf_set_val_u32(ctx, (arch + ".block_count").c_str(), block_count);
    gguf_set_val_u32(ctx, (arch + ".context_length").c_str(), 8192);
    if (head_dim > 0) {
        gguf_set_val_u32(ctx, (arch + ".attention.key_length").c_str(), head_dim);
        gguf_set_val_u32(ctx, (arch + ".attention.value_length").c_str(), head_dim);
    }
    if (n_expert > 0) {
        gguf_set_val_u32(ctx, (arch + ".expert_count").c_str(), n_expert);
        gguf_set_val_u32(ctx, (arch + ".expert_used_count").c_str(), n_expert_used);
    }
    if (!decision_type.empty()) {
        gguf_set_val_str(ctx, (arch + ".decision.type").c_str(), decision_type.c_str());
    }

    struct ggml_init_params ginit = {
        /* .mem_size   = */ ggml_tensor_overhead() * 16,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * gctx = ggml_init(ginit);
    CHECK(gctx != nullptr);

    struct ggml_tensor * w_attn = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, 128, 128);
    ggml_set_name(w_attn, "blk.0.attn_q.weight");
    gguf_add_tensor(ctx, w_attn);

    if (n_expert > 1) {
        const char * exp_names[3] = {
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        };
        for (int k = 0; k < 3; ++k) {
            struct ggml_tensor * w_exp = ggml_new_tensor_3d(gctx, GGML_TYPE_Q4_K, 2048, 1024, n_expert);
            ggml_set_name(w_exp, exp_names[k]);
            gguf_add_tensor(ctx, w_exp);
        }
    }

    const bool ok = gguf_write_to_file(ctx, path.c_str(), true);
    CHECK(ok);

    ggml_free(gctx);
    gguf_free(ctx);

    // Sparse-extend to simulate realistic GGUF file sizes (> 50 MiB gate) without disk allocation
    std::error_code ec;
    fs::resize_file(path, virtual_size_bytes, ec);
    CHECK(!ec);
    return path;
}

static common_model_profile make_dummy_cand(
        const std::string & name,
        const std::string & arch,
        common_model_category cat,
        uint64_t bytes,
        common_decision_type dec = COMMON_DECISION_TYPE_NONE) {
    common_model_profile p;
    p.path              = "/fake/" + name + ".gguf";
    p.name              = name + ".gguf";
    p.arch              = arch;
    p.category          = cat;
    p.decision_type     = dec;
    p.decision_is_encoder = (arch == "modern-bert");
    p.file_size_bytes   = bytes;
    p.n_layers          = 32;
    p.head_dim_k        = 128;
    p.head_dim_v        = 128;
    p.n_ctx_train       = 32768;
    p.valid             = true;
    return p;
}

int main() {
    printf("test-optimize: running unit tests...\n");

    // 1. Missing/invalid file probe check
    {
        const common_model_profile bad = common_probe_model_profile("/tmp/definitely_missing_model_xyz.gguf");
        CHECK(!bad.valid);
        CHECK(!bad.skip_reason.empty());
    }

    // 2. Diverse selection test (<= 3 per category, max arch diversity + fits/exceeds VRAM spread)
    {
        const size_t vram_16g = 16ULL << 30;
        std::vector<common_model_profile> cands = {
            make_dummy_cand("m1", "qwen35moe",  COMMON_MODEL_CATEGORY_MOE, 10ULL << 30),
            make_dummy_cand("m2", "qwen35moe",  COMMON_MODEL_CATEGORY_MOE, 15ULL << 30),
            make_dummy_cand("m3", "qwen3moe",   COMMON_MODEL_CATEGORY_MOE, 17ULL << 30),
            make_dummy_cand("m4", "k2-horizon", COMMON_MODEL_CATEGORY_MOE, 20ULL << 30),
            make_dummy_cand("d1", "qwen35",     COMMON_MODEL_CATEGORY_DENSE, 1ULL << 30),
            make_dummy_cand("d2", "lfm3",       COMMON_MODEL_CATEGORY_DENSE, 2ULL << 30),
            make_dummy_cand("d3", "llama",      COMMON_MODEL_CATEGORY_DENSE, 12ULL << 30),
            make_dummy_cand("d4", "llama",      COMMON_MODEL_CATEGORY_DENSE, 19ULL << 30),
            make_dummy_cand("c1", "modern-bert", COMMON_MODEL_CATEGORY_DECISION, 256ULL << 20, COMMON_DECISION_TYPE_GLIDE_ENCODER),
            make_dummy_cand("c2", "qwen35",      COMMON_MODEL_CATEGORY_DECISION, 3ULL << 30,   COMMON_DECISION_TYPE_PPLX_DECIDER),
            make_dummy_cand("c3", "qwen35",      COMMON_MODEL_CATEGORY_DECISION, 15ULL << 30,  COMMON_DECISION_TYPE_PPLX_DECIDER),
            make_dummy_cand("c4", "gemma4",      COMMON_MODEL_CATEGORY_DECISION, 12ULL << 30,  COMMON_DECISION_TYPE_LFM3_D1),
        };

        const auto sel = common_select_optimization_models(cands, vram_16g, 3);
        int n_moe = 0, n_dense = 0, n_dec = 0;
        std::set<std::string> moe_arches, dec_arches;
        bool dense_has_exceed = false;
        for (const auto & s : sel) {
            if (s.category == COMMON_MODEL_CATEGORY_MOE) {
                n_moe++;
                moe_arches.insert(s.arch);
            } else if (s.category == COMMON_MODEL_CATEGORY_DENSE) {
                n_dense++;
                if (s.file_size_bytes > vram_16g) dense_has_exceed = true;
            } else if (s.category == COMMON_MODEL_CATEGORY_DECISION) {
                n_dec++;
                dec_arches.insert(s.arch);
            }
        }
        CHECK(n_moe == 3 && moe_arches.size() == 3);
        CHECK(n_dense == 3 && dense_has_exceed);
        CHECK(n_dec == 3 && dec_arches.size() == 3);
        printf("  [PASS] diverse model selection (3 distinct MoE, 3 spread Dense, 3 distinct Decision)\n");
    }

    // 3. Synthetic GGUF probing + universal JSON generation + adaptive loader
    const std::string tmp_dir = "/tmp/llama_opt_unit_test";
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    fs::create_directories(tmp_dir, ec);
    CHECK(!ec);

    const std::string p_moe      = write_synthetic_gguf(tmp_dir + "/test-moe.gguf",      "qwen35moe",   40, 256, 256, 8, "",         19ULL << 30);
    const std::string p_moe_near = write_synthetic_gguf(tmp_dir + "/test-moe-near.gguf", "qwen3moe",    48, 128, 128, 8, "",         17ULL << 30);
    const std::string p_dense    = write_synthetic_gguf(tmp_dir + "/test-dense.gguf",    "llama",       28, 128,   0, 0, "",          2ULL << 30);
    const std::string p_dense_lg = write_synthetic_gguf(tmp_dir + "/test-dense-lg.gguf", "llama",       52, 128,   0, 0, "",         19ULL << 30);
    const std::string p_laya     = write_synthetic_gguf(tmp_dir + "/test-laya.gguf",     "modern-bert", 24,   0,   0, 0, "laya",    200ULL << 20);

    const common_model_profile pm = common_probe_model_profile(p_moe);
    const common_model_profile pd = common_probe_model_profile(p_dense);
    const common_model_profile pl = common_probe_model_profile(p_laya);
    CHECK(pm.valid && pm.category == COMMON_MODEL_CATEGORY_MOE && pm.n_expert == 256 && pm.expert_bytes_total > 0 && pm.min_expert_slice_bytes > 0);
    CHECK(pd.valid && pd.category == COMMON_MODEL_CATEGORY_DENSE && pd.head_dim_k == 128);
    CHECK(pl.valid && pl.category == COMMON_MODEL_CATEGORY_DECISION && pl.decision_is_encoder);
    printf("  [PASS] synthetic GGUF probing (MoE, Dense, Encoder Decision)\n");

    common_opt_model_summary s_moe;
    s_moe.profile = pm;
    s_moe.baseline.ok = s_moe.baseline.coherent = true;
    s_moe.baseline.score = 30.0;
    s_moe.best.ok = s_moe.best.coherent = true;
    s_moe.best.score = 70.0;
    s_moe.best_round = 1;
    s_moe.best_config.moe_backend = "upstream";
    s_moe.best_config.moe_cache_cpu_share = 0.5f;
    s_moe.best_config.flash_attn = "on";
    s_moe.best_config.kv_type_k = "q4_0";
    s_moe.best_config.kv_type_v = "q4_0";
    s_moe.best_config.n_batch = 2048;
    s_moe.best_config.n_ubatch = 1024;
    s_moe.best_config.kv_unified = true;
    s_moe.best_config.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"] = "moe";

    const json universal = common_derive_universal_settings({ s_moe }, "Test GPU", 16144ULL << 20, 12);
    CHECK(universal.contains("profiles"));
    CHECK(universal.at("profiles").contains("moe.near_vram"));
    CHECK(universal.at("profiles").contains("moe.exceeds_vram"));
    CHECK(universal.at("profiles").contains("dense.small"));
    CHECK(universal.at("profiles").contains("dense.large"));
    CHECK(universal.at("profiles").contains("decision.encoder"));
    CHECK(universal.contains("head_dim_rules"));
    // Without dense.large examples, conservative q4_0 + prefetch defaults must be preserved (never inherit dense.small f16)
    CHECK(universal.at("profiles").at("dense.large").value("kv_type_k", "") == "q4_0");

    // Verify no tested model filenames leak into universal profiles
    const std::string profiles_str = universal.at("profiles").dump();
    CHECK(profiles_str.find("test-moe.gguf") == std::string::npos);

    const std::string settings_path = tmp_dir + "/optimized-settings.json";
    {
        std::ofstream sf(settings_path);
        sf << universal.dump(2);
    }

    // Apply onto MoE exceeding VRAM (including pre-padded empty tensor_buft_overrides like common_params_parse produces)
    {
        common_params params;
        params.model.path = p_moe;
        params.n_ctx = 4096;
        params.tensor_buft_overrides.resize(llama_max_tensor_buft_overrides(), { nullptr, nullptr });
        params.optimize_settings_path = settings_path;
        CHECK(common_optimize_settings_apply(params));
        CHECK(!params.tensor_buft_overrides.empty());
        CHECK(params.tensor_buft_overrides[0].pattern != nullptr);
        CHECK(params.moe_cache_size > 0);
        CHECK(params.cache_type_k == GGML_TYPE_Q4_0);
        CHECK(params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED);
        // Idempotent second call
        CHECK(!common_optimize_settings_apply(params));
    }

    // Apply onto near-VRAM MoE (~1.09x VRAM -> Strategy A default moe_cache_backend=none)
    {
        common_params params;
        params.model.path = p_moe_near;
        params.n_ctx = 4096;
        params.optimize_settings_path = settings_path;
        CHECK(common_optimize_settings_apply(params));
        CHECK(params.moe_cache_size == 0);
        CHECK(params.fit_params);
    }

    // Apply onto Dense small
    {
        common_params params;
        params.model.path = p_dense;
        params.optimize_settings_path = settings_path;
        CHECK(common_optimize_settings_apply(params));
        CHECK(params.tensor_buft_overrides.empty());
        CHECK(params.moe_cache_size == 0);
        CHECK(params.cache_type_k == GGML_TYPE_Q8_0);
    }

    // High-context KV step-down guardrail (512k context on dense.small steps Q8_0 down to Q4_0)
    {
        common_params params;
        params.model.path = p_dense;
        params.n_ctx = 524288;
        params.optimize_settings_path = settings_path;
        CHECK(common_optimize_settings_apply(params));
        CHECK(params.cache_type_k == GGML_TYPE_Q4_0);
        CHECK(params.cache_type_v == GGML_TYPE_Q4_0);
    }

    // Oversized dense/decision relaxes router global n_gpu_layers=99 to -1 while preserving explicit lower NGL
    {
        common_params p_ngl99;
        p_ngl99.model.path = p_dense_lg;
        p_ngl99.n_gpu_layers = 99;
        p_ngl99.optimize_settings_path = settings_path;
        CHECK(common_optimize_settings_apply(p_ngl99));
        CHECK(p_ngl99.n_gpu_layers == -1);

        common_params p_ngl28;
        p_ngl28.model.path = p_dense_lg;
        p_ngl28.n_gpu_layers = 28;
        p_ngl28.optimize_settings_path = settings_path;
        CHECK(common_optimize_settings_apply(p_ngl28));
        CHECK(p_ngl28.n_gpu_layers == 28);
    }

    // --optimize-settings applies JSON env vars while preserving explicit --no-kv-quant FP16 override
    {
        json tweaked = universal;
        tweaked["profiles"]["dense.small"]["env_vars"]["GGML_CUDA_VOLTA_FORCE_MMQ"] = "1";
        tweaked["profiles"]["dense.small"]["env_vars"]["GGML_CUDA_PREFETCH_WEIGHTS"] = "0";
        const std::string tweaked_path = tmp_dir + "/tweaked.json";
        {
            std::ofstream tf(tweaked_path);
            tf << tweaked.dump(2);
        }
        common_params params;
        std::vector<std::string> args = {
            "llama-server",
            "--no-kv-quant",
            "--optimize-settings", tweaked_path,
        };
        std::vector<char *> argv;
        for (auto & s : args) {
            argv.push_back(s.data());
        }
        CHECK(common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER));
        params.model.path = p_dense;
        CHECK(common_optimize_settings_apply(params));
        CHECK(params.optimize_settings_applied);
        CHECK(params.cache_type_k == GGML_TYPE_F16 && params.cache_type_v == GGML_TYPE_F16);
        const char * mmq    = std::getenv("GGML_CUDA_VOLTA_FORCE_MMQ");
        const char * pref   = std::getenv("GGML_CUDA_PREFETCH_WEIGHTS");
        CHECK(mmq    != nullptr && std::strcmp(mmq,    "1") == 0);
        CHECK(pref   != nullptr && std::strcmp(pref,   "0") == 0);
    }

    // Apply onto Decision encoder with existing 7999 batch preset preserved
    {
        common_params params;
        params.model.path = p_laya;
        params.n_ctx = 7999;
        params.n_batch = 7999;
        params.n_ubatch = 7999;
        params.optimize_settings_path = settings_path;
        CHECK(common_optimize_settings_apply(params));
        CHECK(params.n_batch == 7999 && params.n_ubatch == 7999);
    }

    // Missing JSON graceful fallback
    {
        common_params params;
        params.model.path = p_dense;
        params.optimize_settings_path = tmp_dir + "/does_not_exist.json";
        CHECK(!common_optimize_settings_apply(params));
    }

    // Verify arg parser registers --optimization-test and --optimize-settings cleanly
    {
        common_params params;
        std::vector<std::string> args = {
            "llama-server",
            "--optimize-settings", settings_path,
            "--optimization-test",
        };
        std::vector<char *> argv;
        for (auto & s : args) {
            argv.push_back(s.data());
        }
        CHECK(common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER));
        CHECK(params.optimization_test);
        CHECK(params.optimize_settings_path == settings_path);
    }

    fs::remove_all(tmp_dir, ec);
    printf("test-optimize: ALL TESTS PASSED\n");
    return 0;
}
