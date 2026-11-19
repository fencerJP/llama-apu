#include "optimize.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using json = common_json;

const char * common_model_category_name(common_model_category category) {
    switch (category) {
        case COMMON_MODEL_CATEGORY_DENSE:    return "dense";
        case COMMON_MODEL_CATEGORY_MOE:      return "moe";
        case COMMON_MODEL_CATEGORY_DECISION: return "decision";
        case COMMON_MODEL_CATEGORY_UNKNOWN:  return "unknown";
    }
    return "unknown";
}

namespace {

struct kv_type_entry {
    const char * name;
    ggml_type    type;
};

static const kv_type_entry KV_TYPE_TABLE[] = {
    { "f32",     GGML_TYPE_F32     },
    { "f16",     GGML_TYPE_F16     },
    { "bf16",    GGML_TYPE_BF16    },
    { "f8_e4m3", GGML_TYPE_F8_E4M3 },
    { "q8_0",    GGML_TYPE_Q8_0    },
    { "q4_0",    GGML_TYPE_Q4_0    },
    { "q4_1",    GGML_TYPE_Q4_1    },
    { "iq4_nl",  GGML_TYPE_IQ4_NL  },
    { "q5_0",    GGML_TYPE_Q5_0    },
    { "q5_1",    GGML_TYPE_Q5_1    },
};

static ggml_type kv_type_from_str(const std::string & s, ggml_type fallback) {
    for (const auto & e : KV_TYPE_TABLE) {
        if (s == e.name) {
            return e.type;
        }
    }
    return fallback;
}

static llama_flash_attn_type flash_attn_from_str(const std::string & s, llama_flash_attn_type fallback) {
    if (s == "on" || s == "1" || s == "enabled") {
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (s == "off" || s == "0" || s == "disabled") {
        return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (s == "auto") {
        return LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    return fallback;
}

static const char * flash_attn_to_str(llama_flash_attn_type t) {
    switch (t) {
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return "on";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return "off";
        case LLAMA_FLASH_ATTN_TYPE_AUTO:     return "auto";
    }
    return "auto";
}

static size_t kv_type_bits_per_elem(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:     return 32;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:    return 16;
        case GGML_TYPE_F8_E4M3: return 8;
        case GGML_TYPE_Q8_0:    return 9;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_IQ4_NL:  return 5;
        default:                return 16;
    }
}

static std::string to_lower_ascii(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = char(c - 'A' + 'a');
        }
    }
    return s;
}

static bool gguf_read_u64_checked(const gguf_context * ctx, const std::string & key, uint64_t & out) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return false;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  out = gguf_get_val_u8(ctx, id); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(ctx, id); return true;
        case GGUF_TYPE_UINT64: out = gguf_get_val_u64(ctx, id); return true;
        case GGUF_TYPE_INT8: {
            const int8_t v = gguf_get_val_i8(ctx, id);
            if (v < 0) return false;
            out = uint64_t(v);
            return true;
        }
        case GGUF_TYPE_INT16: {
            const int16_t v = gguf_get_val_i16(ctx, id);
            if (v < 0) return false;
            out = uint64_t(v);
            return true;
        }
        case GGUF_TYPE_INT32: {
            const int32_t v = gguf_get_val_i32(ctx, id);
            if (v < 0) return false;
            out = uint64_t(v);
            return true;
        }
        case GGUF_TYPE_INT64: {
            const int64_t v = gguf_get_val_i64(ctx, id);
            if (v < 0) return false;
            out = uint64_t(v);
            return true;
        }
        default:
            return false;
    }
}

static bool gguf_read_head_count_kv(const gguf_context * ctx, const std::string & arch, uint32_t & out) {
    out = 0;
    const std::string key = arch + ".attention.head_count_kv";
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id >= 0) {
        if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
            uint64_t val = 0;
            if (gguf_read_u64_checked(ctx, key, val) && val > 0 && val <= 1024) {
                out = (uint32_t) val;
                return true;
            }
        } else {
            const size_t n = gguf_get_arr_n(ctx, id);
            const gguf_type et = gguf_get_arr_type(ctx, id);
            const void * data = gguf_get_arr_data(ctx, id);
            if (data && n > 0 && n <= 1024) {
                uint64_t sum = 0;
                size_t cnt = 0;
                for (size_t i = 0; i < n; ++i) {
                    int64_t v = 0;
                    switch (et) {
                        case GGUF_TYPE_UINT8:  v = static_cast<const uint8_t  *>(data)[i]; break;
                        case GGUF_TYPE_UINT16: v = static_cast<const uint16_t *>(data)[i]; break;
                        case GGUF_TYPE_UINT32: v = static_cast<const uint32_t *>(data)[i]; break;
                        case GGUF_TYPE_UINT64: v = (int64_t) static_cast<const uint64_t *>(data)[i]; break;
                        case GGUF_TYPE_INT8:   v = static_cast<const int8_t   *>(data)[i]; break;
                        case GGUF_TYPE_INT16:  v = static_cast<const int16_t  *>(data)[i]; break;
                        case GGUF_TYPE_INT32:  v = static_cast<const int32_t  *>(data)[i]; break;
                        case GGUF_TYPE_INT64:  v = static_cast<const int64_t  *>(data)[i]; break;
                        default: v = 0; break;
                    }
                    if (v > 0 && v <= 1024) {
                        sum += uint64_t(v);
                        cnt++;
                    }
                }
                if (cnt > 0) {
                    out = (uint32_t) std::clamp<uint64_t>((sum + cnt - 1) / cnt, 1ULL, 1024ULL);
                    return true;
                }
            }
        }
    }
    uint64_t fallback = 0;
    if (gguf_read_u64_checked(ctx, arch + ".attention.head_count", fallback) && fallback > 0 && fallback <= 1024) {
        out = (uint32_t) fallback;
        return true;
    }
    return false;
}

static bool has_active_buft_overrides(const std::vector<llama_model_tensor_buft_override> & overrides) {
    for (const auto & ov : overrides) {
        if (ov.pattern != nullptr || ov.buft != nullptr) {
            return true;
        }
    }
    return false;
}

static bool gguf_read_str_checked(const gguf_context * ctx, const std::string & key, std::string & out) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return false;
    }
    const char * s = gguf_get_val_str(ctx, id);
    if (!s) {
        return false;
    }
    out = s;
    return true;
}

static bool is_auxiliary_filename(const std::string & low_name) {
    return low_name.find("mmproj") != std::string::npos ||
           low_name.find("dspark") != std::string::npos ||
           low_name.find("dflash") != std::string::npos ||
           low_name.find("mtp-") != std::string::npos ||
           low_name.find("-mtp.") != std::string::npos;
}

static bool is_non_llm_arch(const std::string & low_arch, const std::string & low_name) {
    if (low_arch.find("tts") != std::string::npos ||
        low_arch.find("codec") != std::string::npos ||
        low_arch.find("whisper") != std::string::npos ||
        low_arch.find("clip") != std::string::npos ||
        low_arch.find("dac") != std::string::npos ||
        low_arch.find("breeze") != std::string::npos) {
        return true;
    }
    if (low_name.find("embed") != std::string::npos ||
        low_name.find("-tts") != std::string::npos ||
        low_name.find("breeze-tts") != std::string::npos) {
        return true;
    }
    return false;
}

static bool is_routed_expert_tensor(const char * name) {
    if (!name) {
        return false;
    }
    // match only FFN expert tables (attn_v_exps stays resident on GPU)
    return (strstr(name, "ffn_") != nullptr &&
            (strstr(name, "_exps") != nullptr || strstr(name, "_chexps") != nullptr)) ||
           strstr(name, "mlp.experts") != nullptr;
}

static void query_gpu_device_info(size_t & free_bytes, size_t & total_bytes, std::string & gpu_name, int & cpu_threads) {
    free_bytes = 0;
    total_bytes = 0;
    gpu_name.clear();
    cpu_threads = std::max(1, common_cpu_get_num_math());

    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        const enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t dev_free = 0;
            size_t dev_total = 0;
            ggml_backend_dev_memory(dev, &dev_free, &dev_total);
            if (dev_total > 0) {
                free_bytes = dev_free;
                total_bytes = dev_total;
                const char * desc = ggml_backend_dev_description(dev);
                gpu_name = desc ? desc : ggml_backend_dev_name(dev);
                break;
            }
        }
    }
    if (total_bytes == 0) {
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            size_t dev_free = 0;
            size_t dev_total = 0;
            ggml_backend_dev_memory(dev, &dev_free, &dev_total);
            if (dev_total > 0) {
                free_bytes = dev_free;
                total_bytes = dev_total;
                const char * desc = ggml_backend_dev_description(dev);
                gpu_name = desc ? desc : ggml_backend_dev_name(dev);
                break;
            }
        }
    }
    if (gpu_name.empty()) {
        gpu_name = "CPU-only";
    }
}

static bool write_text_file_atomic(const std::string & path, const std::string & content) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        f.write(content.data(), std::streamsize(content.size()));
        f.flush();
        if (!f) {
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

static bool read_text_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static json env_map_to_json(const std::map<std::string, std::string> & m) {
    json obj = json::object();
    for (const auto & [k, v] : m) {
        obj[k] = v;
    }
    return obj;
}

static std::map<std::string, std::string> json_to_env_map(const json & obj) {
    std::map<std::string, std::string> m;
    if (!obj.is_object()) {
        return m;
    }
    for (const auto & item : obj.items()) {
        if (item.value().is_string()) {
            m[item.key()] = item.value().get<std::string>();
        }
    }
    return m;
}

static json config_to_json(const common_opt_trial_config & cfg) {
    json j = json::object();
    j["label"]               = cfg.label;
    j["hypothesis"]          = cfg.hypothesis;
    j["model_path"]          = cfg.model_path;
    j["category"]            = common_model_category_name(cfg.category);
    j["decision_is_encoder"] = cfg.decision_is_encoder;
    j["n_ctx"]               = cfg.n_ctx;
    j["n_batch"]             = cfg.n_batch;
    j["n_ubatch"]            = cfg.n_ubatch;
    j["n_gpu_layers"]        = cfg.n_gpu_layers;
    j["n_threads"]           = cfg.n_threads;
    j["cpu_moe"]             = cfg.cpu_moe;
    j["fit_params"]          = cfg.fit_params;
    j["kv_unified"]          = cfg.kv_unified;
    j["no_op_offload"]       = cfg.no_op_offload;
    j["use_mmap"]            = cfg.use_mmap;
    j["flash_attn"]          = cfg.flash_attn;
    j["kv_type_k"]           = cfg.kv_type_k;
    j["kv_type_v"]           = cfg.kv_type_v;
    j["moe_backend"]         = cfg.moe_backend;
    j["moe_cache_mib"]       = (unsigned long long) cfg.moe_cache_mib;
    j["moe_cache_cpu_share"] = (double) cfg.moe_cache_cpu_share;
    j["env_vars"]            = env_map_to_json(cfg.env_vars);
    return j;
}

static common_opt_trial_config json_to_config(const json & j) {
    common_opt_trial_config cfg;
    cfg.label               = j.value("label", "");
    cfg.hypothesis          = j.value("hypothesis", "");
    cfg.model_path          = j.value("model_path", "");
    const std::string cat   = j.value("category", "unknown");
    if (cat == "moe") cfg.category = COMMON_MODEL_CATEGORY_MOE;
    else if (cat == "dense") cfg.category = COMMON_MODEL_CATEGORY_DENSE;
    else if (cat == "decision") cfg.category = COMMON_MODEL_CATEGORY_DECISION;
    cfg.decision_is_encoder = j.value("decision_is_encoder", false);
    cfg.n_ctx               = j.value("n_ctx", 2048);
    cfg.n_batch             = j.value("n_batch", 2048);
    cfg.n_ubatch            = j.value("n_ubatch", 512);
    cfg.n_gpu_layers        = j.value("n_gpu_layers", -1);
    cfg.n_threads           = j.value("n_threads", -1);
    cfg.cpu_moe             = j.value("cpu_moe", false);
    cfg.fit_params          = j.value("fit_params", true);
    cfg.kv_unified          = j.value("kv_unified", false);
    cfg.no_op_offload       = j.value("no_op_offload", false);
    cfg.use_mmap            = j.value("use_mmap", true);
    cfg.flash_attn          = j.value("flash_attn", "auto");
    cfg.kv_type_k           = j.value("kv_type_k", "f16");
    cfg.kv_type_v           = j.value("kv_type_v", "f16");
    cfg.moe_backend         = j.value("moe_backend", "none");
    cfg.moe_cache_mib       = (size_t) j.value("moe_cache_mib", 0ULL);
    cfg.moe_cache_cpu_share = (float) j.value("moe_cache_cpu_share", 0.0);
    if (j.contains("env_vars")) {
        cfg.env_vars = json_to_env_map(j.at("env_vars"));
    }
    return cfg;
}

static json result_to_json(const common_opt_trial_result & r) {
    json j = json::object();
    j["ok"]            = r.ok;
    j["coherent"]      = r.coherent;
    j["exit_code"]     = r.exit_code;
    j["load_ms"]       = r.load_ms;
    j["pp_tps"]        = r.pp_tps;
    j["tg_tps"]        = r.tg_tps;
    j["pp_warm_tps"]   = r.pp_warm_tps;
    j["tg_warm_tps"]   = r.tg_warm_tps;
    j["eval_tps"]      = r.eval_tps;
    j["eval_ms"]       = r.eval_ms;
    j["score"]         = r.score;
    j["vram_used_mib"] = (unsigned long long) r.vram_used_mib;
    j["vram_free_mib"] = (unsigned long long) r.vram_free_mib;
    j["cache_summary"] = r.cache_summary;
    j["error"]         = r.error;
    return j;
}

static common_opt_trial_result json_to_result(const json & j) {
    common_opt_trial_result r;
    r.ok            = j.value("ok", false);
    r.coherent      = j.value("coherent", false);
    r.exit_code     = j.value("exit_code", -1);
    r.load_ms       = j.value("load_ms", 0.0);
    r.pp_tps        = j.value("pp_tps", 0.0);
    r.tg_tps        = j.value("tg_tps", 0.0);
    r.pp_warm_tps   = j.value("pp_warm_tps", 0.0);
    r.tg_warm_tps   = j.value("tg_warm_tps", 0.0);
    r.eval_tps      = j.value("eval_tps", 0.0);
    r.eval_ms       = j.value("eval_ms", 0.0);
    r.score         = j.value("score", 0.0);
    r.vram_used_mib = (size_t) j.value("vram_used_mib", 0ULL);
    r.vram_free_mib = (size_t) j.value("vram_free_mib", 0ULL);
    r.cache_summary = j.value("cache_summary", "");
    r.error         = j.value("error", "");
    return r;
}

static double compute_trial_score(common_model_category cat, const common_opt_trial_result & r) {
    if (!r.ok || !r.coherent) {
        return 0.0;
    }
    switch (cat) {
        case COMMON_MODEL_CATEGORY_MOE:
            return 0.60 * r.tg_warm_tps + 0.25 * (r.pp_warm_tps / 5.0) + 0.15 * r.tg_tps;
        case COMMON_MODEL_CATEGORY_DENSE:
            return 0.70 * r.tg_tps + 0.30 * (r.pp_tps / 10.0);
        case COMMON_MODEL_CATEGORY_DECISION:
            return r.eval_tps;
        case COMMON_MODEL_CATEGORY_UNKNOWN:
            return 0.0;
    }
    return 0.0;
}

static std::string resolve_self_executable(const std::string & fallback) {
#ifndef _WIN32
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0 && n < (ssize_t) sizeof(buf)) {
        buf[n] = '\0';
        return std::string(buf);
    }
#endif
    return fallback;
}

static void apply_env_map(const std::map<std::string, std::string> & env_vars) {
    for (const auto & [k, v] : env_vars) {
#ifdef _WIN32
        _putenv_s(k.c_str(), v.c_str());
#else
        if (v.empty()) {
            unsetenv(k.c_str());
        } else {
            setenv(k.c_str(), v.c_str(), 1);
        }
#endif
    }
}

static size_t estimate_kv_cache_bytes(
        const common_model_profile & profile,
        uint32_t n_ctx,
        ggml_type tk,
        ggml_type tv) {
    if (profile.decision_is_encoder || profile.n_layers == 0) {
        return 64ULL << 20;
    }
    const uint32_t eff_ctx = n_ctx > 0 ? n_ctx : 4096U;
    const uint64_t hk = profile.head_dim_k > 0 ? profile.head_dim_k : 128ULL;
    const uint64_t hv = profile.head_dim_v > 0 ? profile.head_dim_v : hk;
    const uint64_t kv_heads = profile.n_head_kv > 0 ? profile.n_head_kv : 8ULL;
    const uint64_t kv_layers = (profile.full_attn_interval > 1)
        ? std::max<uint64_t>(1ULL, (uint64_t(profile.n_layers) + profile.full_attn_interval - 1ULL) / profile.full_attn_interval)
        : uint64_t(profile.n_layers);
    const uint64_t k_bytes = kv_layers * eff_ctx * kv_heads * hk * kv_type_bits_per_elem(tk) / 8ULL;
    const uint64_t v_bytes = kv_layers * eff_ctx * kv_heads * hv * kv_type_bits_per_elem(tv) / 8ULL;
    return size_t(std::max<uint64_t>(32ULL << 20, k_bytes + v_bytes));
}

static size_t compute_dynamic_moe_budget_mib(
        const common_model_profile & profile,
        size_t gpu_vram_avail_bytes,
        double fill_ratio,
        size_t reserve_mib,
        uint32_t n_ctx,
        ggml_type tk,
        ggml_type tv) {
    if (gpu_vram_avail_bytes == 0 || profile.expert_bytes_total == 0) {
        return 0;
    }
    const size_t kv_bytes = estimate_kv_cache_bytes(profile, n_ctx, tk, tv);
    const size_t reserve_bytes = reserve_mib << 20;
    const size_t fixed_need = size_t(profile.non_expert_bytes) + kv_bytes + reserve_bytes;
    if (gpu_vram_avail_bytes <= fixed_need) {
        return 0;
    }
    const size_t spare = gpu_vram_avail_bytes - fixed_need;
    const size_t raw_target = size_t(double(spare) * std::clamp(fill_ratio, 0.25, 0.96));
    const size_t max_useful = size_t(profile.expert_bytes_total) + (64ULL << 20);
    const size_t capped = std::min(raw_target, max_useful);
    const size_t mib = (capped >> 20) & ~size_t(63); // align down to 64 MiB
    return mib >= 512 ? mib : 0;
}

static bool verify_tensor_outputs_finite(llama_context * ctx, const llama_model * model, int32_t last_idx) {
    const float * logits = llama_get_logits_ith(ctx, last_idx);
    if (logits) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = vocab ? std::min<int32_t>(llama_vocab_n_tokens(vocab), 256) : 64;
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) {
                return false;
            }
        }
        return true;
    }
    const float * embd = llama_get_embeddings_ith(ctx, last_idx);
    if (embd) {
        const int32_t n_embd = std::min<int32_t>(llama_model_n_embd_out(model), 256);
        for (int32_t i = 0; i < n_embd; ++i) {
            if (!std::isfinite(embd[i])) {
                return false;
            }
        }
        return true;
    }
    // Check index 0 fallback for pooled models
    const float * logits0 = llama_get_logits_ith(ctx, 0);
    if (logits0) {
        return std::isfinite(logits0[0]);
    }
    const float * embd0 = llama_get_embeddings(ctx);
    if (embd0) {
        return std::isfinite(embd0[0]);
    }
    return false;
}

static llama_token pick_test_token(const llama_vocab * vocab, int step, llama_token bos) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 1) {
        return 0;
    }
    if (step == 0 && bos >= 0 && bos < n_vocab) {
        return bos;
    }
    const llama_token eos = llama_vocab_eos(vocab);
    llama_token t = (131 + step * 37) % std::max<int32_t>(2, std::min<int32_t>(n_vocab, 4096));
    if (t == eos) {
        t = (t + 7) % n_vocab;
    }
    return t;
}

static bool eval_prompt_tokens(
        llama_context * ctx,
        const llama_model * model,
        const std::vector<llama_token> & tokens,
        int32_t n_batch,
        bool is_pure_encoder,
        bool check_last_output) {
    const int32_t total = (int32_t) tokens.size();
    int32_t offset = 0;
    const llama_pos base_pos = llama_get_memory(ctx)
        ? std::max<llama_pos>(0, llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1)
        : 0;

    while (offset < total) {
        const int32_t chunk = std::min<int32_t>(n_batch, total - offset);
        common_batch batch(ctx);
        for (int32_t i = 0; i < chunk; ++i) {
            const bool need_out = check_last_output && (offset + i == total - 1);
            batch.add(tokens[size_t(offset + i)], base_pos + offset + i, 0, need_out);
        }
        const enum llama_process_type ptype = is_pure_encoder
            ? LLAMA_PROCESS_TYPE_ENCODE
            : LLAMA_PROCESS_TYPE_DECODE;
        const int rc = llama_process(ctx, ptype, batch.get());
        if (rc != 0) {
            return false;
        }
        offset += chunk;
        if (check_last_output && offset == total) {
            llama_synchronize(ctx);
            if (!verify_tensor_outputs_finite(ctx, model, chunk - 1)) {
                return false;
            }
        }
    }
    if (!check_last_output) {
        llama_synchronize(ctx);
    }
    return true;
}

static bool eval_decode_steps(
        llama_context * ctx,
        const llama_model * model,
        const llama_vocab * vocab,
        int32_t n_gen,
        double & out_ms) {
    llama_pos pos = llama_get_memory(ctx)
        ? std::max<llama_pos>(0, llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1)
        : 0;
    common_batch batch(ctx);
    llama_token tok = pick_test_token(vocab, 777, LLAMA_TOKEN_NULL);

    const auto t0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < n_gen; ++i) {
        batch.clear();
        batch.add(tok, pos++, 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            return false;
        }
        llama_synchronize(ctx);
        if (!verify_tensor_outputs_finite(ctx, model, 0)) {
            return false;
        }
        tok = pick_test_token(vocab, 900 + i, LLAMA_TOKEN_NULL);
    }
    const auto t1 = std::chrono::steady_clock::now();
    out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

static std::string extract_cache_summary_from_log(const std::string & log_text) {
    std::vector<std::string> lines;
    std::istringstream iss(log_text);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("llama_moe_cache:") != std::string::npos ||
            line.find("MoE cache size =") != std::string::npos ||
            line.find("sched_new: MoE cache") != std::string::npos ||
            line.find("MoE cache ineligible") != std::string::npos ||
            line.find("[APU MoE Chunk Loader]") != std::string::npos ||
            line.find("[moe-cache]") != std::string::npos) {
            lines.push_back(line);
        }
    }
    if (lines.empty()) {
        return "";
    }
    std::string joined;
    const size_t start = lines.size() > 3 ? lines.size() - 3 : 0;
    for (size_t i = start; i < lines.size(); ++i) {
        if (!joined.empty()) joined += " | ";
        joined += lines[i];
    }
    return joined;
}

static bool run_trial_subprocess(
        const std::string & exe_path,
        const common_opt_trial_config & cfg,
        int trial_seq,
        common_opt_trial_result & out_result) {
    const int pid_self =
#ifdef _WIN32
        0;
#else
        (int) getpid();
#endif
    const std::string base = "/tmp/llama_opt_" + std::to_string(pid_self) + "_" + std::to_string(trial_seq);
    const std::string in_path  = base + "_in.json";
    const std::string out_path = base + "_out.json";
    const std::string log_path = base + "_log.txt";

    json req = config_to_json(cfg);
    req["result_path"] = out_path;
    if (!write_text_file_atomic(in_path, req.dump(2))) {
        out_result.error = "failed to write trial request json";
        return false;
    }

#ifdef _WIN32
    std::string cmd = "\"" + exe_path + "\" --optimization-test-worker \"" + in_path + "\" > \"" + log_path + "\" 2>&1";
    int rc = std::system(cmd.c_str());
    out_result.exit_code = rc;
#else
    const pid_t child = fork();
    if (child < 0) {
        out_result.error = "fork() failed";
        std::error_code ec;
        fs::remove(in_path, ec);
        return false;
    }
    if (child == 0) {
        int fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            close(fd);
        }
        std::vector<char *> argv_exec;
        std::string arg0 = exe_path;
        std::string arg1 = "--optimization-test-worker";
        std::string arg2 = in_path;
        argv_exec.push_back(arg0.data());
        argv_exec.push_back(arg1.data());
        argv_exec.push_back(arg2.data());
        argv_exec.push_back(nullptr);
        execv(exe_path.c_str(), argv_exec.data());
        _exit(127);
    }

    int status = 0;
    const auto t0 = std::chrono::steady_clock::now();
    bool finished = false;
    while (true) {
        const pid_t w = waitpid(child, &status, WNOHANG);
        if (w == child) {
            finished = true;
            break;
        }
        if (w < 0) {
            break;
        }
        const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed_s > 180.0) {
            kill(child, SIGKILL);
            waitpid(child, &status, 0);
            out_result.error = "trial timed out after 180s";
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (finished) {
        if (WIFEXITED(status)) {
            out_result.exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            out_result.exit_code = 128 + WTERMSIG(status);
            out_result.error = "worker terminated by signal " + std::to_string(WTERMSIG(status));
        }
    }
#endif

    std::string out_text;
    if (read_text_file(out_path, out_text)) {
        json rj = json::parse_no_throw(out_text);
        if (!rj.is_discarded()) {
            const int exit_saved = out_result.exit_code;
            out_result = json_to_result(rj);
            out_result.exit_code = exit_saved;
        }
    } else if (out_result.error.empty()) {
        out_result.error = "worker failed without producing output (exit=" + std::to_string(out_result.exit_code) + ")";
    }

    std::string log_text;
    if (read_text_file(log_path, log_text)) {
        if (out_result.cache_summary.empty()) {
            out_result.cache_summary = extract_cache_summary_from_log(log_text);
        }
        if (!out_result.ok && out_result.error.empty()) {
            const size_t tail = log_text.size() > 400 ? log_text.size() - 400 : 0;
            out_result.error = log_text.substr(tail);
        }
    }

    std::error_code ec;
    fs::remove(in_path, ec);
    fs::remove(out_path, ec);
    fs::remove(log_path, ec);

    out_result.score = compute_trial_score(cfg.category, out_result);
    return out_result.ok;
}

static common_opt_trial_config make_baseline_config(
        const common_model_profile & p,
        int cpu_threads) {
    common_opt_trial_config cfg;
    cfg.label               = "Round 0 (Baseline)";
    cfg.hypothesis          = "Neutral llama.cpp defaults without optimization flags";
    cfg.model_path          = p.path;
    cfg.category            = p.category;
    cfg.decision_is_encoder = p.decision_is_encoder;
    cfg.n_ctx               = (p.n_ctx_train > 0) ? (int32_t) std::min<uint32_t>(2048U, p.n_ctx_train) : 2048;
    cfg.n_threads           = std::min(std::max(1, cpu_threads), 12);
    cfg.fit_params          = true;
    cfg.flash_attn          = "auto";
    cfg.kv_type_k           = "f16";
    cfg.kv_type_v           = "f16";
    cfg.n_batch             = p.decision_is_encoder ? cfg.n_ctx : 2048;
    cfg.n_ubatch            = p.decision_is_encoder ? cfg.n_ctx : 512;
    cfg.moe_backend         = "none";
    // Explicitly clear Volta/custom env vars for neutral baseline comparison
    cfg.env_vars["LLAMA_SM70_SKINNY"]                  = "";
    cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]          = "";
    cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"]        = "";
    cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]         = "";
    cfg.env_vars["GGML_CUDA_REGISTER_HOST"]            = "";
    cfg.env_vars["GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB"]  = "";
    return cfg;
}

static void print_trial_metrics(
        const common_model_profile & p,
        const common_opt_trial_config & cfg,
        const common_opt_trial_result & r,
        double baseline_score) {
    if (!r.ok) {
        printf("  [RESULT] FAILED (exit=%d): %s\n", r.exit_code, r.error.c_str());
        fflush(stdout);
        return;
    }
    const double delta_pct = (baseline_score > 0.0)
        ? ((r.score - baseline_score) / baseline_score) * 100.0
        : 0.0;
    printf("  [STATS ] load=%.0fms | vram=%zu MiB used (%zu MiB free) | coherent=%s\n",
           r.load_ms, r.vram_used_mib, r.vram_free_mib, r.coherent ? "PASS" : "FAIL");
    if (p.category == COMMON_MODEL_CATEGORY_MOE) {
        printf("  [PERF  ] cold: pp=%.2f tg=%.2f t/s | warm: pp=%.2f tg=%.2f t/s | score=%.2f (%+.1f%% vs baseline)\n",
               r.pp_tps, r.tg_tps, r.pp_warm_tps, r.tg_warm_tps, r.score, delta_pct);
    } else if (p.category == COMMON_MODEL_CATEGORY_DENSE) {
        printf("  [PERF  ] pp=%.2f t/s | tg=%.2f t/s | score=%.2f (%+.1f%% vs baseline)\n",
               r.pp_tps, r.tg_tps, r.score, delta_pct);
    } else {
        printf("  [PERF  ] eval=%.2f tok/s (%.2f ms/call) | score=%.2f (%+.1f%% vs baseline)\n",
               r.eval_tps, r.eval_ms, r.score, delta_pct);
    }
    if (!r.cache_summary.empty()) {
        printf("  [CACHE ] %s\n", r.cache_summary.c_str());
    }
    (void) cfg;
    fflush(stdout);
}

static common_opt_trial_config plan_round_config(
        const common_model_profile & p,
        int round_idx,
        size_t gpu_total_bytes,
        int cpu_threads,
        const std::vector<common_opt_trial_config> & hist_cfg,
        const std::vector<common_opt_trial_result> & hist_res) {
    common_opt_trial_config cfg = make_baseline_config(p, cpu_threads);
    const double size_ratio = (gpu_total_bytes > 0)
        ? double(p.file_size_bytes) / double(gpu_total_bytes)
        : 1.0;
    const bool fits_vram = size_ratio <= 0.85;
    const bool near_vram = !fits_vram && size_ratio <= 1.15;

    // Reset baseline blanks to clean Volta baseline defaults before per-round changes
    cfg.env_vars["LLAMA_SM70_SKINNY"]         = "0";
    cfg.env_vars["GGML_CUDA_REGISTER_HOST"]   = "1";
    cfg.flash_attn = "on";

    if (p.category == COMMON_MODEL_CATEGORY_MOE && near_vram) {
        const size_t est_85 = compute_dynamic_moe_budget_mib(p, gpu_total_bytes, 0.85, 768, cfg.n_ctx, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
        const size_t est_92 = compute_dynamic_moe_budget_mib(p, gpu_total_bytes, 0.92, 640, cfg.n_ctx, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);

        if (round_idx == 1) {
            cfg.label               = "Round 1 (Strategy A static split + FA on + Q4 KV + MMQ=moe)";
            cfg.hypothesis          = "Near-VRAM MoE (<=1.15x VRAM); test Strategy A static pinned-layer split without -cmoe LRU copy overhead";
            cfg.cpu_moe             = false;
            cfg.fit_params          = true;
            cfg.moe_backend         = "none";
            cfg.kv_type_k           = "q4_0";
            cfg.kv_type_v           = "q4_0";
            cfg.n_batch             = 2048;
            cfg.n_ubatch            = 1024;
            cfg.kv_unified          = true;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            return cfg;
        }
        if (round_idx == 2) {
            cfg.label               = "Round 2 (Upstream llama_moe_cache head-to-head vs Strategy A)";
            cfg.hypothesis          = "Compare -cmoe + upstream llama_moe_cache (85% spare VRAM, cpu_share=0.5) vs Round 1 Strategy A static split";
            cfg.cpu_moe             = true;
            cfg.fit_params          = false;
            cfg.moe_backend         = "upstream";
            cfg.moe_cache_mib       = est_85;
            cfg.moe_cache_cpu_share = 0.5f;
            cfg.kv_type_k           = "q4_0";
            cfg.kv_type_v           = "q4_0";
            cfg.n_batch             = 2048;
            cfg.n_ubatch            = 512;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            return cfg;
        }
        const bool r1_ok = hist_res.size() > 1 && hist_res[1].ok && hist_res[1].coherent;
        const bool r2_ok = hist_res.size() > 2 && hist_res[2].ok && hist_res[2].coherent;
        const bool r2_beats_r1 = r2_ok && (!r1_ok || hist_res[2].score > hist_res[1].score);
        if (!r2_beats_r1) {
            cfg = r1_ok ? hist_cfg[1] : cfg;
            cfg.label      = "Round 3 (Tune Strategy A: ubatch=2048 + host weight prefetch)";
            cfg.hypothesis = "Strategy A static split won; test n_ubatch=2048 and GGML_CUDA_PREFETCH_WEIGHTS=1 for cold expert transfers";
            cfg.cpu_moe    = false;
            cfg.fit_params = true;
            cfg.moe_backend = "none";
            cfg.kv_type_k  = "q4_0";
            cfg.kv_type_v  = "q4_0";
            cfg.n_batch    = 2048;
            cfg.n_ubatch   = 2048;
            cfg.kv_unified = true;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]  = "1";
            return cfg;
        }
        cfg.label               = "Round 3 (Tune Upstream llama_moe_cache: High-Fill 92% + ubatch=1024)";
        cfg.hypothesis          = "Upstream cache beat Strategy A; test 92% spare VRAM, cpu_share=0.0, ubatch=1024, kv_unified, and prefetch";
        cfg.cpu_moe             = true;
        cfg.fit_params          = false;
        cfg.moe_backend         = "upstream";
        cfg.moe_cache_mib       = est_92 > 0 ? est_92 : est_85;
        cfg.moe_cache_cpu_share = 0.0f;
        cfg.kv_type_k           = "q4_0";
        cfg.kv_type_v           = "q4_0";
        cfg.n_batch             = 2048;
        cfg.n_ubatch            = 1024;
        cfg.kv_unified          = true;
        cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
        cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
        cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]  = "1";
        return cfg;
    }

    if (p.category == COMMON_MODEL_CATEGORY_MOE && !fits_vram) {
        const size_t est_85 = compute_dynamic_moe_budget_mib(p, gpu_total_bytes, 0.85, 768, cfg.n_ctx, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
        const size_t est_92 = compute_dynamic_moe_budget_mib(p, gpu_total_bytes, 0.92, 640, cfg.n_ctx, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);

        if (round_idx == 1) {
            cfg.label               = "Round 1 (Upstream llama_moe_cache 85% spare VRAM + Q4 KV + MMQ=moe)";
            cfg.hypothesis          = "Keep experts on CPU (-cmoe), cache hot experts in 85% spare VRAM via upstream llama_moe_cache (cpu_share=0.5), Q4 KV, Volta MMQ for MoE";
            cfg.cpu_moe             = true;
            cfg.fit_params          = false;
            cfg.kv_type_k           = "q4_0";
            cfg.kv_type_v           = "q4_0";
            cfg.n_batch             = 2048;
            cfg.n_ubatch            = 512;
            cfg.moe_backend         = "upstream";
            cfg.moe_cache_mib       = est_85;
            cfg.moe_cache_cpu_share = 0.5f;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            return cfg;
        }

        const bool r1_ok = hist_res.size() > 1 && hist_res[1].ok && hist_res[1].coherent;
        if (round_idx == 2) {
#ifdef COMMON_HAS_BACKEND_MOE_CACHE
            if (p.min_expert_slice_bytes > 0) {
                const size_t min_exp_kb = std::max<size_t>(64, size_t(p.min_expert_slice_bytes / 1024ULL) / 2ULL);
                cfg.label         = "Round 2 (Custom backend --moe-cache w/ adapted min_expert_kb)";
                cfg.hypothesis    = "Head-to-head test of custom backend --moe-cache at matching budget with GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB=" + std::to_string(min_exp_kb) + " so fine-grained experts remain eligible";
                cfg.cpu_moe       = true;
                cfg.fit_params    = false;
                cfg.kv_type_k     = "q4_0";
                cfg.kv_type_v     = "q4_0";
                cfg.n_batch       = 2048;
                cfg.n_ubatch      = 512;
                cfg.moe_backend   = "custom";
                cfg.moe_cache_mib = r1_ok ? hist_cfg[1].moe_cache_mib : std::max<size_t>(512, (est_85 * 3) / 4);
                cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]         = "moe";
                cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"]       = "1";
                cfg.env_vars["GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB"] = std::to_string(min_exp_kb);
                return cfg;
            }
#endif
            if (!r1_ok) {
                cfg.label               = "Round 2 (OOM Recovery: Upstream MoE cache 68% spare VRAM)";
                cfg.hypothesis          = "Round 1 failed/OOMed; back off fill_ratio to 68% with larger 1280 MiB reserve";
                cfg.cpu_moe             = true;
                cfg.fit_params          = false;
                cfg.kv_type_k           = "q4_0";
                cfg.kv_type_v           = "q4_0";
                cfg.moe_backend         = "upstream";
                cfg.moe_cache_mib       = compute_dynamic_moe_budget_mib(p, gpu_total_bytes, 0.68, 1280, cfg.n_ctx, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
                cfg.moe_cache_cpu_share = 0.5f;
                cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"] = "moe";
                return cfg;
            }
            cfg.label               = "Round 2 (Upstream High-Fill 92% VRAM + cpu_share=0.0 + Prefetch)";
            cfg.hypothesis          = "Test GPU-only miss handling (cpu_share=0.0) with 92% spare VRAM fill and n_ubatch=1024 vs Round 1 cpu_share=0.5";
            cfg.cpu_moe             = true;
            cfg.fit_params          = false;
            cfg.kv_type_k           = "q4_0";
            cfg.kv_type_v           = "q4_0";
            cfg.n_batch             = 2048;
            cfg.n_ubatch            = 1024;
            cfg.moe_backend         = "upstream";
            cfg.moe_cache_mib       = est_92 > 0 ? est_92 : est_85;
            cfg.moe_cache_cpu_share = 0.0f;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]  = "1";
            return cfg;
        }

        // Round 3: refine whichever backend won Rounds 1-2
        const bool r2_ok = hist_res.size() > 2 && hist_res[2].ok && hist_res[2].coherent;
        const bool r2_beats_r1 = r2_ok && (!r1_ok || hist_res[2].score > hist_res[1].score);
        if (r2_beats_r1 && hist_cfg[2].moe_backend == "custom") {
            cfg = hist_cfg[2];
            cfg.label      = "Round 3 (Tune Custom --moe-cache: 92% VRAM + admit_after=1 + ubatch=1024)";
            cfg.hypothesis = "Round 2 custom backend won; maximize pool budget to 92% spare VRAM, immediate admission, and ubatch=1024";
            cfg.moe_cache_mib = est_92 > 0 ? est_92 : hist_cfg[2].moe_cache_mib;
            cfg.n_ubatch      = 1024;
            cfg.kv_unified    = true;
            cfg.env_vars["GGML_CUDA_MOE_CACHE_ADMIT_AFTER"]      = "1";
            cfg.env_vars["GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS"] = "2";
            cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]           = "1";
            return cfg;
        }

        // Refine upstream backend
        cfg.label      = "Round 3 (Tune Upstream llama_moe_cache: High-Fill + Best Share Sweep)";
        cfg.cpu_moe    = true;
        cfg.fit_params = false;
        cfg.kv_type_k  = "q4_0";
        cfg.kv_type_v  = "q4_0";
        cfg.n_batch    = 2048;
        cfg.n_ubatch   = 1024;
        cfg.kv_unified = true;
        cfg.moe_backend = "upstream";
        cfg.moe_cache_mib = (r2_ok && hist_res[2].vram_free_mib > 350) ? (est_92 > 0 ? est_92 : est_85) : est_85;
        if (r2_beats_r1 && hist_cfg[2].moe_cache_cpu_share < 0.1f) {
            cfg.moe_cache_cpu_share = 0.25f;
            cfg.hypothesis = "cpu_share=0.0 beat 0.5; test hybrid cpu_share=0.25 + 92% VRAM + kv_unified + prefetch";
        } else {
            cfg.moe_cache_cpu_share = 0.5f;
            cfg.hypothesis = "Preserve winning cpu_share=0.5 with high-fill 92% cache budget, n_ubatch=1024, kv_unified, and weight prefetch";
        }
        cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
        cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
        cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]  = "1";
        return cfg;
    }

    if (p.category == COMMON_MODEL_CATEGORY_MOE && fits_vram) {
        if (round_idx == 1) {
            cfg.label        = "Round 1 (Full GPU MoE + FA on + Q4 KV + MMQ=moe)";
            cfg.hypothesis   = "Model fits in VRAM; keep all experts resident on GPU with FlashAttention, Q4 KV, and Volta MMQ";
            cfg.n_gpu_layers = 999;
            cfg.kv_type_k    = "q4_0";
            cfg.kv_type_v    = "q4_0";
            cfg.n_ubatch     = 1024;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            return cfg;
        }
        if (round_idx == 2) {
            cfg.label        = "Round 2 (Full GPU MoE + Q8 KV + kv_unified)";
            cfg.hypothesis   = "Test higher-precision Q8_0 KV cache + Q8 Volta TC FlashAttention + kv_unified";
            cfg.n_gpu_layers = 999;
            cfg.kv_type_k    = "q8_0";
            cfg.kv_type_v    = "q8_0";
            cfg.n_ubatch     = 1024;
            cfg.kv_unified   = true;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            return cfg;
        }
        const bool r2_ok = hist_res.size() > 2 && hist_res[2].ok && hist_res[2].coherent;
        const bool r1_ok = hist_res.size() > 1 && hist_res[1].ok && hist_res[1].coherent;
        const auto & win_cfg = (r2_ok && (!r1_ok || hist_res[2].score > hist_res[1].score)) ? hist_cfg[2] : hist_cfg[1];
        cfg = win_cfg;
        cfg.label      = "Round 3 (Test cuBLAS vs Volta MMQ on winning KV/ubatch)";
        cfg.hypothesis = "Compare GGML_CUDA_VOLTA_FORCE_MMQ=0 (cuBLAS) vs moe on winning KV type";
        cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"] = "0";
        return cfg;
    }

    if (p.category == COMMON_MODEL_CATEGORY_DENSE) {
        if (round_idx == 1) {
            const bool big_dense = size_ratio > 0.65;
            cfg.label        = big_dense ? "Round 1 (Large Dense: FA + Q4 KV + Prefetch + ubatch=1024)" : "Round 1 (Dense: FA + Q8 KV + Q8_TC + ubatch=1024)";
            cfg.hypothesis   = "Enable FlashAttention, Volta Q8_TC, optimal KV quant for model footprint, and host weight prefetch";
            cfg.kv_type_k    = big_dense ? "q4_0" : "q8_0";
            cfg.kv_type_v    = big_dense ? "q4_0" : "q8_0";
            cfg.n_batch      = 2048;
            cfg.n_ubatch     = 1024;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "0";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]  = fits_vram ? "0" : "1";
            return cfg;
        }
        const bool r1_ok = hist_res.size() > 1 && hist_res[1].ok && hist_res[1].coherent;
        if (round_idx == 2) {
            if (!r1_ok) {
                cfg.label      = "Round 2 (OOM Recovery: Q4_0 KV + fit_params + ubatch=512)";
                cfg.hypothesis = "Round 1 failed/OOMed; enforce Q4 KV and smaller ubatch=512";
                cfg.kv_type_k  = "q4_0";
                cfg.kv_type_v  = "q4_0";
                cfg.n_ubatch   = 512;
                return cfg;
            }
            const bool very_small = size_ratio <= 0.45;
            cfg.label        = very_small ? "Round 2 (Test Volta MMQ=1 + F16 KV)" : "Round 2 (Test Volta MMQ=1 + Q4_0 KV)";
            cfg.hypothesis   = "Compare GGML_CUDA_VOLTA_FORCE_MMQ=1 + alternate KV precision vs Round 1 cuBLAS";
            cfg.kv_type_k    = very_small ? "f16" : "q4_0";
            cfg.kv_type_v    = very_small ? "f16" : "q4_0";
            cfg.n_batch      = 2048;
            cfg.n_ubatch     = 1024;
            cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "1";
            cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
            cfg.env_vars["GGML_CUDA_PREFETCH_WEIGHTS"]  = fits_vram ? "0" : "1";
            return cfg;
        }
        const bool r2_ok = hist_res.size() > 2 && hist_res[2].ok && hist_res[2].coherent;
        const auto & win_cfg = (r2_ok && (!r1_ok || hist_res[2].score > hist_res[1].score)) ? hist_cfg[2] : (r1_ok ? hist_cfg[1] : hist_cfg[0]);
        cfg = win_cfg;
        cfg.label      = "Round 3 (Tune batching ubatch=2048 + kv_unified on winner)";
        cfg.hypothesis = "Test full ubatch=2048 + kv_unified=true on winning kernel and KV precision";
        cfg.n_batch    = 2048;
        cfg.n_ubatch   = (fits_vram || (r1_ok && hist_res[1].vram_free_mib > 2048)) ? 2048 : 512;
        cfg.kv_unified = true;
        return cfg;
    }

    // Decision models
    if (round_idx == 1) {
        const bool tight = size_ratio > 0.65;
        cfg.label      = p.decision_is_encoder ? "Round 1 (Encoder Decision: FA + F16 + full ubatch)" : "Round 1 (Causal Decision: FA + cuBLAS + large ubatch)";
        cfg.hypothesis = "Optimize prompt classification latency using FlashAttention and large single-pass ubatch";
        cfg.kv_type_k  = (p.decision_is_encoder || !tight) ? "f16" : "q4_0";
        cfg.kv_type_v  = (p.decision_is_encoder || !tight) ? "f16" : "q4_0";
        cfg.n_batch    = p.decision_is_encoder ? cfg.n_ctx : 2048;
        cfg.n_ubatch   = p.decision_is_encoder ? cfg.n_ctx : 2048;
        cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "0";
        cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
        return cfg;
    }
    const bool r1_ok = hist_res.size() > 1 && hist_res[1].ok && hist_res[1].coherent;
    if (round_idx == 2) {
        const bool tight = size_ratio > 0.65;
        cfg.label      = "Round 2 (Decision: Test Volta MMQ=1 + Q8/Q4 KV)";
        cfg.hypothesis = "Compare Volta MMQ=1 and Q8_0 (or Q4_0 when tight) KV vs Round 1";
        cfg.kv_type_k  = p.decision_is_encoder ? "f16" : (tight ? "q4_0" : "q8_0");
        cfg.kv_type_v  = p.decision_is_encoder ? "f16" : (tight ? "q4_0" : "q8_0");
        cfg.n_batch    = p.decision_is_encoder ? cfg.n_ctx : 2048;
        cfg.n_ubatch   = p.decision_is_encoder ? cfg.n_ctx : 2048;
        cfg.env_vars["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "1";
        cfg.env_vars["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
        return cfg;
    }
    const bool r2_ok = hist_res.size() > 2 && hist_res[2].ok && hist_res[2].coherent;
    const auto & win_cfg = (r2_ok && (!r1_ok || hist_res[2].score > hist_res[1].score)) ? hist_cfg[2] : (r1_ok ? hist_cfg[1] : hist_cfg[0]);
    cfg = win_cfg;
    cfg.label      = "Round 3 (Decision: Tune ubatch + kv_unified)";
    cfg.hypothesis = p.decision_is_encoder
        ? "Verify encoder throughput with kv_unified=true and matched n_batch=n_ubatch"
        : "Test ubatch=1024 + kv_unified=true vs ubatch=2048 on winning kernel";
    if (!p.decision_is_encoder) {
        cfg.n_batch  = 2048;
        cfg.n_ubatch = 1024;
    }
    cfg.kv_unified = true;
    return cfg;
}

static std::string majority_or_best_str(
        const std::vector<const common_opt_model_summary *> & items,
        const std::function<std::string(const common_opt_model_summary &)> & getter,
        const std::string & def_val) {
    if (items.empty()) {
        return def_val;
    }
    std::map<std::string, double> weight_by_val;
    for (const auto * s : items) {
        const std::string val = getter(*s);
        const double gain = (s->baseline.score > 0.0 && s->best.score > 0.0)
            ? std::max(1.0, s->best.score / s->baseline.score)
            : 1.0;
        weight_by_val[val] += gain;
    }
    std::string best = def_val;
    double best_w = -1.0;
    for (const auto & [k, w] : weight_by_val) {
        if (w > best_w) {
            best_w = w;
            best = k;
        }
    }
    return best;
}

static int majority_or_best_int(
        const std::vector<const common_opt_model_summary *> & items,
        const std::function<int(const common_opt_model_summary &)> & getter,
        int def_val) {
    if (items.empty()) {
        return def_val;
    }
    std::map<int, double> weight_by_val;
    for (const auto * s : items) {
        const int val = getter(*s);
        const double gain = (s->baseline.score > 0.0 && s->best.score > 0.0)
            ? std::max(1.0, s->best.score / s->baseline.score)
            : 1.0;
        weight_by_val[val] += gain;
    }
    int best = def_val;
    double best_w = -1.0;
    for (const auto & [k, w] : weight_by_val) {
        if (w > best_w) {
            best_w = w;
            best = k;
        }
    }
    return best;
}

static bool majority_or_best_bool(
        const std::vector<const common_opt_model_summary *> & items,
        const std::function<bool(const common_opt_model_summary &)> & getter,
        bool def_val) {
    return majority_or_best_int(items, [&](const common_opt_model_summary & s) {
        return getter(s) ? 1 : 0;
    }, def_val ? 1 : 0) != 0;
}

static json build_universal_profile_rule(
        const std::vector<const common_opt_model_summary *> & items,
        const std::string & def_fa,
        const std::string & def_kv_k,
        const std::string & def_kv_v,
        int def_batch,
        int def_ubatch,
        bool def_kv_unified,
        bool is_moe_exceeds,
        bool is_moe_fits,
        bool is_encoder,
        const std::string & def_force_mmq,
        const std::string & def_prefetch,
        const std::string & def_moe_backend,
        double def_fill_ratio,
        int def_reserve_mib,
        double def_cpu_share,
        bool adapt_min_expert_kb,
        const std::string & derived_from) {
    json rule = json::object();
    const std::string fa = majority_or_best_str(items, [](const common_opt_model_summary & s) {
        return s.best_config.flash_attn;
    }, def_fa);
    const std::string kv_k = majority_or_best_str(items, [](const common_opt_model_summary & s) {
        return s.best_config.kv_type_k;
    }, def_kv_k);
    const std::string kv_v = majority_or_best_str(items, [](const common_opt_model_summary & s) {
        return s.best_config.kv_type_v;
    }, def_kv_v);
    const int n_batch = majority_or_best_int(items, [](const common_opt_model_summary & s) {
        return s.best_config.n_batch;
    }, def_batch);
    const int n_ubatch = majority_or_best_int(items, [](const common_opt_model_summary & s) {
        return s.best_config.n_ubatch;
    }, def_ubatch);
    const bool kv_unified = majority_or_best_bool(items, [](const common_opt_model_summary & s) {
        return s.best_config.kv_unified;
    }, def_kv_unified);
    const std::string force_mmq = majority_or_best_str(items, [&](const common_opt_model_summary & s) {
        auto it = s.best_config.env_vars.find("GGML_CUDA_VOLTA_FORCE_MMQ");
        return (it != s.best_config.env_vars.end() && !it->second.empty()) ? it->second : def_force_mmq;
    }, def_force_mmq);
    const std::string prefetch = majority_or_best_str(items, [&](const common_opt_model_summary & s) {
        auto it = s.best_config.env_vars.find("GGML_CUDA_PREFETCH_WEIGHTS");
        return (it != s.best_config.env_vars.end() && !it->second.empty()) ? it->second : def_prefetch;
    }, def_prefetch);

    rule["flash_attn"]         = fa;
    rule["kv_type_k"]          = kv_k;
    rule["kv_type_v"]          = kv_v;
    rule["n_batch"]            = n_batch;
    rule["n_ubatch"]           = is_encoder ? std::max(n_batch, n_ubatch) : n_ubatch;
    rule["kv_unified"]         = kv_unified;
    rule["encoder_match_batch"] = is_encoder;

    json env = json::object();
    env["LLAMA_SM70_SKINNY"]           = "0";
    env["GGML_CUDA_VOLTA_FORCE_MMQ"]   = force_mmq;
    env["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
    env["GGML_CUDA_REGISTER_HOST"]     = "1";
    env["GGML_CUDA_PREFETCH_WEIGHTS"]  = prefetch;
    rule["env_vars"] = env;

    if (is_moe_exceeds) {
        const std::string backend = majority_or_best_str(items, [](const common_opt_model_summary & s) {
            return s.best_config.moe_backend;
        }, def_moe_backend);
        double best_share = def_cpu_share;
        if (!items.empty()) {
            const common_opt_model_summary * top = items.front();
            for (const auto * s : items) {
                if (s->best.score > top->best.score) {
                    top = s;
                }
            }
            if (top->best_config.moe_backend == backend) {
                best_share = double(top->best_config.moe_cache_cpu_share);
            }
        }
        rule["moe_cache_backend"]             = backend;
        rule["moe_cache_vram_fill_ratio"]     = def_fill_ratio;
        rule["moe_cache_reserve_mib"]         = def_reserve_mib;
        rule["moe_cache_cpu_share"]           = best_share;
        rule["moe_cache_adapt_min_expert_kb"] = adapt_min_expert_kb;
        rule["cpu_moe"]                       = (backend != "none");
    } else if (is_moe_fits) {
        rule["moe_cache_backend"] = "none";
        rule["cpu_moe"]           = false;
    }
    rule["derived_from"] = derived_from;
    return rule;
}

static std::string select_profile_key(const common_model_profile & profile, size_t gpu_total_bytes) {
    const double ratio = gpu_total_bytes > 0
        ? double(profile.file_size_bytes) / double(gpu_total_bytes)
        : 1.0;

    if (profile.category == COMMON_MODEL_CATEGORY_DECISION) {
        if (profile.decision_is_encoder) {
            return "decision.encoder";
        }
        return ratio <= 0.65 ? "decision.causal_fits" : "decision.causal_tight";
    }
    if (profile.category == COMMON_MODEL_CATEGORY_MOE) {
        if (ratio <= 0.85) {
            return "moe.fits_vram";
        }
        if (ratio <= 1.15) {
            return "moe.near_vram";
        }
        return "moe.exceeds_vram";
    }
    if (ratio <= 0.45) {
        return "dense.small";
    }
    if (ratio <= 0.85) {
        return "dense.medium";
    }
    return "dense.large";
}

} // namespace

common_model_profile common_probe_model_profile(const std::string & path) {
    common_model_profile p;
    p.path = path;

    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
        p.skip_reason = "file does not exist or is not regular";
        return p;
    }

    const fs::path fs_path(path);
    p.name = fs_path.filename().string();
    const std::string low_name = to_lower_ascii(p.name);
    if (is_auxiliary_filename(low_name)) {
        p.skip_reason = "auxiliary projector/draft/mtp model";
        return p;
    }

    const uint64_t single_size = fs::file_size(fs_path, ec);
    if (ec || single_size < 50ULL * 1024ULL * 1024ULL) {
        p.skip_reason = "file smaller than 50 MiB threshold";
        return p;
    }

    struct gguf_init_params gparams = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context_ptr ctx0(gguf_init_from_file(path.c_str(), gparams));
    if (!ctx0) {
        p.skip_reason = "failed to parse GGUF header";
        return p;
    }

    uint64_t split_no = 0;
    if (gguf_read_u64_checked(ctx0.get(), "split.no", split_no) && split_no > 0) {
        p.skip_reason = "non-zero split shard";
        return p;
    }

    std::string gtype;
    if (gguf_read_str_checked(ctx0.get(), "general.type", gtype)) {
        const std::string low_gtype = to_lower_ascii(gtype);
        if (low_gtype == "projector" || low_gtype == "adapter" || low_gtype == "lora" || low_gtype == "imatrix") {
            p.skip_reason = "non-model GGUF type: " + gtype;
            return p;
        }
    }

    if (!gguf_read_str_checked(ctx0.get(), "general.architecture", p.arch) || p.arch.empty()) {
        p.skip_reason = "missing general.architecture";
        return p;
    }

    const common_gguf_info ginfo = common_get_gguf_info(path);
    p.decision_type = ginfo.decision_type;
    p.n_ctx_train   = ginfo.n_ctx_train;

    const std::string low_arch = to_lower_ascii(p.arch);
    const bool is_valid_decision =
        p.decision_type != COMMON_DECISION_TYPE_NONE &&
        p.decision_type != COMMON_DECISION_TYPE_UNKNOWN;
    if (!is_valid_decision && is_non_llm_arch(low_arch, low_name)) {
        p.skip_reason = "embedding/audio/tts utility model";
        return p;
    }

    uint64_t tmp_u64 = 0;
    if (gguf_read_u64_checked(ctx0.get(), p.arch + ".block_count", tmp_u64) && tmp_u64 <= 1024) {
        p.n_layers = (uint32_t) tmp_u64;
    }
    if (p.n_layers == 0) {
        p.skip_reason = "block_count is zero";
        return p;
    }
    if (gguf_read_u64_checked(ctx0.get(), p.arch + ".expert_count", tmp_u64) && tmp_u64 <= 4096) {
        p.n_expert = (uint32_t) tmp_u64;
    }
    if (gguf_read_u64_checked(ctx0.get(), p.arch + ".expert_used_count", tmp_u64) && tmp_u64 <= 4096) {
        p.n_expert_used = (uint32_t) tmp_u64;
    }

    uint64_t head_k = 0;
    uint64_t head_v = 0;
    if (gguf_read_u64_checked(ctx0.get(), p.arch + ".attention.key_length", head_k) && head_k <= 2048) {
        p.head_dim_k = (uint32_t) head_k;
    }
    if (gguf_read_u64_checked(ctx0.get(), p.arch + ".attention.value_length", head_v) && head_v <= 2048) {
        p.head_dim_v = (uint32_t) head_v;
    }
    if (p.head_dim_k == 0) {
        uint64_t embd = 0;
        uint64_t n_head = 0;
        if (gguf_read_u64_checked(ctx0.get(), p.arch + ".embedding_length", embd) &&
            gguf_read_u64_checked(ctx0.get(), p.arch + ".attention.head_count", n_head) &&
            n_head > 0 && embd >= n_head) {
            p.head_dim_k = (uint32_t) (embd / n_head);
        }
    }
    if (p.head_dim_v == 0) {
        p.head_dim_v = p.head_dim_k;
    }

    gguf_read_head_count_kv(ctx0.get(), p.arch, p.n_head_kv);
    uint64_t fai_u64 = 0;
    if (gguf_read_u64_checked(ctx0.get(), p.arch + ".full_attention_interval", fai_u64) && fai_u64 >= 1 && fai_u64 <= 64) {
        p.full_attn_interval = (uint32_t) fai_u64;
    }

    p.is_hybrid_recurrent =
        gguf_find_key(ctx0.get(), (p.arch + ".ssm.state_size").c_str()) >= 0 ||
        gguf_find_key(ctx0.get(), (p.arch + ".full_attention_interval").c_str()) >= 0 ||
        low_arch.find("lfm") != std::string::npos ||
        low_arch.find("qwen35") != std::string::npos;

    // Collect split shards if split.count > 1 (capped at 256 per security check)
    uint64_t n_split = 1;
    gguf_read_u64_checked(ctx0.get(), "split.count", n_split);
    if (n_split == 0 || n_split > 256) {
        p.skip_reason = "invalid split.count";
        return p;
    }

    std::vector<std::string> shard_paths;
    if (n_split > 1) {
        char prefix[1024];
        if (llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 0, (uint16_t) n_split) > 0) {
            for (uint16_t s = 0; s < (uint16_t) n_split; ++s) {
                char s_path[1024];
                if (llama_split_path(s_path, sizeof(s_path), prefix, s, (uint16_t) n_split) > 0) {
                    shard_paths.emplace_back(s_path);
                }
            }
        }
    }
    if (shard_paths.empty()) {
        shard_paths.push_back(path);
    }

    uint64_t sum_file_bytes = 0;
    uint64_t sum_expert_slices = 0;
    uint64_t count_expert_tensors = 0;

    for (size_t s = 0; s < shard_paths.size(); ++s) {
        const std::string & spath = shard_paths[s];
        const uint64_t sz = fs::file_size(spath, ec);
        if (ec) {
            p.skip_reason = "missing split shard: " + spath;
            return p;
        }
        if (sum_file_bytes > UINT64_MAX - sz) {
            p.skip_reason = "file size overflow";
            return p;
        }
        sum_file_bytes += sz;

        gguf_context * sc = nullptr;
        gguf_context_ptr sc_owned;
        if (s == 0) {
            sc = ctx0.get();
        } else {
            sc_owned.reset(gguf_init_from_file(spath.c_str(), gparams));
            sc = sc_owned.get();
        }
        if (!sc) {
            p.skip_reason = "failed to parse split GGUF header";
            return p;
        }

        const int64_t n_tensors = gguf_get_n_tensors(sc);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const enum ggml_type ttype = gguf_get_tensor_type(sc, i);
            if ((int) ttype < 0 || (int) ttype >= (int) GGML_TYPE_COUNT || ggml_type_name(ttype) == nullptr || ggml_blck_size(ttype) == 0) {
                p.skip_reason = "unsupported tensor type id " + std::to_string((int) ttype);
                return p;
            }
            const size_t tsize = gguf_get_tensor_size(sc, i);
            const char * tname = gguf_get_tensor_name(sc, i);
            if (tname && strstr(tname, "per_layer_token_embd") != nullptr && tsize >= (4ULL << 30)) {
                p.has_ple_engram = true;
            }
            if (is_routed_expert_tensor(tname)) {
                if (p.expert_bytes_total <= UINT64_MAX - tsize) {
                    p.expert_bytes_total += tsize;
                }
                if (p.n_expert > 1 && tsize >= p.n_expert) {
                    const uint64_t slice = tsize / p.n_expert;
                    if (p.min_expert_slice_bytes == 0 || slice < p.min_expert_slice_bytes) {
                        p.min_expert_slice_bytes = slice;
                    }
                    sum_expert_slices += slice;
                    count_expert_tensors++;
                }
            } else {
                if (p.non_expert_bytes <= UINT64_MAX - tsize) {
                    p.non_expert_bytes += tsize;
                }
            }
        }
    }

    p.file_size_bytes = sum_file_bytes;
    if (count_expert_tensors > 0) {
        p.avg_expert_slice_bytes = sum_expert_slices / count_expert_tensors;
    }

    if (is_valid_decision) {
        p.category = COMMON_MODEL_CATEGORY_DECISION;
        p.decision_is_encoder =
            low_arch == "modern-bert" ||
            p.decision_type == COMMON_DECISION_TYPE_LAYA ||
            p.decision_type == COMMON_DECISION_TYPE_GLIDE_ENCODER ||
            p.decision_type == COMMON_DECISION_TYPE_CLEF;
    } else if (p.n_expert > 1) {
        p.category = COMMON_MODEL_CATEGORY_MOE;
    } else {
        p.category = COMMON_MODEL_CATEGORY_DENSE;
    }

    p.valid = true;
    return p;
}

std::vector<common_model_profile> common_discover_models(
        const std::vector<std::string> & search_dirs,
        bool verbose) {
    std::vector<std::string> roots = search_dirs;
    if (roots.empty()) {
        roots = { "/models", "/opt/models", "." };
    }

    std::set<std::string> seen_dirs;
    std::set<std::string> seen_files;
    std::vector<std::string> gguf_paths;

    std::error_code ec;
    for (const auto & dir : roots) {
        if (dir.empty() || !fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
            continue;
        }
        const std::string canon_dir = fs::weakly_canonical(dir, ec).string();
        if (!seen_dirs.insert(canon_dir).second) {
            continue;
        }
        if (verbose) {
            printf("[optimize] Scanning directory: %s\n", dir.c_str());
        }
        fs::recursive_directory_iterator it(dir, fs::directory_options::follow_directory_symlink | fs::directory_options::skip_permission_denied, ec);
        fs::recursive_directory_iterator end;
        for (; it != end; it.increment(ec)) {
            if (ec) {
                continue;
            }
            if (it.depth() > 3) {
                it.disable_recursion_pending();
                continue;
            }
            if (!it->is_regular_file(ec)) {
                continue;
            }
            const fs::path fp = it->path();
            if (fp.extension() != ".gguf") {
                continue;
            }
            const fs::path resolved = fs::canonical(fp, ec);
            const std::string key = ec ? fp.string() : resolved.string();
            if (seen_files.insert(key).second) {
                gguf_paths.push_back(key);
            }
        }
    }

    std::sort(gguf_paths.begin(), gguf_paths.end());
    std::vector<common_model_profile> results;
    results.reserve(gguf_paths.size());
    for (const auto & fp : gguf_paths) {
        common_model_profile prof = common_probe_model_profile(fp);
        if (verbose) {
            if (prof.valid) {
                printf("  [found] %-8s | %-12s | %6.2f GiB | L=%2u | E=%3u | dk=%3u | %s\n",
                       common_model_category_name(prof.category),
                       prof.arch.c_str(),
                       double(prof.file_size_bytes) / double(1ULL << 30),
                       prof.n_layers,
                       prof.n_expert,
                       prof.head_dim_k,
                       prof.name.c_str());
            } else {
                printf("  [skip ] %s (%s)\n", fp.c_str(), prof.skip_reason.c_str());
            }
        }
        if (prof.valid) {
            results.push_back(std::move(prof));
        }
    }
    if (verbose) {
        fflush(stdout);
    }
    return results;
}

std::vector<common_model_profile> common_select_optimization_models(
        const std::vector<common_model_profile> & candidates,
        size_t total_vram_bytes,
        size_t max_per_category) {
    std::vector<common_model_profile> selected;
    const uint64_t fit_threshold = total_vram_bytes > 0
        ? uint64_t(double(total_vram_bytes) * 0.85)
        : (14ULL << 30);

    const common_model_category cats[] = {
        COMMON_MODEL_CATEGORY_MOE,
        COMMON_MODEL_CATEGORY_DENSE,
        COMMON_MODEL_CATEGORY_DECISION,
    };

    for (const auto cat : cats) {
        std::vector<const common_model_profile *> pool;
        for (const auto & c : candidates) {
            if (c.valid && c.category == cat) {
                pool.push_back(&c);
            }
        }
        if (pool.empty() || max_per_category == 0) {
            continue;
        }
        std::sort(pool.begin(), pool.end(), [](const common_model_profile * a, const common_model_profile * b) {
            return a->file_size_bytes < b->file_size_bytes;
        });

        std::vector<const common_model_profile *> chosen;
        std::set<std::string> used_paths;

        auto pick_best = [&](const std::function<double(const common_model_profile &)> & score_fn) {
            const common_model_profile * best = nullptr;
            double best_s = -1e18;
            for (const auto * c : pool) {
                if (used_paths.count(c->path)) {
                    continue;
                }
                const double s = score_fn(*c);
                if (s > best_s) {
                    best_s = s;
                    best = c;
                }
            }
            if (best) {
                chosen.push_back(best);
                used_paths.insert(best->path);
            }
        };

        // Pick 1: prefer smallest distinct candidate, especially one that fits VRAM if any exists
        pick_best([&](const common_model_profile & c) {
            double s = -std::log2(double(std::max<uint64_t>(1, c.file_size_bytes)));
            if (c.file_size_bytes <= fit_threshold) {
                s += 100.0;
            }
            return s;
        });

        // Pick remaining up to max_per_category: maximize unseen architecture, unseen VRAM regime, and log-size spread
        while (chosen.size() < max_per_category && chosen.size() < pool.size()) {
            bool has_fit = false;
            bool has_exceed = false;
            std::set<std::string> seen_arches;
            std::set<common_decision_type> seen_dec;
            for (const auto * ch : chosen) {
                seen_arches.insert(ch->arch);
                seen_dec.insert(ch->decision_type);
                if (ch->file_size_bytes <= fit_threshold) has_fit = true;
                else has_exceed = true;
            }

            pick_best([&](const common_model_profile & c) {
                double s = 0.0;
                if (!seen_arches.count(c.arch)) {
                    s += 1000.0;
                }
                if (cat == COMMON_MODEL_CATEGORY_DECISION && !seen_dec.count(c.decision_type)) {
                    s += 500.0;
                }
                const bool c_fits = c.file_size_bytes <= fit_threshold;
                if ((c_fits && !has_fit) || (!c_fits && !has_exceed)) {
                    s += 400.0;
                }
                // Minimum log-distance from already-chosen models promotes spread across small/mid/large sizes
                double min_dist = 1e9;
                const double lc = std::log2(double(std::max<uint64_t>(1, c.file_size_bytes)));
                for (const auto * ch : chosen) {
                    const double lch = std::log2(double(std::max<uint64_t>(1, ch->file_size_bytes)));
                    min_dist = std::min(min_dist, std::abs(lc - lch));
                }
                s += min_dist * 50.0;
                return s;
            });
        }

        for (const auto * ch : chosen) {
            selected.push_back(*ch);
        }
    }

    return selected;
}

int common_run_optimization_worker(const std::string & trial_json_path) {
    std::string raw;
    if (!read_text_file(trial_json_path, raw)) {
        fprintf(stderr, "optimization-worker: cannot read %s\n", trial_json_path.c_str());
        return 2;
    }

    json j = json::parse_no_throw(raw);
    if (j.is_discarded() || !j.contains("result_path")) {
        fprintf(stderr, "optimization-worker: invalid trial json %s\n", trial_json_path.c_str());
        return 2;
    }

    const std::string result_path = j.at("result_path").get<std::string>();
    const common_opt_trial_config cfg = json_to_config(j);
    common_opt_trial_result res;
    res.exit_code = 1;

    // Apply trial environment overrides BEFORE llama_backend_init()
    apply_env_map(cfg.env_vars);

    common_init();
    llama_backend_init();

    common_params params;
    params.model.path            = cfg.model_path;
    params.n_ctx                 = cfg.n_ctx;
    params.n_batch               = cfg.n_batch;
    params.n_ubatch              = cfg.n_ubatch;
    params.n_gpu_layers          = cfg.n_gpu_layers;
    params.cpuparams.n_threads   = cfg.n_threads;
    params.cpuparams_batch.n_threads = cfg.n_threads;
    params.fit_params            = cfg.fit_params;
    params.kv_unified            = cfg.kv_unified;
    params.no_op_offload         = cfg.no_op_offload;
    params.load_mode             = cfg.use_mmap ? LLAMA_LOAD_MODE_AUTO : LLAMA_LOAD_MODE_NONE;
    params.warmup                = true;
    params.verbosity             = 4; // Capture cache stats in worker log
    params.flash_attn_type       = flash_attn_from_str(cfg.flash_attn, LLAMA_FLASH_ATTN_TYPE_AUTO);
    params.cache_type_k          = kv_type_from_str(cfg.kv_type_k, GGML_TYPE_F16);
    params.cache_type_v          = kv_type_from_str(cfg.kv_type_v, GGML_TYPE_F16);
    params.optimize_settings_path.clear();
    params.optimize_settings_applied = true;

    bool use_cpu_moe = cfg.cpu_moe;
    if (cfg.moe_backend == "upstream" && cfg.moe_cache_mib > 0) {
        use_cpu_moe                = true;
        params.moe_cache_size      = cfg.moe_cache_mib << 20;
        params.moe_cache_cpu_share = cfg.moe_cache_cpu_share;
    }
#ifdef COMMON_HAS_BACKEND_MOE_CACHE
    else if (cfg.moe_backend == "custom" && cfg.moe_cache_mib > 0) {
        params.moe_cache.mode          = COMMON_MOE_CACHE_MODE_ON;
        params.moe_cache.budget_mib    = cfg.moe_cache_mib;
        params.moe_cache.mode_explicit = true;
        params.no_extra_bufts          = true;
    }
#endif

    // Respect cpu_moe by adding the FFN expert CPU override just as arg.cpp does
    if (use_cpu_moe && cfg.moe_backend != "custom") {
        params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
    }
    // Pad tensor_buft_overrides buffer for common_fit_apply() (matches common/arg.cpp)
    if (params.fit_params || use_cpu_moe) {
        const size_t ntbo = llama_max_tensor_buft_overrides();
        while (params.tensor_buft_overrides.size() < ntbo) {
            params.tensor_buft_overrides.push_back({ nullptr, nullptr });
        }
    }

    const auto t_load0 = std::chrono::steady_clock::now();
    common_init_result_ptr init_res;
    try {
        init_res = common_init_from_params(params);
    } catch (const std::exception & e) {
        res.error = e.what();
        write_text_file_atomic(result_path, result_to_json(res).dump(2));
        llama_backend_free();
        return 1;
    }
    const auto t_load1 = std::chrono::steady_clock::now();
    res.load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

    llama_model   * model = init_res ? init_res->model()   : nullptr;
    llama_context * ctx   = init_res ? init_res->context() : nullptr;
    if (!model || !ctx) {
        res.error = "failed to load model/context";
        write_text_file_atomic(result_path, result_to_json(res).dump(2));
        llama_backend_free();
        return 1;
    }

    size_t vram_free = 0;
    size_t vram_total = 0;
    std::string gpu_name;
    int cpu_threads = 0;
    query_gpu_device_info(vram_free, vram_total, gpu_name, cpu_threads);
    res.vram_free_mib = vram_free >> 20;
    res.vram_used_mib = vram_total > vram_free ? ((vram_total - vram_free) >> 20) : 0;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token bos = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : LLAMA_TOKEN_NULL;
    const bool pure_encoder = llama_model_has_encoder(model) && !llama_model_has_decoder(model);

    bool bench_ok = true;
    const int32_t eff_n_ctx = (int32_t) llama_n_ctx(ctx);
    const int32_t eff_batch = std::max<int32_t>(1, (int32_t) llama_n_batch(ctx));
    const int32_t eff_ubatch = std::max<int32_t>(1, (int32_t) llama_n_ubatch(ctx));

    if (cfg.category == COMMON_MODEL_CATEGORY_DECISION) {
        const int32_t max_single = cfg.decision_is_encoder ? std::min(eff_batch, eff_ubatch) : eff_batch;
        const int32_t eval_tok = std::clamp<int32_t>(std::min(512, eff_n_ctx - 8), 32, max_single);
        const int32_t warmup_tok = std::clamp<int32_t>(32, 8, eval_tok);
        std::vector<llama_token> wtok(warmup_tok), ptok(eval_tok);
        for (int32_t i = 0; i < warmup_tok; ++i) wtok[i] = pick_test_token(vocab, i, bos);
        for (int32_t i = 0; i < eval_tok; ++i)   ptok[i] = pick_test_token(vocab, 50 + i, bos);

        bench_ok = eval_prompt_tokens(ctx, model, wtok, eff_batch, pure_encoder, true);
        const int n_passes = 3;
        double total_ms = 0.0;
        for (int p_idx = 0; bench_ok && p_idx < n_passes; ++p_idx) {
            if (llama_get_memory(ctx)) {
                llama_memory_clear(llama_get_memory(ctx), true);
            }
            const auto t0 = std::chrono::steady_clock::now();
            bench_ok = eval_prompt_tokens(ctx, model, ptok, eff_batch, pure_encoder, true);
            const auto t1 = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        if (bench_ok && total_ms > 0.0) {
            res.eval_ms  = total_ms / double(n_passes);
            res.eval_tps = (double(eval_tok) * double(n_passes)) / (total_ms / 1000.0);
        }
    } else if (cfg.category == COMMON_MODEL_CATEGORY_MOE) {
        const int32_t pp_tok = std::clamp<int32_t>(128, 16, eff_n_ctx / 2);
        const int32_t tg_tok = 48;
        std::vector<llama_token> wtok(4), p1(pp_tok), p2(pp_tok);
        for (int32_t i = 0; i < 4; ++i)      wtok[i] = pick_test_token(vocab, i, bos);
        for (int32_t i = 0; i < pp_tok; ++i) p1[i]   = pick_test_token(vocab, 100 + i, bos);
        for (int32_t i = 0; i < pp_tok; ++i) p2[i]   = pick_test_token(vocab, 600 + i, bos);

        bench_ok = eval_prompt_tokens(ctx, model, wtok, eff_batch, pure_encoder, false);
        if (bench_ok && llama_get_memory(ctx)) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        double ms_pp1 = 0.0, ms_tg1 = 0.0, ms_pp2 = 0.0, ms_tg2 = 0.0;
        if (bench_ok) {
            const auto t0 = std::chrono::steady_clock::now();
            bench_ok = eval_prompt_tokens(ctx, model, p1, eff_batch, pure_encoder, true);
            const auto t1 = std::chrono::steady_clock::now();
            ms_pp1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        if (bench_ok) {
            bench_ok = eval_decode_steps(ctx, model, vocab, tg_tok, ms_tg1);
        }
        if (bench_ok && llama_get_memory(ctx)) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        if (bench_ok) {
            const auto t0 = std::chrono::steady_clock::now();
            bench_ok = eval_prompt_tokens(ctx, model, p2, eff_batch, pure_encoder, true);
            const auto t1 = std::chrono::steady_clock::now();
            ms_pp2 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        if (bench_ok) {
            bench_ok = eval_decode_steps(ctx, model, vocab, tg_tok, ms_tg2);
        }
        if (bench_ok) {
            res.pp_tps      = ms_pp1 > 0.0 ? (double(pp_tok) / (ms_pp1 / 1000.0)) : 0.0;
            res.tg_tps      = ms_tg1 > 0.0 ? (double(tg_tok) / (ms_tg1 / 1000.0)) : 0.0;
            res.pp_warm_tps = ms_pp2 > 0.0 ? (double(pp_tok) / (ms_pp2 / 1000.0)) : 0.0;
            res.tg_warm_tps = ms_tg2 > 0.0 ? (double(tg_tok) / (ms_tg2 / 1000.0)) : 0.0;
        }
    } else {
        // Dense
        const int32_t pp_tok = std::clamp<int32_t>(256, 32, eff_n_ctx / 2);
        const int32_t tg_tok = 48;
        std::vector<llama_token> wtok(4), p1(pp_tok);
        for (int32_t i = 0; i < 4; ++i)      wtok[i] = pick_test_token(vocab, i, bos);
        for (int32_t i = 0; i < pp_tok; ++i) p1[i]   = pick_test_token(vocab, 200 + i, bos);

        bench_ok = eval_prompt_tokens(ctx, model, wtok, eff_batch, pure_encoder, false);
        if (bench_ok && llama_get_memory(ctx)) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        double ms_pp = 0.0, ms_tg = 0.0;
        if (bench_ok) {
            const auto t0 = std::chrono::steady_clock::now();
            bench_ok = eval_prompt_tokens(ctx, model, p1, eff_batch, pure_encoder, true);
            const auto t1 = std::chrono::steady_clock::now();
            ms_pp = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        if (bench_ok) {
            bench_ok = eval_decode_steps(ctx, model, vocab, tg_tok, ms_tg);
        }
        if (bench_ok) {
            res.pp_tps = ms_pp > 0.0 ? (double(pp_tok) / (ms_pp / 1000.0)) : 0.0;
            res.tg_tps = ms_tg > 0.0 ? (double(tg_tok) / (ms_tg / 1000.0)) : 0.0;
        }
    }

    res.coherent  = bench_ok;
    res.ok        = bench_ok;
    res.exit_code = bench_ok ? 0 : 1;
    res.score     = compute_trial_score(cfg.category, res);
    if (!bench_ok && res.error.empty()) {
        res.error = "eval or output coherence check failed";
    }

    // Re-sample VRAM after eval while context and MoE cache allocations are live
    {
        size_t post_free = 0;
        size_t post_total = 0;
        std::string post_gpu;
        int post_threads = 0;
        query_gpu_device_info(post_free, post_total, post_gpu, post_threads);
        if (post_total > 0) {
            const size_t post_used = post_total > post_free ? ((post_total - post_free) >> 20) : 0;
            const size_t post_free_mib = post_free >> 20;
            res.vram_used_mib = std::max(res.vram_used_mib, post_used);
            res.vram_free_mib = res.vram_free_mib > 0 ? std::min(res.vram_free_mib, post_free_mib) : post_free_mib;
        }
    }

    init_res.reset(); // prints llama_moe_cache summary on destruction before exiting
    common_log_flush(common_log_main());
    write_text_file_atomic(result_path, result_to_json(res).dump(2));
    llama_backend_free();
    return res.ok ? 0 : 1;
}

common_json common_derive_universal_settings(
        const std::vector<common_opt_model_summary> & summaries,
        const std::string & gpu_name,
        size_t gpu_vram_total_bytes,
        int cpu_threads) {
    json root = json::object();
    root["schema_version"] = 1;
    root["generated_at_epoch"] = (long long) std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    json hw = json::object();
    hw["gpu_name"]                = gpu_name;
    hw["gpu_vram_total_mib"]      = (unsigned long long) (gpu_vram_total_bytes >> 20);
    hw["cpu_threads"]             = cpu_threads;
#ifdef COMMON_HAS_BACKEND_MOE_CACHE
    hw["has_backend_moe_cache"]   = true;
#else
    hw["has_backend_moe_cache"]   = false;
#endif
    root["hardware"] = hw;

    // Group tested examples into category buckets to derive universalized rules
    std::vector<const common_opt_model_summary *> moe_fit, moe_near, moe_exc;
    std::vector<const common_opt_model_summary *> dense_sm, dense_md, dense_lg;
    std::vector<const common_opt_model_summary *> dec_enc, dec_fit, dec_tight;
    std::map<uint32_t, std::vector<const common_opt_model_summary *>> by_head_dim;

    for (const auto & s : summaries) {
        if (!s.best.ok) {
            continue;
        }
        const std::string pkey = select_profile_key(s.profile, gpu_vram_total_bytes);
        if      (pkey == "moe.fits_vram")          moe_fit.push_back(&s);
        else if (pkey == "moe.near_vram")          moe_near.push_back(&s);
        else if (pkey == "moe.exceeds_vram")       moe_exc.push_back(&s);
        else if (pkey == "dense.small")            dense_sm.push_back(&s);
        else if (pkey == "dense.medium")           dense_md.push_back(&s);
        else if (pkey == "dense.large")            dense_lg.push_back(&s);
        else if (pkey == "decision.encoder")       dec_enc.push_back(&s);
        else if (pkey == "decision.causal_fits")   dec_fit.push_back(&s);
        else if (pkey == "decision.causal_tight")  dec_tight.push_back(&s);

        if (s.profile.head_dim_k > 0) {
            by_head_dim[s.profile.head_dim_k].push_back(&s);
        }
    }

    // Fallbacks only between small and medium dense when < 2 fit samples; keep conservative dense.large defaults when none fit
    std::vector<const common_opt_model_summary *> all_fit_dense = dense_sm;
    all_fit_dense.insert(all_fit_dense.end(), dense_md.begin(), dense_md.end());
    if (dense_md.empty()) dense_md = all_fit_dense;
    if (dense_sm.empty()) dense_sm = all_fit_dense;
    if (dec_tight.empty()) dec_tight = dec_fit;

    json defaults = json::object();
    defaults["flash_attn"]   = "on";
    defaults["kv_unified"]   = true;
    defaults["fit_params"]   = true;
    json def_env = json::object();
    def_env["LLAMA_SM70_SKINNY"]           = "0";
    def_env["GGML_CUDA_VOLTA_FORCE_MMQ"]   = "moe";
    def_env["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
    def_env["GGML_CUDA_REGISTER_HOST"]     = "1";
    def_env["GGML_CUDA_PREFETCH_WEIGHTS"]  = "1";
    defaults["env_vars"] = def_env;
    root["universal_defaults"] = defaults;

    json profiles = json::object();
    profiles["moe.fits_vram"] = build_universal_profile_rule(
        moe_fit, "on", "q4_0", "q4_0", 2048, 1024, true,
        false, true, false, "moe", "0", "none", 0.0, 768, 0.0, false,
        "universal MoE models fitting <=85% VRAM (all experts resident on GPU)");

    profiles["moe.near_vram"] = build_universal_profile_rule(
        moe_near, "on", "q4_0", "q4_0", 2048, 1024, true,
        true, false, false, "moe", "0", "none", 0.85, 768, 0.5, false,
        "universal MoE models at 85%-115% VRAM (Strategy A static split vs upstream cache head-to-head)");

    profiles["moe.exceeds_vram"] = build_universal_profile_rule(
        moe_exc, "on", "q4_0", "q4_0", 2048, 1024, true,
        true, false, false, "moe", "1", "upstream", 0.88, 640, 0.5, true,
        "universal MoE models exceeding >115% VRAM (dynamic spare-VRAM expert caching)");

    profiles["dense.small"] = build_universal_profile_rule(
        dense_sm, "on", "q8_0", "q8_0", 2048, 2048, true,
        false, false, false, "0", "0", "none", 0.0, 512, 0.0, false,
        "universal dense models <=45% VRAM (headroom allows Q8/F16 KV and large ubatch)");

    profiles["dense.medium"] = build_universal_profile_rule(
        dense_md, "on", "q8_0", "q8_0", 2048, 1024, true,
        false, false, false, "0", "0", "none", 0.0, 512, 0.0, false,
        "universal dense models 45%-85% VRAM");

    profiles["dense.large"] = build_universal_profile_rule(
        dense_lg, "on", "q4_0", "q4_0", 2048, 512, true,
        false, false, false, "0", "1", "none", 0.0, 768, 0.0, false,
        "universal dense models >85% VRAM (Q4 KV + host weight prefetch)");

    profiles["decision.encoder"] = build_universal_profile_rule(
        dec_enc, "on", "f16", "f16", 2048, 2048, true,
        false, false, true, "0", "0", "none", 0.0, 256, 0.0, false,
        "universal bidirectional encoder decision models (n_ubatch >= n_batch single-pass)");

    profiles["decision.causal_fits"] = build_universal_profile_rule(
        dec_fit, "on", "q8_0", "q8_0", 2048, 2048, true,
        false, false, false, "0", "0", "none", 0.0, 512, 0.0, false,
        "universal causal decision models <=65% VRAM (fast prompt classification)");

    profiles["decision.causal_tight"] = build_universal_profile_rule(
        dec_tight, "on", "q4_0", "q4_0", 2048, 1024, true,
        false, false, false, "0", "1", "none", 0.0, 640, 0.0, false,
        "universal causal decision models >65% VRAM (Q4 KV preserves VRAM margin)");

    root["profiles"] = profiles;

    json head_rules = json::object();
    const uint32_t std_dims[] = { 128U, 256U, 512U };
    for (uint32_t d : std_dims) {
        const auto it = by_head_dim.find(d);
        json hr = json::object();
        if (it != by_head_dim.end() && !it->second.empty()) {
            hr["flash_attn"] = majority_or_best_str(it->second, [](const common_opt_model_summary & s) {
                return s.best_config.flash_attn;
            }, "on");
            hr["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
        } else {
            hr["flash_attn"] = "on";
            hr["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "1";
        }
        head_rules[std::to_string(d)] = hr;
    }
    root["head_dim_rules"] = head_rules;

    json ev = json::array();
    for (const auto & s : summaries) {
        json e = json::object();
        e["model"]            = s.profile.name;
        e["arch"]             = s.profile.arch;
        e["category"]         = common_model_category_name(s.profile.category);
        e["derived_bucket"]   = select_profile_key(s.profile, gpu_vram_total_bytes);
        e["file_size_gib"]    = double(s.profile.file_size_bytes) / double(1ULL << 30);
        e["best_round"]       = s.best_round;
        e["baseline_score"]   = s.baseline.score;
        e["best_score"]       = s.best.score;
        e["speedup_pct"]      = (s.baseline.score > 0.0)
            ? ((s.best.score - s.baseline.score) / s.baseline.score) * 100.0
            : 0.0;
        e["winning_strategy"] = s.winning_strategy;
        ev.push_back(e);
    }
    root["evidence"] = ev;

    return root;
}

int common_run_optimization_test(const common_params & params, const std::string & self_exe) {
    size_t vram_free = 0;
    size_t vram_total = 0;
    std::string gpu_name;
    int cpu_threads = 0;
    query_gpu_device_info(vram_free, vram_total, gpu_name, cpu_threads);

    printf("========================================================================\n");
    printf("llama.cpp Hardware & Model Optimization Test Loop (--optimization-test)\n");
    printf("========================================================================\n");
    printf("Device     : %s (total=%zu MiB, free=%zu MiB)\n",
           gpu_name.c_str(), vram_total >> 20, vram_free >> 20);
    printf("CPU Threads: %d math threads\n", cpu_threads);
#ifdef COMMON_HAS_BACKEND_MOE_CACHE
    printf("MoE Backend: BOTH upstream (llama_moe_cache) AND custom (--moe-cache) available\n");
#else
    printf("MoE Backend: upstream (llama_moe_cache) active\n");
#endif
    printf("\n");

    std::vector<std::string> search_dirs;
    if (!params.models_dir.empty()) {
        search_dirs.push_back(params.models_dir);
    }
    search_dirs.push_back("/models");
    search_dirs.push_back("/opt/models");
    search_dirs.push_back(".");

    const auto discovered = common_discover_models(search_dirs, true);
    const auto selected = common_select_optimization_models(discovered, vram_total, 3);

    if (selected.empty()) {
        LOG_ERR("%s: no valid GGUF models discovered under search directories\n", __func__);
        return 1;
    }

    printf("\n[optimize] Selected diverse benchmark matrix (%zu models, up to 3 per category):\n", selected.size());
    for (size_t i = 0; i < selected.size(); ++i) {
        const auto & m = selected[i];
        printf("  %zu. [%-8s] %-12s | %6.2f GiB | %s | min_exp=%llu KiB | %s\n",
               i + 1,
               common_model_category_name(m.category),
               m.arch.c_str(),
               double(m.file_size_bytes) / double(1ULL << 30),
               (vram_total > 0 && double(m.file_size_bytes) <= double(vram_total) * 0.85) ? "fits-VRAM" : "exceeds-VRAM",
               (unsigned long long) (m.min_expert_slice_bytes >> 10),
               m.name.c_str());
    }
    printf("\n");
    fflush(stdout);

    const std::string exe = resolve_self_executable(self_exe);
    int trial_seq = 0;
    std::vector<common_opt_model_summary> summaries;
    summaries.reserve(selected.size());

    for (size_t mi = 0; mi < selected.size(); ++mi) {
        const auto & p = selected[mi];
        printf("------------------------------------------------------------------------\n");
        printf("[MODEL %zu/%zu] %s (%s, arch=%s, %.2f GiB, L=%u, E=%ux%u, dk=%u)\n",
               mi + 1, selected.size(),
               p.name.c_str(),
               common_model_category_name(p.category),
               p.arch.c_str(),
               double(p.file_size_bytes) / double(1ULL << 30),
               p.n_layers, p.n_expert, p.n_expert_used, p.head_dim_k);
        printf("------------------------------------------------------------------------\n");

        std::vector<common_opt_trial_config> hist_cfg;
        std::vector<common_opt_trial_result> hist_res;

        // Pass 0: Baseline
        const common_opt_trial_config base_cfg = make_baseline_config(p, cpu_threads);
        common_opt_trial_result base_res;
        printf("  Running %s: %s\n", base_cfg.label.c_str(), base_cfg.hypothesis.c_str());
        fflush(stdout);
        run_trial_subprocess(exe, base_cfg, trial_seq++, base_res);
        print_trial_metrics(p, base_cfg, base_res, base_res.score);
        hist_cfg.push_back(base_cfg);
        hist_res.push_back(base_res);

        common_opt_trial_config best_cfg = base_cfg;
        common_opt_trial_result best_res = base_res;
        int best_round = 0;
        int rounds_run = 0;

        for (int round = 1; round <= 3; ++round) {
            const common_opt_trial_config rcfg = plan_round_config(p, round, vram_total, cpu_threads, hist_cfg, hist_res);
            common_opt_trial_result rres;
            printf("\n  Running %s\n", rcfg.label.c_str());
            printf("  [ADJUST] Hypothesis: %s\n", rcfg.hypothesis.c_str());
            printf("  [PARAMS] fa=%s ctk=%s ctv=%s b=%d ub=%d kv_uni=%s moe=%s(%zuMiB,share=%.2f)\n",
                   rcfg.flash_attn.c_str(), rcfg.kv_type_k.c_str(), rcfg.kv_type_v.c_str(),
                   rcfg.n_batch, rcfg.n_ubatch, rcfg.kv_unified ? "true" : "false",
                   rcfg.moe_backend.c_str(), rcfg.moe_cache_mib, rcfg.moe_cache_cpu_share);
            fflush(stdout);

            run_trial_subprocess(exe, rcfg, trial_seq++, rres);
            print_trial_metrics(p, rcfg, rres, base_res.ok ? base_res.score : 0.0);
            hist_cfg.push_back(rcfg);
            hist_res.push_back(rres);
            rounds_run = round;

            if (rres.ok && rres.coherent && (!best_res.ok || rres.score > best_res.score)) {
                const double gain_vs_best = best_res.score > 0.0
                    ? ((rres.score - best_res.score) / best_res.score) * 100.0
                    : 100.0;
                printf("  [DECIDE] KEEP Round %d (new best score=%.2f, %+.1f%% over prior best)\n",
                       round, rres.score, gain_vs_best);
                best_cfg   = rcfg;
                best_res   = rres;
                best_round = round;
            } else {
                printf("  [DECIDE] REVERT to Round %d (score=%.2f remained higher)\n",
                       best_round, best_res.score);
            }
            fflush(stdout);
        }

        common_opt_model_summary sum;
        sum.profile          = p;
        sum.baseline         = base_res;
        sum.best             = best_res;
        sum.best_config      = best_cfg;
        sum.best_round       = best_round;
        sum.rounds_run       = rounds_run;
        sum.winning_strategy = best_cfg.label;
        summaries.push_back(sum);

        const double total_gain = (base_res.score > 0.0 && best_res.score > 0.0)
            ? ((best_res.score - base_res.score) / base_res.score) * 100.0
            : 0.0;
        printf("\n  [MODEL SUMMARY] Best = Round %d (%s) | Baseline=%.2f -> Best=%.2f (%+.1f%%)\n\n",
               best_round, best_cfg.label.c_str(), base_res.score, best_res.score, total_gain);
        fflush(stdout);
    }

    const json out_json = common_derive_universal_settings(summaries, gpu_name, vram_total, cpu_threads);
    const std::string out_path = (fs::current_path() / "optimized-settings.json").string();
    if (!write_text_file_atomic(out_path, out_json.dump(2) + "\n")) {
        LOG_ERR("%s: failed to write %s\n", __func__, out_path.c_str());
        return 1;
    }

    printf("========================================================================\n");
    printf("Universalized Rules Derived -> %s\n", out_path.c_str());
    printf("========================================================================\n");
    if (out_json.contains("profiles")) {
        for (const auto & item : out_json.at("profiles").items()) {
            const json & prof = item.value();
            printf("  %-22s: fa=%-4s kv=%s/%s b=%d ub=%d kv_uni=%s mmq=%s",
                   item.key().c_str(),
                   prof.value("flash_attn", "auto").c_str(),
                   prof.value("kv_type_k", "f16").c_str(),
                   prof.value("kv_type_v", "f16").c_str(),
                   prof.value("n_batch", 2048),
                   prof.value("n_ubatch", 512),
                   prof.value("kv_unified", false) ? "1" : "0",
                   prof.at("env_vars").value("GGML_CUDA_VOLTA_FORCE_MMQ", "").c_str());
            if (prof.contains("moe_cache_backend")) {
                printf(" moe=%s(fill=%.2f,res=%dMiB,share=%.2f)",
                       prof.value("moe_cache_backend", "none").c_str(),
                       prof.value("moe_cache_vram_fill_ratio", 0.0),
                       prof.value("moe_cache_reserve_mib", 0),
                       prof.value("moe_cache_cpu_share", 0.0));
            }
            printf("\n");
        }
    }
    printf("========================================================================\n");
    fflush(stdout);
    return 0;
}

bool common_optimize_settings_apply(common_params & params) {
    if (params.optimize_settings_applied || params.optimize_settings_path.empty() || params.model.path.empty()) {
        return false;
    }
    params.optimize_settings_applied = true;

    std::string raw;
    if (!read_text_file(params.optimize_settings_path, raw)) {
        LOG_WRN("[optimize-settings] file not found or unreadable: %s (keeping existing params)\n",
                params.optimize_settings_path.c_str());
        return false;
    }

    json root = json::parse_no_throw(raw);
    if (root.is_discarded() || !root.contains("profiles")) {
        LOG_WRN("[optimize-settings] invalid JSON in %s (keeping existing params)\n",
                params.optimize_settings_path.c_str());
        return false;
    }

    const common_model_profile profile = common_probe_model_profile(params.model.path);
    if (!profile.valid) {
        LOG_WRN("[optimize-settings] failed to probe model profile for %s (%s)\n",
                params.model.path.c_str(), profile.skip_reason.c_str());
        return false;
    }

    size_t vram_free = 0;
    size_t vram_total = 0;
    std::string gpu_name;
    int cpu_threads = 0;
    query_gpu_device_info(vram_free, vram_total, gpu_name, cpu_threads);

    if (root.contains("hardware")) {
        const size_t hw_mib = (size_t) root.at("hardware").value("gpu_vram_total_mib", 0ULL);
        const size_t hw_bytes = hw_mib << 20;
        if (hw_bytes > 0 && (vram_total == 0 || hw_bytes < vram_total)) {
            vram_total = hw_bytes;
            if (vram_free == 0 || vram_free > vram_total) {
                vram_free = vram_total;
            }
        }
    }

    // Classify against live available GPU memory so multi-model router setups adapt dynamically
    const size_t ref_vram = (vram_free > 0 && vram_free < vram_total) ? vram_free : vram_total;
    const std::string profile_key = select_profile_key(profile, ref_vram);

    const json & profiles = root.at("profiles");
    if (!profiles.contains(profile_key)) {
        LOG_WRN("[optimize-settings] no rule matching '%s' in %s\n",
                profile_key.c_str(), params.optimize_settings_path.c_str());
        return false;
    }
    const json & rule = profiles.at(profile_key);

    // 1. Apply universal defaults + head_dim_rules + rule env_vars
    std::map<std::string, std::string> env_merged;
    if (root.contains("universal_defaults") && root.at("universal_defaults").contains("env_vars")) {
        for (const auto & [k, v] : json_to_env_map(root.at("universal_defaults").at("env_vars"))) {
            env_merged[k] = v;
        }
    }
    if (root.contains("head_dim_rules") && profile.head_dim_k > 0) {
        const std::string hd_key = std::to_string(profile.head_dim_k);
        const json & hd_rules = root.at("head_dim_rules");
        if (hd_rules.contains(hd_key)) {
            const json & hr = hd_rules.at(hd_key);
            if (hr.contains("GGML_CUDA_VOLTA_Q8_FATTN_TC")) {
                env_merged["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = hr.value("GGML_CUDA_VOLTA_Q8_FATTN_TC", "1");
            }
        }
    }
    if (rule.contains("env_vars")) {
        for (const auto & [k, v] : json_to_env_map(rule.at("env_vars"))) {
            env_merged[k] = v;
        }
    }

    // 2. Apply attention, KV, and batching adaptations cleanly
    if (rule.contains("flash_attn")) {
        params.flash_attn_type = flash_attn_from_str(rule.value("flash_attn", "auto"), params.flash_attn_type);
    }
    const ggml_type kv_k = kv_type_from_str(rule.value("kv_type_k", "f16"), params.cache_type_k);
    const ggml_type kv_v = kv_type_from_str(rule.value("kv_type_v", "f16"), params.cache_type_v);
    if (params.no_kv_quant) {
        params.cache_type_k = GGML_TYPE_F16;
        params.cache_type_v = GGML_TYPE_F16;
    } else {
        params.cache_type_k = kv_k;
        params.cache_type_v = kv_v;
    }

    if (rule.contains("kv_unified")) {
        params.kv_unified = rule.value("kv_unified", params.kv_unified);
    }

    const int32_t rule_batch  = rule.value("n_batch", params.n_batch);
    const int32_t rule_ubatch = rule.value("n_ubatch", params.n_ubatch);
    if (profile.decision_is_encoder || rule.value("encoder_match_batch", false)) {
        // Never shrink explicit preset batch limits (e.g. 7999/8192 for GLiDE/Laya); keep ubatch == batch
        params.n_batch  = std::max(params.n_batch, rule_batch);
        params.n_ubatch = std::max(params.n_ubatch, params.n_batch);
        params.n_batch  = std::max(params.n_batch, params.n_ubatch);
    } else {
        params.n_batch  = std::max(params.n_batch, rule_batch);
        params.n_ubatch = std::min(params.n_batch, rule_ubatch);
    }

    // Account for draft and multimodal projector VRAM overhead when planning budgets
    const size_t live_avail = vram_free > 0 ? vram_free : vram_total;
    uint64_t extra_vram_bytes = 0;
    auto add_extra_vram_file = [&](const std::string & fpath) {
        if (fpath.empty()) {
            return;
        }
        std::error_code f_ec;
        const uint64_t fsz = fs::file_size(fpath, f_ec);
        if (!f_ec && fsz > 0) {
            extra_vram_bytes += fsz;
        }
    };
    add_extra_vram_file(params.speculative.draft.mparams.path);
    add_extra_vram_file(params.mmproj.path);
    if (extra_vram_bytes > 0) {
        extra_vram_bytes += 256ULL << 20;
    }
    const size_t eff_avail_bytes = (live_avail > extra_vram_bytes) ? size_t(live_avail - extra_vram_bytes) : 0;

    // 3. Guardrail: step down KV precision when high context + resident weights exceed live VRAM
    const std::string moe_backend = rule.value("moe_cache_backend", "none");
    const bool uses_dynamic_moe = (profile.category == COMMON_MODEL_CATEGORY_MOE && moe_backend != "none" && params.moe_cache_size == 0);
    if (uses_dynamic_moe && params.n_ctx == 0) {
        const uint32_t cap = profile.n_ctx_train > 0 ? std::min<uint32_t>(profile.n_ctx_train, 32768U) : 32768U;
        params.n_ctx = (int32_t) cap;
    }
    const uint32_t eff_ctx = params.n_ctx > 0
        ? (uint32_t) params.n_ctx
        : (profile.n_ctx_train > 0 ? std::min<uint32_t>(profile.n_ctx_train, 32768U) : 4096U);

    const uint64_t resident_bytes = (uses_dynamic_moe && profile.non_expert_bytes > 0)
        ? profile.non_expert_bytes
        : profile.file_size_bytes;
    auto kv_fits_vram = [&](ggml_type tk, ggml_type tv) -> bool {
        if (live_avail == 0) {
            return true;
        }
        const uint64_t kv_bytes = estimate_kv_cache_bytes(profile, eff_ctx, tk, tv);
        return resident_bytes + kv_bytes + extra_vram_bytes + (768ULL << 20) <= live_avail;
    };
    if (!params.no_kv_quant && !profile.decision_is_encoder && live_avail > 0 && !kv_fits_vram(params.cache_type_k, params.cache_type_v)) {
        const ggml_type orig_k = params.cache_type_k;
        const ggml_type orig_v = params.cache_type_v;
        auto is_f16_or_higher = [](ggml_type t) {
            return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
        };
        if (is_f16_or_higher(params.cache_type_k) || is_f16_or_higher(params.cache_type_v)) {
            if (is_f16_or_higher(params.cache_type_k)) params.cache_type_k = GGML_TYPE_Q8_0;
            if (is_f16_or_higher(params.cache_type_v)) params.cache_type_v = GGML_TYPE_Q8_0;
        }
        if (!kv_fits_vram(params.cache_type_k, params.cache_type_v)) {
            params.cache_type_k = GGML_TYPE_Q4_0;
            params.cache_type_v = GGML_TYPE_Q4_0;
        }
        if (params.cache_type_k != orig_k || params.cache_type_v != orig_v) {
            LOG_INF("[optimize-settings] stepped KV cache down to %s/%s for n_ctx=%u VRAM headroom\n",
                    ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v), eff_ctx);
        }
    }

    // 4. Apply dynamic MoE cache sizing after KV step-down and draft/mmproj VRAM deductions
    size_t chosen_cache_mib = 0;
    std::string chosen_backend = "none";

    if (uses_dynamic_moe) {
        const double fill_ratio = rule.value("moe_cache_vram_fill_ratio", 0.85);
        const size_t reserve_mib = (size_t) rule.value("moe_cache_reserve_mib", 768);
        chosen_cache_mib = compute_dynamic_moe_budget_mib(
            profile, eff_avail_bytes, fill_ratio, reserve_mib, eff_ctx, params.cache_type_k, params.cache_type_v);

        if (chosen_cache_mib > 0) {
            params.fit_params = false;
#ifdef COMMON_HAS_BACKEND_MOE_CACHE
            if (moe_backend == "custom") {
                params.moe_cache_size          = 0;
                params.moe_cache.mode          = COMMON_MOE_CACHE_MODE_ON;
                params.moe_cache.budget_mib    = chosen_cache_mib;
                params.moe_cache.mode_explicit = true;
                params.no_extra_bufts          = true;
                if (rule.value("moe_cache_adapt_min_expert_kb", true) && profile.min_expert_slice_bytes > 0) {
                    const size_t min_exp_kb = std::max<size_t>(64, size_t(profile.min_expert_slice_bytes / 1024ULL) / 2ULL);
                    env_merged["GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB"] = std::to_string(min_exp_kb);
                }
                chosen_backend = "custom";
            } else
#endif
            {
                params.moe_cache_size      = chosen_cache_mib << 20;
                params.moe_cache_cpu_share = (float) rule.value("moe_cache_cpu_share", 0.5);
                params.no_host             = false;
#ifdef COMMON_HAS_BACKEND_MOE_CACHE
                params.moe_cache.mode      = COMMON_MOE_CACHE_MODE_OFF;
#endif
                // Insert CPU expert override only if no non-empty user overrides exist, preserving ntbo padding
                if (!has_active_buft_overrides(params.tensor_buft_overrides)) {
                    params.tensor_buft_overrides.clear();
                    params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
                    const size_t ntbo = llama_max_tensor_buft_overrides();
                    while (params.tensor_buft_overrides.size() < ntbo) {
                        params.tensor_buft_overrides.push_back({ nullptr, nullptr });
                    }
                }
                chosen_backend = "upstream";
            }
        }
    }

    // Relax global router preset n-gpu-layers=99 to auto (-1) on oversized non-MoE models so fit_params can split layers
    if (profile.category != COMMON_MODEL_CATEGORY_MOE &&
        params.fit_params &&
        params.n_gpu_layers >= 99 &&
        live_avail > 0 &&
        profile.file_size_bytes + extra_vram_bytes + (768ULL << 20) > live_avail) {
        params.n_gpu_layers = -1;
        LOG_INF("[optimize-settings] relaxed n-gpu-layers=99 to auto (-1) for oversized model (%s)\n",
                profile.name.c_str());
    }

#ifdef COMMON_HAS_BACKEND_MOE_CACHE
    if (chosen_backend != "custom" &&
        params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO &&
        !params.moe_cache.mode_explicit) {
        params.moe_cache.mode = COMMON_MOE_CACHE_MODE_OFF;
    }
#endif

    // Preserve explicit --no-kv-quant FP16 override
    if (params.no_kv_quant) {
        params.cache_type_k = GGML_TYPE_F16;
        params.cache_type_v = GGML_TYPE_F16;
    }
    apply_env_map(env_merged);

    LOG_INF("[optimize-settings] model='%s' arch=%s cat=%s size=%.2f GiB dk=%u | "
            "VRAM free=%zu MiB total=%zu MiB | matched rule='%s'\n",
            profile.name.c_str(), profile.arch.c_str(), common_model_category_name(profile.category),
            double(profile.file_size_bytes) / double(1ULL << 30), profile.head_dim_k,
            vram_free >> 20, vram_total >> 20, profile_key.c_str());
    LOG_INF("[optimize-settings] applied: fa=%s kv=%s/%s b=%d ub=%d kv_unified=%s moe=%s(%zu MiB, share=%.2f) MMQ=%s prefetch=%s\n",
            flash_attn_to_str(params.flash_attn_type),
            ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
            params.n_batch, params.n_ubatch, params.kv_unified ? "true" : "false",
            chosen_backend.c_str(), chosen_cache_mib, params.moe_cache_cpu_share,
            env_merged["GGML_CUDA_VOLTA_FORCE_MMQ"].c_str(),
            env_merged["GGML_CUDA_PREFETCH_WEIGHTS"].c_str());
    return true;
}
