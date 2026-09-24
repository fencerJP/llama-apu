#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

#include "ggml-apu-bridge.h"

// Forward declaration of llama structs
struct llama_model;
struct ggml_tensor;

enum apu_moe_router_mode {
    APU_MOE_ROUTER_AUTO = 0,
    APU_MOE_ROUTER_ON   = 1,
    APU_MOE_ROUTER_OFF  = 2,
};

struct apu_moe_router_info {
    bool        is_moe             = false;
    std::string arch_name          = "";
    uint32_t    n_expert_layers    = 0;
    uint32_t    n_experts          = 0;
    uint32_t    n_experts_used     = 0;
    uint32_t    n_embd             = 0;
    size_t      total_router_bytes = 0;
    size_t      sram_limit_bytes   = 32 * 1024 * 1024; // Default 32MB AIE2P tile SRAM
    apu_moe_router_mode mode       = APU_MOE_ROUTER_AUTO;
    bool        pinned_in_sram     = false;
    bool        fallback_to_dram   = false;
    std::string warning_msg        = "";
    std::vector<std::string> router_tensor_names;
};

class apu_moe_router_manager {
public:
    static apu_moe_router_manager & get();

    void set_mode(apu_moe_router_mode mode);
    void set_sram_limit_mb(int32_t limit_mb);

    apu_moe_router_mode get_mode() const;
    int32_t get_sram_limit_mb() const;

    // Evaluates a model for MoE router isolation & SRAM budgeting
    apu_moe_router_info evaluate_model(const llama_model & model);

    // Direct placement & pinning of router gating matrices into AIE2P SRAM
    bool pin_router_matrices(llama_model & model);

    // Retrieve active info/telemetry
    apu_moe_router_info get_info() const;

    // Reset state for new model
    void reset();

private:
    apu_moe_router_manager();
    ~apu_moe_router_manager();

    mutable std::mutex mutex_;
    apu_moe_router_mode mode_ = APU_MOE_ROUTER_AUTO;
    int32_t sram_limit_mb_    = 32; // Default 32MB
    apu_moe_router_info active_info_{};
    std::unique_ptr<apu_gem_buffer> sram_buffer_;
    std::vector<void *> pinned_pointers_;
};

// Standalone audit helper for apu-cli and test suites
bool apu_audit_moe_router(const std::string & model_path,
                          apu_moe_router_mode mode,
                          int32_t limit_mb,
                          bool verbose,
                          apu_moe_router_info & out_info,
                          std::string & out_log);
