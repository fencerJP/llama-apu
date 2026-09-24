#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Target NPU hardware architecture
typedef enum {
    APU_XCLBIN_TARGET_NPU1_AIE2  = 1, // Phoenix / Hawk Point 20-tile (4x5 array)
    APU_XCLBIN_TARGET_NPU2_AIE2P = 2  // Strix Point / Krackan / Gorgon 32-tile (4x8 array)
} apu_xclbin_target;

#ifdef __cplusplus
}
#endif

struct apu_xclbin_topology {
    std::string arch_name;
    uint32_t hidden_dim     = 2048;
    uint32_t num_heads      = 32;
    uint32_t num_kv_heads   = 8;
    uint32_t num_layers     = 16;
    uint32_t vocab_size     = 128000;
    uint32_t context_length = 8192;
    uint32_t head_dim       = 64;
    uint32_t ffn_dim        = 8192;
    uint32_t num_experts    = 0;
};

struct apu_xclbin_synth_options {
    apu_xclbin_target target     = APU_XCLBIN_TARGET_NPU2_AIE2P;
    bool router_sram_enabled     = true;
    uint32_t router_sram_limit_mb = 32;
    std::string custom_pdi_path;
    std::string output_path;
    std::string model_stem;
    std::string parent_stem;
    bool register_user_profile   = true;
};

struct apu_xclbin_synth_status {
    bool success                 = false;
    std::string error;
    std::string output_path;
    size_t file_size             = 0;
    size_t pinned_sram_bytes     = 0;
    uint32_t pinned_layers       = 0;
    bool dma_alignment_ok        = false;
    bool xclbinutil_invoked      = false;
};

// Milestone 7.1.1: Extract structural topology from GGUF/model descriptor
bool apu_extract_topology(const std::string & model_path, apu_xclbin_topology & out_topo);

// Milestone 7.1.3: Synthesize complete XCLBIN binary with full metadata sections
bool apu_synthesize_xclbin(const apu_xclbin_topology & topo,
                           const apu_xclbin_synth_options & opts,
                           apu_xclbin_synth_status & out_status);

// Milestone 7.1.4: Validate binary integrity, magic, and embedded sections
bool apu_validate_xclbin_metadata(const std::string & xclbin_path,
                                  bool verbose,
                                  std::string & out_log);

// Milestone 7.1 Unit & integration verification gate
bool test_apu_xclbin_synth(bool verbose);
