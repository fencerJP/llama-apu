#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

struct apu_convert_options {
    std::string source_path;
    std::string output_dir;
    std::string target_quant    = "TQ2_0";
    std::string fallback_quant  = "Q4_K_M";
    bool skip_sidecar           = false;
    bool skip_xclbin            = false;
    float memory_ceiling_gb     = 50.0f;
};

struct apu_convert_result {
    bool success                = false;
    std::string error;
    std::string gguf_path;
    std::string q4nx_path;
    std::string xclbin_path;
    std::string selected_quant;
    size_t gguf_size_bytes      = 0;
    size_t q4nx_size_bytes      = 0;
    bool sidecar_linked         = false;
    bool profile_registered     = false;
};

// Milestone 7.2.1: Run unified end-to-end conversion
bool apu_run_conversion_pipeline(const apu_convert_options & opts, apu_convert_result & result);

// Milestone 7.2.4: Unit test verification gate
bool test_apu_convert(bool verbose);
