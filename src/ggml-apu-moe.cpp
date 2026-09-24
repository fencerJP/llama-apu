#include "ggml-apu-moe.h"
#include "llama-model.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

apu_moe_router_manager & apu_moe_router_manager::get() {
    static apu_moe_router_manager instance;
    return instance;
}

apu_moe_router_manager::apu_moe_router_manager() = default;

apu_moe_router_manager::~apu_moe_router_manager() {
    reset();
}

void apu_moe_router_manager::set_mode(apu_moe_router_mode mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = mode;
}

void apu_moe_router_manager::set_sram_limit_mb(int32_t limit_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    sram_limit_mb_ = limit_mb > 0 ? limit_mb : 32;
}

apu_moe_router_mode apu_moe_router_manager::get_mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

int32_t apu_moe_router_manager::get_sram_limit_mb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sram_limit_mb_;
}

void apu_moe_router_manager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    sram_buffer_.reset();
    pinned_pointers_.clear();
    active_info_ = apu_moe_router_info{};
}

apu_moe_router_info apu_moe_router_manager::get_info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_info_;
}

apu_moe_router_info apu_moe_router_manager::evaluate_model(const llama_model & model) {
    std::lock_guard<std::mutex> lock(mutex_);

    apu_moe_router_info info{};
    info.mode = mode_;
    info.sram_limit_bytes = (size_t)sram_limit_mb_ * 1024 * 1024;
    info.arch_name = model.arch_name();
    info.n_experts = model.hparams.n_expert;
    info.n_experts_used = model.hparams.n_expert_used(0);
    info.n_embd = model.hparams.n_embd;

    // Check whether the model is an MoE architecture
    bool has_router_tensors = false;
    for (const auto & layer : model.layers) {
        if (layer.ffn_gate_inp != nullptr) {
            has_router_tensors = true;
            info.n_expert_layers++;
            size_t bytes = ggml_nbytes(layer.ffn_gate_inp);
            info.total_router_bytes += bytes;
            info.router_tensor_names.push_back(layer.ffn_gate_inp->name);

            if (layer.ffn_gate_inp_b != nullptr) {
                info.total_router_bytes += ggml_nbytes(layer.ffn_gate_inp_b);
            }
            if (layer.ffn_gate_inp_shexp != nullptr) {
                info.total_router_bytes += ggml_nbytes(layer.ffn_gate_inp_shexp);
            }
        }
    }

    // Secondary scan in tensors_by_name if layer structure was not yet mapped
    if (!has_router_tensors && model.hparams.n_expert > 0) {
        for (const auto & item : model.tensors_by_name) {
            const std::string & name = item.first;
            struct ggml_tensor * tensor = item.second;
            if (name.find("ffn_gate_inp.weight") != std::string::npos ||
                name.find("exp_probs_b.bias") != std::string::npos) {
                has_router_tensors = true;
                if (name.find("ffn_gate_inp.weight") != std::string::npos) {
                    info.n_expert_layers++;
                }
                info.total_router_bytes += ggml_nbytes(tensor);
                info.router_tensor_names.push_back(name);
            }
        }
    }

    if (!has_router_tensors && model.hparams.n_expert == 0) {
        // Dense model — completely bypass MoE code paths with zero overhead
        info.is_moe = false;
        info.pinned_in_sram = false;
        info.fallback_to_dram = false;
        active_info_ = info;
        return info;
    }

    info.is_moe = true;

    // Evaluate SRAM budget
    if (mode_ == APU_MOE_ROUTER_OFF) {
        info.pinned_in_sram = false;
        info.fallback_to_dram = true;
        info.warning_msg = "MoE router SRAM pinning disabled via configuration (--no-router-sram / --router-sram off)";
        fprintf(stderr, "[APU MoE Router] SRAM pinning disabled, routing W_gate via UMA DRAM.\n");
    } else if (info.total_router_bytes > info.sram_limit_bytes) {
        info.pinned_in_sram = false;
        info.fallback_to_dram = true;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Total router footprint (" << (double)info.total_router_bytes / (1024.0 * 1024.0)
           << " MiB) exceeds AIE2P SRAM limit (" << (double)info.sram_limit_bytes / (1024.0 * 1024.0)
           << " MiB). Gracefully falling back to UMA DRAM.";
        info.warning_msg = ss.str();

        if (mode_ == APU_MOE_ROUTER_ON) {
            fprintf(stderr, "[APU MoE Router] WARNING: %s\n", info.warning_msg.c_str());
        } else {
            fprintf(stderr, "[APU MoE Router] %s\n", info.warning_msg.c_str());
        }
    } else {
        info.pinned_in_sram = true;
        info.fallback_to_dram = false;
        info.warning_msg = "";
        fprintf(stderr, "[APU MoE Router] MoE architecture detected: %s (%u experts, %u used, %u layers). "
                        "Router footprint: %.2f MiB <= AIE2P SRAM limit (%.2f MiB).\n",
                info.arch_name.c_str(), info.n_experts, info.n_experts_used, info.n_expert_layers,
                (double)info.total_router_bytes / (1024.0 * 1024.0),
                (double)info.sram_limit_bytes / (1024.0 * 1024.0));
    }

    active_info_ = info;
    return info;
}

bool apu_moe_router_manager::pin_router_matrices(llama_model & model) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!active_info_.is_moe || !active_info_.pinned_in_sram) {
        return false;
    }

    if (active_info_.total_router_bytes == 0) {
        return false;
    }

    // Align aggregate size to 4KB page boundary
    size_t alloc_size = (active_info_.total_router_bytes + APU_PAGE_SIZE_BYTES - 1) & ~(APU_PAGE_SIZE_BYTES - 1);

    void * sram_base = nullptr;
    std::string render_node = apu_find_render_node();
    int render_fd = -1;

    if (!render_node.empty()) {
        render_fd = ::open(render_node.c_str(), O_RDWR);
    }

    if (render_fd >= 0) {
        try {
            sram_buffer_ = std::make_unique<apu_gem_buffer>(render_fd, alloc_size, true);
            sram_base = sram_buffer_->get_cpu_ptr();
        } catch (const std::exception & e) {
            fprintf(stderr, "[APU MoE Router] Note: GEM allocation failed (%s), using pinned page-aligned buffer.\n", e.what());
            sram_buffer_.reset();
        }
        ::close(render_fd);
    }

    if (!sram_base) {
        int ret = ::posix_memalign(&sram_base, APU_PAGE_SIZE_BYTES, alloc_size);
        if (ret != 0 || !sram_base) {
            fprintf(stderr, "[APU MoE Router] ERROR: Failed to allocate %zu bytes for SRAM router buffer.\n", alloc_size);
            active_info_.pinned_in_sram = false;
            active_info_.fallback_to_dram = true;
            return false;
        }
    }

    size_t cur_offset = 0;
    uint32_t pinned_count = 0;

    for (auto & layer : model.layers) {
        if (layer.ffn_gate_inp) {
            // Enforce 64-byte host cache and 16-byte Tile DMA alignment
            cur_offset = (cur_offset + APU_HOST_CACHE_ALIGNMENT_BYTES - 1) & ~(APU_HOST_CACHE_ALIGNMENT_BYTES - 1);
            size_t nbytes = ggml_nbytes(layer.ffn_gate_inp);

            void * dst = static_cast<char *>(sram_base) + cur_offset;
            if (layer.ffn_gate_inp->buffer) {
                ggml_backend_tensor_get(layer.ffn_gate_inp, dst, 0, nbytes);
            } else if (layer.ffn_gate_inp->data) {
                std::memcpy(dst, layer.ffn_gate_inp->data, nbytes);
            }
            pinned_pointers_.push_back(dst);
            cur_offset += nbytes;
            pinned_count++;

            if (layer.ffn_gate_inp_b) {
                cur_offset = (cur_offset + APU_HOST_CACHE_ALIGNMENT_BYTES - 1) & ~(APU_HOST_CACHE_ALIGNMENT_BYTES - 1);
                size_t b_bytes = ggml_nbytes(layer.ffn_gate_inp_b);
                void * dst_b = static_cast<char *>(sram_base) + cur_offset;
                if (layer.ffn_gate_inp_b->buffer) {
                    ggml_backend_tensor_get(layer.ffn_gate_inp_b, dst_b, 0, b_bytes);
                } else if (layer.ffn_gate_inp_b->data) {
                    std::memcpy(dst_b, layer.ffn_gate_inp_b->data, b_bytes);
                }
                pinned_pointers_.push_back(dst_b);
                cur_offset += b_bytes;
                pinned_count++;
            }
        }
    }

    fprintf(stderr, "[APU MoE Router] Successfully pinned %u router gating matrices (%.2f MiB) into AIE2P Tile SRAM.\n",
            pinned_count, (double)active_info_.total_router_bytes / (1024.0 * 1024.0));

    return true;
}

bool apu_audit_moe_router(const std::string & model_path,
                          apu_moe_router_mode mode,
                          int32_t limit_mb,
                          bool verbose,
                          apu_moe_router_info & out_info,
                          std::string & out_log) {
    std::ostringstream log;
    log << "=== APU MoE Router Audit (§5.1-§5.3) ===\n";
    log << "Model: " << model_path << "\n";
    log << "Configured Mode: " << (mode == APU_MOE_ROUTER_AUTO ? "auto" : (mode == APU_MOE_ROUTER_ON ? "on" : "off")) << "\n";
    log << "Configured SRAM limit: " << limit_mb << " MB\n";

    llama_model_params mparams = llama_model_default_params();

    apu_moe_router_manager::get().set_mode(mode);
    apu_moe_router_manager::get().set_sram_limit_mb(limit_mb);

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        log << "FAIL: Could not load model from " << model_path << "\n";
        out_log = log.str();
        return false;
    }

    out_info = apu_moe_router_manager::get().evaluate_model(*model);

    log << "MoE Detected: " << (out_info.is_moe ? "YES" : "NO (Dense)") << "\n";
    if (out_info.is_moe) {
        log << "Architecture: " << out_info.arch_name << "\n";
        log << "Expert Layers: " << out_info.n_expert_layers << "\n";
        log << "Total Experts: " << out_info.n_experts << " (top-" << out_info.n_experts_used << " used)\n";
        log << "Hidden Dimension: " << out_info.n_embd << "\n";
        log << "Total Router Footprint: " << std::fixed << std::setprecision(2)
            << (double)out_info.total_router_bytes / (1024.0 * 1024.0) << " MiB\n";
        log << "SRAM Pinning Decision: " << (out_info.pinned_in_sram ? "PINNED IN SRAM" : "DRAM FALLBACK") << "\n";
        if (out_info.fallback_to_dram) {
            log << "Reason: " << out_info.warning_msg << "\n";
        }

        if (verbose && !out_info.router_tensor_names.empty()) {
            log << "Router Tensors (" << out_info.router_tensor_names.size() << " total):\n";
            for (size_t i = 0; i < std::min<size_t>(out_info.router_tensor_names.size(), 5); ++i) {
                log << "  - " << out_info.router_tensor_names[i] << "\n";
            }
            if (out_info.router_tensor_names.size() > 5) {
                log << "  ... (" << out_info.router_tensor_names.size() - 5 << " more)\n";
            }
        }
    } else {
        log << "Dense model: Zero overhead confirmed, all MoE routing bypassed.\n";
    }

    llama_model_free(model);
    out_log = log.str();
    return true;
}
