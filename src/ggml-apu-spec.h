#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

enum apu_spec_mode {
    APU_SPEC_AUTO = 0,
    APU_SPEC_ON   = 1,
    APU_SPEC_OFF  = 2,
};

struct apu_spec_stats {
    uint64_t total_draft_tokens    = 0;
    uint64_t accepted_draft_tokens = 0;
    uint64_t rejected_draft_tokens = 0;
    uint64_t draft_batches         = 0;
    uint64_t rollback_events       = 0;
    double   acceptance_rate       = 0.0; // alpha = accepted / total_draft
    uint64_t timeline_sync_passes  = 0;
    int64_t  sync_latency_us       = 0;   // cumulative sync latency in us
    bool     fallback_occurred     = false;
    std::string last_fallback_reason;
};

class apu_spec_coordinator {
public:
    static apu_spec_coordinator & get();

    void set_mode(apu_spec_mode mode);
    apu_spec_mode get_mode() const;

    void set_timeline_sync(bool enabled);
    bool get_timeline_sync() const;

    // Check if speculative coordination should run
    bool is_active() const;

    // Record candidate tokens drafted
    void record_draft_batch(uint32_t n_draft);

    // Record target verification results (accepted vs drafted tokens in batch)
    void record_acceptance(uint32_t n_accepted, uint32_t n_draft);

    // Record KV cache rollback event
    void record_rollback(uint32_t n_rolled_back);

    // DRM timeline synchronization point between candidate drafting and target verification pass
    bool synchronize_draft_to_target();

    // Trigger graceful fallback to non-speculative execution
    void record_fallback(const std::string & reason);

    // Telemetry retrieval & reset
    apu_spec_stats get_stats() const;
    void reset();

    // Formatted telemetry output
    std::string format_summary() const;
    void print_summary() const;

private:
    apu_spec_coordinator();
    ~apu_spec_coordinator();

    apu_spec_coordinator(const apu_spec_coordinator &) = delete;
    apu_spec_coordinator & operator=(const apu_spec_coordinator &) = delete;

    void ensure_drm_syncobj_initialized();

    mutable std::mutex mutex_;
    apu_spec_mode mode_         = APU_SPEC_AUTO;
    bool timeline_sync_enabled_ = true;
    apu_spec_stats stats_;

    int drm_fd_                 = -1;
    uint32_t syncobj_handle_    = 0;
    uint64_t timeline_point_    = 0;
    bool drm_available_         = false;
};
