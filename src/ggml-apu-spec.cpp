#include "ggml-apu-spec.h"
#include "ggml-apu-bridge.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <drm/drm.h>

#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

apu_spec_coordinator & apu_spec_coordinator::get() {
    static apu_spec_coordinator instance;
    return instance;
}

apu_spec_coordinator::apu_spec_coordinator() {
    ensure_drm_syncobj_initialized();
}

apu_spec_coordinator::~apu_spec_coordinator() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (syncobj_handle_ != 0 && drm_fd_ >= 0) {
        struct drm_syncobj_destroy req{};
        req.handle = syncobj_handle_;
        ::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_DESTROY, &req);
        syncobj_handle_ = 0;
    }
    if (drm_fd_ >= 0) {
        ::close(drm_fd_);
        drm_fd_ = -1;
    }
}

void apu_spec_coordinator::ensure_drm_syncobj_initialized() {
    if (drm_fd_ >= 0) {
        return;
    }

    drm_fd_ = ::open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (drm_fd_ < 0) {
        drm_available_ = false;
        return;
    }

    struct drm_syncobj_create req{};
    req.flags = 0;
    if (::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_CREATE, &req) == 0) {
        syncobj_handle_ = req.handle;
        drm_available_  = true;
    } else {
        ::close(drm_fd_);
        drm_fd_         = -1;
        drm_available_  = false;
    }
}

void apu_spec_coordinator::set_mode(apu_spec_mode mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = mode;
}

apu_spec_mode apu_spec_coordinator::get_mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

void apu_spec_coordinator::set_timeline_sync(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    timeline_sync_enabled_ = enabled;
}

bool apu_spec_coordinator::get_timeline_sync() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timeline_sync_enabled_;
}

bool apu_spec_coordinator::is_active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_ != APU_SPEC_OFF;
}

void apu_spec_coordinator::record_draft_batch(uint32_t n_draft) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_draft_tokens += n_draft;
    stats_.draft_batches++;
}

void apu_spec_coordinator::record_acceptance(uint32_t n_accepted, uint32_t n_draft) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.accepted_draft_tokens += n_accepted;
    if (n_draft > n_accepted) {
        uint32_t n_rejected = n_draft - n_accepted;
        stats_.rejected_draft_tokens += n_rejected;
        stats_.rollback_events++;
    }
    if (stats_.total_draft_tokens > 0) {
        stats_.acceptance_rate = static_cast<double>(stats_.accepted_draft_tokens) /
                                 static_cast<double>(stats_.total_draft_tokens);
    }
}

void apu_spec_coordinator::record_rollback(uint32_t n_rolled_back) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.rollback_events++;
    stats_.rejected_draft_tokens += n_rolled_back;
}

bool apu_spec_coordinator::synchronize_draft_to_target() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == APU_SPEC_OFF || !timeline_sync_enabled_) {
        return true;
    }

    if (!drm_available_ || drm_fd_ < 0 || syncobj_handle_ == 0) {
        ensure_drm_syncobj_initialized();
        if (!drm_available_) {
            return false;
        }
    }

    auto t_start = std::chrono::steady_clock::now();
    timeline_point_++;

    // 1. Signal timeline point from draft stage
    uint64_t handle64 = syncobj_handle_;
    uint64_t point = timeline_point_;
    struct drm_syncobj_timeline_array sig{};
    sig.handles       = reinterpret_cast<uint64_t>(&handle64);
    sig.points        = reinterpret_cast<uint64_t>(&point);
    sig.count_handles = 1;

    if (::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL, &sig) != 0) {
        return false;
    }

    // 2. Wait timeline point for target verification dispatch (500ms timeout)
    struct drm_syncobj_timeline_wait req{};
    req.handles       = reinterpret_cast<uint64_t>(&handle64);
    req.points        = reinterpret_cast<uint64_t>(&point);
    req.timeout_nsec  = 500000000LL; // 500ms
    req.count_handles = 1;
    req.flags         = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;

    if (::ioctl(drm_fd_, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &req) != 0) {
        return false;
    }

    auto t_end = std::chrono::steady_clock::now();
    int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

    stats_.timeline_sync_passes++;
    stats_.sync_latency_us += elapsed_us;
    return true;
}

void apu_spec_coordinator::record_fallback(const std::string & reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.fallback_occurred   = true;
    stats_.last_fallback_reason = reason;
}

apu_spec_stats apu_spec_coordinator::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void apu_spec_coordinator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = apu_spec_stats();
    timeline_point_ = 0;
}

std::string apu_spec_coordinator::format_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;
    ss << "\n================================================================================\n";
    ss << "               APU SPECULATIVE DECODING TELEMETRY (Phase 6)\n";
    ss << "================================================================================\n";
    ss << " Mode                   : " << (mode_ == APU_SPEC_AUTO ? "AUTO" : (mode_ == APU_SPEC_ON ? "ON" : "OFF")) << "\n";
    ss << " DRM Timeline Sync      : " << (timeline_sync_enabled_ ? (drm_available_ ? "ENABLED (DRM /dev/dri/renderD128)" : "DISABLED (DRM unavailable)") : "OFF") << "\n";
    ss << " Draft Batches          : " << stats_.draft_batches << "\n";
    ss << " Candidate Tokens Drafted: " << stats_.total_draft_tokens << "\n";
    ss << " Candidate Tokens Accepted: " << stats_.accepted_draft_tokens << "\n";
    ss << " Candidate Tokens Rejected: " << stats_.rejected_draft_tokens << "\n";
    ss << std::fixed << std::setprecision(2);
    ss << " Acceptance Rate (alpha): " << (stats_.acceptance_rate * 100.0) << " %\n";
    ss << " KV Cache Rollback Passes: " << stats_.rollback_events << "\n";
    ss << " Timeline Sync Passes   : " << stats_.timeline_sync_passes << "\n";
    if (stats_.timeline_sync_passes > 0) {
        double avg_lat = static_cast<double>(stats_.sync_latency_us) / static_cast<double>(stats_.timeline_sync_passes);
        ss << std::fixed << std::setprecision(2);
        ss << " Avg Sync Latency       : " << avg_lat << " us\n";
    }
    ss << " Fallback Triggered     : " << (stats_.fallback_occurred ? ("YES (" + stats_.last_fallback_reason + ")") : "NO") << "\n";
    ss << "================================================================================\n";
    return ss.str();
}

void apu_spec_coordinator::print_summary() const {
    std::cout << format_summary() << std::endl;
}
