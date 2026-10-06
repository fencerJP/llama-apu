// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 9 §9.2 Cross-Accelerator Dispatch implementation.
// Single-producer DRM timeline sequencer + dedicated dispatch loop thread.
#include "llama-apu-dispatch.h"
#include "ggml-apu-bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <queue>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <drm/drm.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define APU_DISPATCH_X86 1
#endif

#define APU_DISPATCH_DEFAULT_TIMEOUT_NS 5000000000ULL
#define APU_DISPATCH_DEFAULT_PREPIN     65536u
#define APU_DISPATCH_MAX_LAT_SAMPLES    1024u

struct dispatch_item {
    llama_apu_batch_t batch;
    uint64_t point = 0;
    bool signaled = false;
};

struct llama_apu_dispatch_ctx_t {
    llama_apu_dispatch_config_t cfg{};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::queue<std::shared_ptr<dispatch_item>> pending_;
    bool stop = false;
    int drm_fd = -1;
    std::unique_ptr<apu_drm_syncobj> syncobj;
    std::unique_ptr<apu_gem_buffer> prepin;
    std::string render_node;
    std::string accel_node;
    int drm_available = 0;
    int accel_available = 0;
    uint64_t next_point = 1;
    uint64_t last_signal = 0;
    uint64_t last_wait = 0;
    llama_apu_dispatch_stats_t stats{};
    std::vector<double> sig_lat_;
    std::vector<double> wait_lat_;
};

static void dispatch_stat_push(std::vector<double> & v, double x) {
    v.push_back(x);
    if (v.size() > APU_DISPATCH_MAX_LAT_SAMPLES) {
        v.erase(v.begin(), v.begin() + (v.size() - APU_DISPATCH_MAX_LAT_SAMPLES));
    }
}

static double dispatch_stat_avg(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / (double) v.size();
}

static void dispatch_worker_loop(llama_apu_dispatch_ctx_t * ctx) {
    if (ctx->cfg.poll_cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET((unsigned) ctx->cfg.poll_cpu, &set);
        (void) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    for (;;) {
        std::shared_ptr<dispatch_item> item;
        {
            std::unique_lock<std::mutex> lock(ctx->mutex_);
            ctx->cv_.wait(lock, [&] { return ctx->stop || !ctx->pending_.empty(); });
            if (ctx->stop && ctx->pending_.empty()) return;
            item = ctx->pending_.front();
            ctx->pending_.pop();
            if (item->signaled) continue; // flushed synchronously already
        }
        // Signal under mutex to preserve single-producer monotonic order.
        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(ctx->mutex_);
            if (!item->signaled && ctx->syncobj && item->point > ctx->last_signal) {
                auto t0 = std::chrono::steady_clock::now();
                ok = ctx->syncobj->signal_timeline(item->point);
                auto t1 = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                if (ok) {
                    ctx->last_signal = item->point;
                    ctx->stats.signaled++;
                    ctx->stats.last_signal_point = item->point;
                    dispatch_stat_push(ctx->sig_lat_, us);
                    ctx->stats.avg_signal_lat_us = dispatch_stat_avg(ctx->sig_lat_);
                    item->signaled = true;
                } else {
                    ctx->stats.sync_failures++;
                }
            } else if (item->signaled) {
                ok = true;
            } else {
                ctx->stats.sync_failures++;
            }
        }
        (void) ok;
    }
}

llama_apu_dispatch_ctx_t * llama_apu_dispatch_init(const llama_apu_dispatch_config_t * config) {
    llama_apu_dispatch_ctx_t * ctx = new (std::nothrow) llama_apu_dispatch_ctx_t();
    if (!ctx) return nullptr;
    if (config) {
        ctx->cfg = *config;
    } else {
        ctx->cfg.wait_timeout_ns = APU_DISPATCH_DEFAULT_TIMEOUT_NS;
        ctx->cfg.poll_cpu = -1;
        ctx->cfg.enable_worker = 1;
        ctx->cfg.prepin_gem = 1;
        ctx->cfg.prepin_bytes = APU_DISPATCH_DEFAULT_PREPIN;
    }
    if (ctx->cfg.wait_timeout_ns == 0) ctx->cfg.wait_timeout_ns = APU_DISPATCH_DEFAULT_TIMEOUT_NS;
    if (ctx->cfg.prepin_bytes == 0) ctx->cfg.prepin_bytes = APU_DISPATCH_DEFAULT_PREPIN;

    ctx->render_node = apu_find_render_node();
    ctx->accel_node = apu_find_accel_node();
    ctx->drm_available = ctx->render_node.empty() ? 0 : 1;
    ctx->accel_available = ctx->accel_node.empty() ? 0 : 1;

    if (ctx->drm_available) {
        ctx->drm_fd = ::open(ctx->render_node.c_str(), O_RDWR | O_CLOEXEC);
        if (ctx->drm_fd < 0) {
            ctx->drm_available = 0;
        } else {
            try {
                ctx->syncobj = std::make_unique<apu_drm_syncobj>(ctx->drm_fd, false);
            } catch (...) {
                ctx->syncobj.reset();
                ctx->drm_available = 0;
            }
            if (ctx->drm_available && ctx->cfg.prepin_gem) {
                try {
                    ctx->prepin = std::make_unique<apu_gem_buffer>(ctx->drm_fd, ctx->cfg.prepin_bytes, true);
                } catch (...) {
                    ctx->prepin.reset(); // pre-pin best-effort; dispatch still works
                }
            }
        }
    }
    if (ctx->drm_available && ctx->syncobj) {
        std::snprintf(ctx->stats.mode, sizeof(ctx->stats.mode), "hardware-timeline");
    } else {
        std::snprintf(ctx->stats.mode, sizeof(ctx->stats.mode), "cpu-fallback");
    }
    ctx->stats.drm_available = ctx->drm_available;
    ctx->stats.accel_available = ctx->accel_available;

    if (ctx->cfg.enable_worker) {
        try {
            ctx->worker_ = std::thread(dispatch_worker_loop, ctx);
            ctx->stats.worker_running = 1;
        } catch (...) {
            ctx->stats.worker_running = 0;
        }
    }
    return ctx;
}

void llama_apu_dispatch_free(llama_apu_dispatch_ctx_t * ctx) {
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lock(ctx->mutex_);
        ctx->stop = true;
    }
    ctx->cv_.notify_all();
    if (ctx->worker_.joinable()) ctx->worker_.join();
    ctx->syncobj.reset();
    ctx->prepin.reset();
    if (ctx->drm_fd >= 0) {
        ::close(ctx->drm_fd);
        ctx->drm_fd = -1;
    }
    delete ctx;
}

int llama_apu_dispatch_enqueue(llama_apu_dispatch_ctx_t * ctx,
                               const llama_apu_batch_t * batch,
                               uint64_t * out_signal_point) {
    if (!ctx || !batch || !out_signal_point) return LLAMA_APU_ERR_INVALID_SLOT;
    if (batch->num_decode_slots > LLAMA_APU_MAX_BATCH_SLOTS) return LLAMA_APU_ERR_INVALID_SLOT;
    if (batch->num_prefill_chunks > LLAMA_APU_MAX_BATCH_SLOTS) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    uint64_t p = ctx->next_point++;
    auto item = std::make_shared<dispatch_item>();
    item->batch = *batch;
    item->point = p;
    ctx->pending_.push(item);
    ctx->stats.enqueued++;
    *out_signal_point = p;
    ctx->cv_.notify_one();
    return LLAMA_APU_SUCCESS;
}

static int dispatch_signal_locked(llama_apu_dispatch_ctx_t * ctx, uint64_t point) {
    // Caller holds ctx->mutex_. Performs writeback barrier + timeline signal.
    if (point <= ctx->last_signal) return LLAMA_APU_ERR_SYNC_FAIL;
    llama_apu_dispatch_writeback_barrier();
    if (!ctx->drm_available || !ctx->syncobj) return LLAMA_APU_ERR_SYNC_FAIL;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = ctx->syncobj->signal_timeline(point);
    auto t1 = std::chrono::steady_clock::now();
    if (!ok) {
        ctx->stats.sync_failures++;
        return LLAMA_APU_ERR_SYNC_FAIL;
    }
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    ctx->last_signal = point;
    ctx->stats.signaled++;
    ctx->stats.last_signal_point = point;
    dispatch_stat_push(ctx->sig_lat_, us);
    ctx->stats.avg_signal_lat_us = dispatch_stat_avg(ctx->sig_lat_);
    return LLAMA_APU_SUCCESS;
}

int llama_apu_dispatch_submit_sync(llama_apu_dispatch_ctx_t * ctx,
                                   const llama_apu_batch_t * batch,
                                   uint64_t signal_point,
                                   uint64_t wait_point) {
    if (!ctx || !batch) return LLAMA_APU_ERR_INVALID_SLOT;
    if (batch->num_decode_slots > LLAMA_APU_MAX_BATCH_SLOTS) return LLAMA_APU_ERR_INVALID_SLOT;
    if (batch->num_prefill_chunks > LLAMA_APU_MAX_BATCH_SLOTS) return LLAMA_APU_ERR_INVALID_SLOT;
    if (signal_point == 0) return LLAMA_APU_ERR_INVALID_SLOT;

    // CPU fallback when no DRM device: validate + count, no hardware dispatch.
    if (!ctx->drm_available || !ctx->syncobj) {
        std::lock_guard<std::mutex> lock(ctx->mutex_);
        if (signal_point <= ctx->last_signal) return LLAMA_APU_ERR_SYNC_FAIL;
        ctx->last_signal = signal_point;
        ctx->last_wait = wait_point;
        ctx->stats.fallbacks++;
        ctx->stats.last_signal_point = signal_point;
        ctx->stats.last_wait_point = wait_point;
        if (signal_point >= ctx->next_point) ctx->next_point = signal_point + 1;
        return LLAMA_APU_SUCCESS;
    }

    // Flush pending async items older than this point to preserve order.
    {
        std::lock_guard<std::mutex> lock(ctx->mutex_);
        if (signal_point <= ctx->last_signal) return LLAMA_APU_ERR_SYNC_FAIL;
        while (!ctx->pending_.empty()) {
            auto item = ctx->pending_.front();
            if (item->point >= signal_point || item->signaled) {
                if (item->signaled) ctx->pending_.pop();
                break;
            }
            ctx->pending_.pop();
            int rc = dispatch_signal_locked(ctx, item->point);
            item->signaled = (rc == LLAMA_APU_SUCCESS);
            if (rc != LLAMA_APU_SUCCESS) return rc;
        }
        int rc = dispatch_signal_locked(ctx, signal_point);
        if (rc != LLAMA_APU_SUCCESS) return rc;
        if (signal_point >= ctx->next_point) ctx->next_point = signal_point + 1;
    }

    // NPU hardware wait (outside lock so the worker can keep signaling).
    auto t0 = std::chrono::steady_clock::now();
    bool ok = ctx->syncobj->wait_timeline(wait_point, (int64_t) ctx->cfg.wait_timeout_ns);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    {
        std::lock_guard<std::mutex> lock(ctx->mutex_);
        if (ok) {
            ctx->stats.waited++;
            ctx->stats.last_wait_point = wait_point;
            if (wait_point > ctx->last_wait) ctx->last_wait = wait_point;
            dispatch_stat_push(ctx->wait_lat_, us);
            ctx->stats.avg_wait_lat_us = dispatch_stat_avg(ctx->wait_lat_);
            return LLAMA_APU_SUCCESS;
        }
        ctx->stats.timeouts++;
    }
    // Timeout: ERT dummy-signal flush to release the hung NPU wait ring.
    (void) llama_apu_dispatch_ert_flush(ctx, wait_point);
    return LLAMA_APU_ERR_TIMEOUT;
}

int llama_apu_dispatch_wait_point(llama_apu_dispatch_ctx_t * ctx,
                                  uint64_t point,
                                  uint64_t timeout_ns) {
    if (!ctx || point == 0) return LLAMA_APU_ERR_INVALID_SLOT;
    if (!ctx->drm_available || !ctx->syncobj) return LLAMA_APU_ERR_SYNC_FAIL;
    if (timeout_ns == 0) timeout_ns = ctx->cfg.wait_timeout_ns;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = ctx->syncobj->wait_timeline(point, (int64_t) timeout_ns);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    if (ok) {
        ctx->stats.waited++;
        ctx->stats.last_wait_point = point;
        if (point > ctx->last_wait) ctx->last_wait = point;
        dispatch_stat_push(ctx->wait_lat_, us);
        ctx->stats.avg_wait_lat_us = dispatch_stat_avg(ctx->wait_lat_);
        return LLAMA_APU_SUCCESS;
    }
    ctx->stats.timeouts++;
    return LLAMA_APU_ERR_TIMEOUT;
}

int llama_apu_dispatch_ert_flush(llama_apu_dispatch_ctx_t * ctx, uint64_t point) {
    if (!ctx || point == 0) return LLAMA_APU_ERR_INVALID_SLOT;
    if (!ctx->drm_available || !ctx->syncobj) return LLAMA_APU_ERR_SYNC_FAIL;
    // Dummy re-signal releases an ERT microcontroller stuck in hardware wait.
    bool ok = ctx->syncobj->signal_timeline(point);
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    if (ok) {
        ctx->stats.ert_flushes++;
        return LLAMA_APU_SUCCESS;
    }
    ctx->stats.sync_failures++;
    return LLAMA_APU_ERR_SYNC_FAIL;
}

void llama_apu_dispatch_writeback_barrier(void) {
    // iGPU system-scope writeback barrier (host-portable approximation):
    // full sequentially-consistent fence so prior KV writes are visible
    // to NPU Tile DMA via the IOMMU before fence completion.
    std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(APU_DISPATCH_X86)
    _mm_sfence();
#endif
}

void llama_apu_dispatch_tile_invalidate(const void * ptr, size_t bytes) {
    // NPU Tile DMA cache invalidation (host-portable approximation):
    // flush CPU cachelines covering the range, then fence.
    if (ptr && bytes) {
#if defined(APU_DISPATCH_X86)
        const char * p = (const char *) ptr;
        const char * end = p + bytes;
        uintptr_t line = (uintptr_t) p & ~((uintptr_t) 63);
        for (; (const char *) line < end; line += 64) {
            _mm_clflush((const void *) line);
        }
        _mm_sfence();
#else
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    } else {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}

int llama_apu_dispatch_get_stats(const llama_apu_dispatch_ctx_t * ctx,
                                 llama_apu_dispatch_stats_t * out) {
    if (!ctx || !out) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    *out = ctx->stats;
    out->worker_running = ctx->worker_.joinable() ? 1 : 0;
    return LLAMA_APU_SUCCESS;
}

void llama_apu_dispatch_reset_stats(llama_apu_dispatch_ctx_t * ctx) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lock(ctx->mutex_);
    int drm = ctx->stats.drm_available;
    int accel = ctx->stats.accel_available;
    int worker = ctx->stats.worker_running;
    char mode[32];
    std::memcpy(mode, ctx->stats.mode, sizeof(mode));
    ctx->stats = llama_apu_dispatch_stats_t{};
    ctx->stats.drm_available = drm;
    ctx->stats.accel_available = accel;
    ctx->stats.worker_running = worker;
    std::memcpy(ctx->stats.mode, mode, sizeof(mode));
    ctx->sig_lat_.clear();
    ctx->wait_lat_.clear();
}

// Process-wide singleton: one dispatch context per process, so the server
// advisory mirror and CLI tests share the monotonic timeline.
static std::mutex g_shared_mutex;
static llama_apu_dispatch_ctx_t * g_shared_ctx = nullptr;

int llama_apu_dispatch_submit_shared(const llama_apu_batch_t * batch,
                                     uint64_t signal_point,
                                     uint64_t wait_point) {
    if (!batch) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(g_shared_mutex);
    if (!g_shared_ctx) {
        llama_apu_dispatch_config_t cfg{};
        cfg.wait_timeout_ns = APU_DISPATCH_DEFAULT_TIMEOUT_NS;
        cfg.poll_cpu = -1;
        cfg.enable_worker = 1;
        cfg.prepin_gem = 1;
        cfg.prepin_bytes = APU_DISPATCH_DEFAULT_PREPIN;
        g_shared_ctx = llama_apu_dispatch_init(&cfg);
        if (!g_shared_ctx) return LLAMA_APU_ERR_OOM;
    }
    // Re-entrant: submit_sync takes ctx mutex, not the shared mutex. Copy
    // pointers under shared lock, then call outside it.
    llama_apu_dispatch_ctx_t * ctx = g_shared_ctx;
    // NOTE: intentionally not holding g_shared_mutex across submit to avoid
    // serializing unrelated callers beyond the ctx-level single producer.
    return llama_apu_dispatch_submit_sync(ctx, batch, signal_point, wait_point);
}

int llama_apu_dispatch_shared_stats(llama_apu_dispatch_stats_t * out) {
    if (!out) return LLAMA_APU_ERR_INVALID_SLOT;
    std::lock_guard<std::mutex> lock(g_shared_mutex);
    if (!g_shared_ctx) return LLAMA_APU_ERR_INVALID_SLOT;
    return llama_apu_dispatch_get_stats(g_shared_ctx, out);
}
