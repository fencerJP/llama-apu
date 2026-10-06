// SPDX-License-Identifier: Apache-2.0
// llama-apu: Phase 9 §9.2 Cross-Accelerator Dispatch — single-producer DRM
// timeline sequencer + dedicated dispatch loop thread.
//
// Spec: drm_syncobj timeline points signal in strict monotonic order
// (P, P+1, ...). One dedicated dispatch thread owns the counter and issues
// execution calls sequentially. Pipeline per point P:
//   1. iGPU prefill chunk K + system-scope writeback barrier into GEM pool
//   2. AMDGPU signals timeline point P
//   3. XDNA NPU hardware-waits on P before decode (ERT ring)
// Fallbacks: userspace DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT polling worker,
// ERT dummy-signal flush on timeout with isolated slot teardown, GEM/XRT
// pre-pinning at init, DMA_BUF_IOCTL_SYNC CPU bracketing.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "llama-apu-scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_apu_dispatch_config_t {
    uint64_t wait_timeout_ns;      // per-point NPU wait timeout (default 5s)
    int      poll_cpu;             // userspace fallback worker affinity (-1 = none)
    int      enable_worker;        // start dedicated loop thread (default 1)
    int      prepin_gem;           // pre-pin GEM pool at init (default 1)
    size_t   prepin_bytes;         // pre-pin size (default 64 KiB)
} llama_apu_dispatch_config_t;

typedef struct llama_apu_dispatch_stats_t {
    uint64_t enqueued;
    uint64_t signaled;
    uint64_t waited;
    uint64_t timeouts;
    uint64_t ert_flushes;
    uint64_t fallbacks;            // cpu-fallback (no DRM) submissions
    uint64_t sync_failures;        // monotonicity / ioctl failures
    uint64_t last_signal_point;
    uint64_t last_wait_point;
    double   avg_signal_lat_us;
    double   avg_wait_lat_us;
    int      drm_available;
    int      accel_available;
    int      worker_running;
    char     mode[32];             // "hardware-timeline" | "cpu-fallback"
} llama_apu_dispatch_stats_t;

typedef struct llama_apu_dispatch_ctx_t llama_apu_dispatch_ctx_t;

llama_apu_dispatch_ctx_t * llama_apu_dispatch_init(const llama_apu_dispatch_config_t * config);
void llama_apu_dispatch_free(llama_apu_dispatch_ctx_t * ctx);

// Enqueue a batch for async dispatch; assigns next monotonic point P.
// Returns LLAMA_APU_SUCCESS with *out_signal_point = P.
int llama_apu_dispatch_enqueue(llama_apu_dispatch_ctx_t * ctx,
                               const llama_apu_batch_t * batch,
                               uint64_t * out_signal_point);

// Synchronous submit: barrier + signal(P) + NPU wait.
// Honors monotonicity: signal_point must exceed all prior points.
int llama_apu_dispatch_submit_sync(llama_apu_dispatch_ctx_t * ctx,
                                   const llama_apu_batch_t * batch,
                                   uint64_t signal_point,
                                   uint64_t wait_point);

// Wait for a previously signaled point (userspace fallback path).
int llama_apu_dispatch_wait_point(llama_apu_dispatch_ctx_t * ctx,
                                   uint64_t point,
                                   uint64_t timeout_ns);

// ERT flush: dummy re-signal of point P to release a hung NPU wait ring.
int llama_apu_dispatch_ert_flush(llama_apu_dispatch_ctx_t * ctx, uint64_t point);

// Memory-visibility barriers (§9.2): iGPU system-scope writeback + NPU
// Tile DMA invalidation. Host-portable (seq_cst fence + CLFLUSH on x86-64).
void llama_apu_dispatch_writeback_barrier(void);
void llama_apu_dispatch_tile_invalidate(const void * ptr, size_t bytes);

int llama_apu_dispatch_get_stats(const llama_apu_dispatch_ctx_t * ctx,
                                 llama_apu_dispatch_stats_t * out);
void llama_apu_dispatch_reset_stats(llama_apu_dispatch_ctx_t * ctx);

// Process-wide singleton used by llama_apu_submit_hybrid_batch so the
// server advisory mirror exercises the real timeline path without API churn.
int llama_apu_dispatch_submit_shared(const llama_apu_batch_t * batch,
                                     uint64_t signal_point,
                                     uint64_t wait_point);
int llama_apu_dispatch_shared_stats(llama_apu_dispatch_stats_t * out);

#ifdef __cplusplus
}
#endif
