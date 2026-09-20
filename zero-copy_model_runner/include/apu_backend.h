/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file apu_backend.h
 * @brief Public C ABI for the apu-backend zero-copy runtime for AMD Ryzen AI APUs.
 *
 * Exposes zero-copy memory allocation, RDNA 3.5 iGPU prefill, XDNA 2 NPU decode,
 * and unified .q4nx container loading to the llama.cpp fork.
 */

#ifndef APU_BACKEND_H
#define APU_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle to an active apu-backend model context. */
typedef struct ApuBackendContext ApuBackendContext;

/** Execution phase of the APU heterogeneous pipeline. */
typedef enum {
    APU_PHASE_PREFILL = 0,
    APU_PHASE_DECODE  = 1,
} ApuExecutionPhase;

/** Model hyperparameter descriptor. */
typedef struct {
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t num_layers;
    uint32_t vocab_size;
    uint32_t context_length;
} ApuModelHyperparams;

/**
 * Load a model into the apu-backend.
 *
 * If path points to a .q4nx file with an embedded XCLBIN, it loads instantly (< 100ms).
 * If path points to a bare .q4nx file, it resolves the XCLBIN and stamps it into the header.
 * If path points to a .gguf file, it repacks weights into .q4nx, embeds the XCLBIN, and writes
 * the .q4nx file to disk.
 *
 * @param model_path Path to the .q4nx or .gguf file.
 * @param xclbin_override_path Optional explicit .xclbin path (or NULL for auto-matching).
 * @param out_ctx Pointer to receive the initialized context pointer.
 * @return 0 on success, non-zero error code on failure.
 */
int apu_backend_load_model(
    const char* model_path,
    const char* xclbin_override_path,
    ApuBackendContext** out_ctx
);

/**
 * Query model hyperparameters from an active context.
 */
int apu_backend_get_hyperparams(
    const ApuBackendContext* ctx,
    ApuModelHyperparams* out_params
);

/**
 * Allocate or bind the shared zero-copy KV cache (Linux dma-buf).
 *
 * @param ctx Active apu-backend context.
 * @param capacity_bytes Total buffer capacity in bytes.
 * @param out_dmabuf_fd Pointer to receive the exported Linux Prime dma-buf file descriptor.
 * @return 0 on success, non-zero on failure.
 */
int apu_backend_allocate_shared_kv(
    ApuBackendContext* ctx,
    size_t capacity_bytes,
    int* out_dmabuf_fd
);

/**
 * Dispatch a compute-heavy prompt prefill pass on the RDNA 3.5 iGPU.
 *
 * Attention Key and Value matrices are projected directly into the shared dma-buf.
 *
 * @param ctx Active context.
 * @param prompt_tokens Array of input token IDs.
 * @param num_tokens Number of tokens in prompt sequence.
 * @param syncobj_fd Linux DRM syncobj file descriptor for timeline fence signaling.
 * @param timeline_point Monotonic timeline point to signal on completion.
 * @param out_initial_token Pointer to receive initial token prediction T_0.
 * @return 0 on success, non-zero on failure.
 */
int apu_backend_dispatch_prefill(
    ApuBackendContext* ctx,
    const uint32_t* prompt_tokens,
    size_t num_tokens,
    int syncobj_fd,
    uint64_t timeline_point,
    uint32_t* out_initial_token
);

/**
 * Dispatch a single-token autoregressive decode step on the XDNA 2 NPU.
 *
 * In-place appends new KV representations directly into shared DRAM.
 *
 * @param ctx Active context.
 * @param input_token Token emitted from previous step.
 * @param sequence_index Current position in autoregressive sequence.
 * @param temperature Temperature scaling (0.0 = argmax greedy).
 * @param wait_syncobj_fd DRM syncobj FD to wait on before execution (-1 if none).
 * @param wait_timeline_point Timeline point to wait for.
 * @param signal_timeline_point Timeline point to signal on completion.
 * @param out_token Pointer to receive the generated output token ID.
 * @param out_is_eos Pointer to receive bool indicating if EOS was emitted.
 * @return 0 on success, non-zero on failure.
 */
int apu_backend_dispatch_decode_step(
    ApuBackendContext* ctx,
    uint32_t input_token,
    size_t sequence_index,
    float temperature,
    int wait_syncobj_fd,
    uint64_t wait_timeline_point,
    uint64_t signal_timeline_point,
    uint32_t* out_token,
    bool* out_is_eos
);

/**
 * Returns 1 if running on fallback mock drivers, 0 if physical AMD APU silicon.
 */
int apu_backend_is_mock(const ApuBackendContext* ctx);

/**
 * Returns 1 if model container has an embedded XCLBIN hardware graph, 0 otherwise.
 */
int apu_backend_has_embedded_xclbin(const ApuBackendContext* ctx);

/**
 * Query model architecture string (e.g. "phi3", "llama", "qwen2").
 */
int apu_backend_get_architecture(
    const ApuBackendContext* ctx,
    char* out_arch,
    size_t max_len
);

/**
 * Query embedded XCLBIN size in bytes.
 */
size_t apu_backend_get_xclbin_size(const ApuBackendContext* ctx);

/**
 * Query model weights payload size in bytes.
 */
size_t apu_backend_get_payload_size(const ApuBackendContext* ctx);

/**
 * Enable speculative drafting on the active context.
 *
 * @param ctx Active apu-backend context.
 * @param draft_k Number of candidate tokens to draft on NPU per step (1 to 8).
 * @return 0 on success, non-zero on error.
 */
int apu_backend_enable_speculative(ApuBackendContext* ctx, size_t draft_k);

/**
 * Dispatch a speculative decoding step.
 *
 * Drafts K candidate tokens on NPU and performs batched verification on iGPU over shared DMA-BUF.
 *
 * @param ctx Active context.
 * @param current_token Current seed token ID.
 * @param sequence_index Current sequence position.
 * @param out_tokens Array to receive accepted tokens (must hold at least 8 tokens).
 * @param max_tokens Maximum capacity of out_tokens array.
 * @param out_accepted_count Pointer to receive number of accepted tokens.
 * @param out_is_eos Pointer to receive bool indicating if EOS was emitted.
 * @return 0 on success, non-zero on error.
 */
int apu_backend_dispatch_speculative_step(
    ApuBackendContext* ctx,
    uint32_t current_token,
    size_t sequence_index,
    uint32_t* out_tokens,
    size_t max_tokens,
    size_t* out_accepted_count,
    bool* out_is_eos
);

/**
 * Configure dynamic KV-cache attention pruning (SnapKV / Sliding Window).
 *
 * @param ctx Active context.
 * @param max_context_window Window threshold before pruning triggers (e.g. 4096).
 * @param sink_tokens Initial attention sink tokens to protect (e.g. 8).
 * @param keep_recent_tokens Number of most recent tokens to keep (e.g. 1024).
 * @return 0 on success, non-zero on error.
 */
int apu_backend_configure_kv_pruning(
    ApuBackendContext* ctx,
    size_t max_context_window,
    size_t sink_tokens,
    size_t keep_recent_tokens
);

/**
 * Configure 256-bit UMA bus and 2MB huge-page memory optimizations.
 *
 * @param ctx Active context.
 * @param enable_2mb_hugepages Whether to apply MADV_HUGEPAGE hints to DMA-BUF mappings.
 * @return 0 on success, non-zero on error.
 */
int apu_backend_apply_strix_halo_tuning(
    ApuBackendContext* ctx,
    bool enable_2mb_hugepages
);

/**
 * Synthesize a custom XCLBIN hardware binary for a model and write it to disk.
 *
 * @param model_path Path to the input .gguf or .q4nx model file.
 * @param target_arch Target hardware architecture ("npu1" for Phoenix/Hawk Point, "npu2" for Strix Point/Krackan/Gorgon/Strix Halo, or NULL for auto/npu2).
 * @param out_xclbin_path Path to write the synthesized .xclbin binary file.
 * @return 0 on success, non-zero on error.
 */
int apu_backend_create_xclbin_file(
    const char* model_path,
    const char* target_arch,
    const char* out_xclbin_path
);

/**
 * Synthesize a custom XCLBIN hardware binary with explicit format specification.
 *
 * @param model_path Path to the input .gguf or .q4nx model file.
 * @param target_arch Target hardware architecture ("npu1", "npu2", or NULL).
 * @param format_str Format style ("enhanced" or "mimic-builtin").
 * @param out_xclbin_path Path to write the synthesized .xclbin binary file.
 * @return 0 on success, non-zero on error.
 */
int apu_backend_create_xclbin_file_formatted(
    const char* model_path,
    const char* target_arch,
    const char* format_str,
    const char* out_xclbin_path
);

/**
 * Synthesize a custom XCLBIN hardware binary and embed it directly into the .q4nx container header.
 *
 * @param model_path Path to the input .gguf or .q4nx model file.
 * @param target_arch Target hardware architecture ("npu1" or "npu2", or NULL for auto/npu2).
 * @param out_q4nx_path Optional output .q4nx path (or NULL to stamp in-place / default destination).
 * @return 0 on success, non-zero on error.
 */
int apu_backend_create_xclbin_embedded(
    const char* model_path,
    const char* target_arch,
    const char* out_q4nx_path
);

/**
 * Synthesize a custom XCLBIN hardware binary with explicit format and embed into .q4nx header.
 *
 * @param model_path Path to the input .gguf or .q4nx model file.
 * @param target_arch Target hardware architecture ("npu1" or "npu2", or NULL).
 * @param format_str Format style ("enhanced" or "mimic-builtin").
 * @param out_q4nx_path Optional output .q4nx path.
 * @return 0 on success, non-zero on error.
 */
int apu_backend_create_xclbin_embedded_formatted(
    const char* model_path,
    const char* target_arch,
    const char* format_str,
    const char* out_q4nx_path
);

/**
 * Destroy context and cleanly release all hardware rings, mappings, and file descriptors.
 */
void apu_backend_free(ApuBackendContext* ctx);

/**
 * Run AMD Ryzen AI APU hardware diagnostics and print report to stdout.
 *
 * @return 0 on success, non-zero on failure.
 */
int apu_backend_doctor(void);

/**
 * Run APU model manager CLI (convert, stamp, info, list).
 *
 * @param argc Number of command line arguments.
 * @param argv Array of argument strings.
 * @return Exit status code (0 for success).
 */
int apu_backend_model(int argc, const char ** argv);

/**
 * Run XCLBIN hardware graph synthesizer CLI.
 *
 * @param argc Number of command line arguments.
 * @param argv Array of argument strings.
 * @return Exit status code (0 for success).
 */
int apu_backend_synth(int argc, const char ** argv);

#ifdef __cplusplus
}
#endif

#endif /* APU_BACKEND_H */
