// SPDX-License-Identifier: Apache-2.0
//! C Foreign Function Interface (FFI) implementation for `apu-backend`.
//!
//! Exposes a memory-safe, clean C ABI linking `libzero_copy_model_runner.a` /
//! `libapu_backend.a` with the `llama.cpp` fork.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Arc;

use crate::backend::{open_or_mock, DeviceBackend};
use crate::container::{
    load_or_convert_model, ModelGraphTopology, ModelHyperparameters, Q4nxModel, TargetHardware,
    XclbinBuilder, XclbinFormat,
};
use crate::engine::{
    DecodeEngine, DecodeStepRequest, PrefillEngine, PrefillRequest, RocmPrefillEngine,
    Sampler, SamplerConfig, SpeculativeOrchestrator, XrtDecodeEngine,
};
use crate::memory::kv_pruning::{DynamicKvPruner, KvPruningConfig};
use crate::memory::strix_halo_tuning::{StrixHaloConfig, StrixHaloMemoryOptimizer};
use crate::memory::{MemoryBridge, SharedBuffer};

/// C ABI representation of model hyperparameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApuModelHyperparams {
    pub hidden_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
    pub vocab_size: u32,
    pub context_length: u32,
}

impl From<ModelHyperparameters> for ApuModelHyperparams {
    fn from(hp: ModelHyperparameters) -> Self {
        Self {
            hidden_dim: hp.hidden_dim,
            num_heads: hp.num_heads,
            num_kv_heads: hp.num_kv_heads,
            num_layers: hp.num_layers,
            vocab_size: hp.vocab_size,
            context_length: hp.context_length,
        }
    }
}

/// Opaque context handle managed across C ABI boundaries.
pub struct ApuBackendContext {
    pub model: Q4nxModel,
    pub backend: Arc<dyn DeviceBackend>,
    pub prefill_engine: RocmPrefillEngine,
    pub decode_engine: XrtDecodeEngine,
    pub cpu_worker: crate::engine::CpuWorkerEngine,
    pub sampler: Sampler,
    pub shared_kv: Option<SharedBuffer>,
    pub speculative: Option<SpeculativeOrchestrator>,
    pub kv_pruner: Option<DynamicKvPruner>,
    pub strix_optimizer: Option<StrixHaloMemoryOptimizer>,
    pub kv_quant_type: crate::memory::kv_quant::KvCacheQuantType,
    pub router_sram_enabled: bool,
    pub router_sram_limit_mb: usize,
    pub prefill_target: String,
    pub decode_target: String,
}

/// Load a model into the apu-backend.
///
/// Returns 0 on success, non-zero error code on failure.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_load_model(
    model_path: *const c_char,
    xclbin_override_path: *const c_char,
    out_ctx: *mut *mut ApuBackendContext,
) -> i32 {
    if model_path.is_null() || out_ctx.is_null() {
        return -1;
    }

    let model_path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let xclbin_override: Option<&Path> = if !xclbin_override_path.is_null() {
        match CStr::from_ptr(xclbin_override_path).to_str() {
            Ok(s) => Some(Path::new(s)),
            Err(_) => return -3,
        }
    } else {
        None
    };

    // Load or convert model (with embedded XCLBIN by default)
    let model = match load_or_convert_model(Path::new(model_path_str), xclbin_override) {
        Ok(m) => m,
        Err(_) => return -4,
    };

    // Initialize unified APU backend
    let backend = match open_or_mock("/dev/dri/renderD128") {
        Ok(b) => b,
        Err(_) => return -5,
    };

    // Initialize prefill and decode compute engines
    let mut prefill_engine = RocmPrefillEngine::new_with_backend(backend.clone());
    prefill_engine.set_vocab_limit(model.header.hyperparams.vocab_size.max(320_000));
    let mut decode_engine = XrtDecodeEngine::new_with_backend(backend.clone());
    let mut cpu_worker = crate::engine::CpuWorkerEngine::new();

    let gguf_source_path = if model_path_str.ends_with(".gguf") {
        Some(std::path::PathBuf::from(model_path_str))
    } else {
        let p = Path::new(model_path_str).with_extension("gguf");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };

    if let Some(gp) = gguf_source_path {
        if let Ok(reader) = crate::container::reader::GgufModelReader::open(&gp) {
            let r_arc = Arc::new(reader);
            let trans = Arc::new(std::sync::Mutex::new(Some(crate::engine::TransformerContext::new(Arc::clone(&r_arc)))));
            prefill_engine.set_shared_transformer(Arc::clone(&trans));
            decode_engine.set_shared_transformer(Arc::clone(&trans));
            cpu_worker.set_shared_transformer(Arc::clone(&trans));
        }
    }

    let sampler = Sampler::new(SamplerConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        seed: Some(42),
    });

    let ctx = Box::new(ApuBackendContext {
        model,
        backend,
        prefill_engine,
        decode_engine,
        cpu_worker,
        sampler,
        shared_kv: None,
        speculative: None,
        kv_pruner: None,
        strix_optimizer: None,
        kv_quant_type: crate::memory::kv_quant::KvCacheQuantType::Auto,
        router_sram_enabled: true,
        router_sram_limit_mb: 32,
        prefill_target: "gpu".to_string(),
        decode_target: "npu".to_string(),
    });

    *out_ctx = Box::into_raw(ctx);
    0
}

/// Query model structural hyperparameters.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_get_hyperparams(
    ctx: *const ApuBackendContext,
    out_params: *mut ApuModelHyperparams,
) -> i32 {
    if ctx.is_null() || out_params.is_null() {
        return -1;
    }

    let context = &*ctx;
    *out_params = context.model.header.hyperparams.into();
    0
}

/// Allocate and bind the shared zero-copy KV cache (Linux Prime dma-buf).
#[no_mangle]
pub unsafe extern "C" fn apu_backend_allocate_shared_kv(
    ctx: *mut ApuBackendContext,
    capacity_bytes: usize,
    out_dmabuf_fd: *mut i32,
) -> i32 {
    if ctx.is_null() || out_dmabuf_fd.is_null() || capacity_bytes == 0 {
        return -1;
    }

    let context = &mut *ctx;

    let gpu_bo = match MemoryBridge::allocate_gem_bo(&context.backend, capacity_bytes, 64) {
        Ok(bo) => bo,
        Err(_) => return -2,
    };

    let dmabuf = match gpu_bo.export_prime_fd() {
        Ok(h) => h,
        Err(_) => return -3,
    };

    let fd = dmabuf.as_raw_fd();

    let shared_buf = match SharedBuffer::new(dmabuf, true) {
        Ok(b) => b,
        Err(_) => return -4,
    };

    // Attach KV cache to decode engine
    if context.decode_engine.attach_kv_cache(shared_buf.handle()).is_err() {
        return -5;
    }

    context.shared_kv = Some(shared_buf);
    *out_dmabuf_fd = fd;
    0
}

/// Dispatch prompt prefill pass on RDNA 3.5 iGPU.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_dispatch_prefill(
    ctx: *mut ApuBackendContext,
    prompt_tokens: *const u32,
    num_tokens: usize,
    syncobj_fd: i32,
    timeline_point: u64,
    out_initial_token: *mut u32,
) -> i32 {
    if ctx.is_null() || prompt_tokens.is_null() || out_initial_token.is_null() || num_tokens == 0 {
        return -1;
    }

    let context = &mut *ctx;
    let kv_handle = match context.shared_kv.as_ref() {
        Some(sb) => sb.handle(),
        None => return -2,
    };

    let tokens_slice = std::slice::from_raw_parts(prompt_tokens, num_tokens);

    let req = PrefillRequest {
        token_ids: tokens_slice,
        start_offset: 0,
        batch_size: 1,
    };

    let res = if context.prefill_target == "cpu" {
        match context.cpu_worker.dispatch_prefill(
            req,
            kv_handle,
            syncobj_fd,
            timeline_point,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[apu_backend_prefill CPU ERROR] {:?}", e);
                return -3;
            }
        }
    } else {
        match context.prefill_engine.dispatch_prefill(
            req.clone(),
            kv_handle,
            syncobj_fd,
            timeline_point,
        ) {
            Ok(r) => r,
            Err(_) => match context.cpu_worker.dispatch_prefill(
                req,
                kv_handle,
                syncobj_fd,
                timeline_point,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[apu_backend_prefill ERROR] {:?}", e);
                    return -3;
                }
            },
        }
    };

    *out_initial_token = res.initial_token_id;
    0
}

/// Dispatch single-token autoregressive decode step on XDNA 2 NPU.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_dispatch_decode_step(
    ctx: *mut ApuBackendContext,
    input_token: u32,
    sequence_index: usize,
    temperature: f32,
    wait_syncobj_fd: i32,
    wait_timeline_point: u64,
    signal_timeline_point: u64,
    out_token: *mut u32,
    out_is_eos: *mut bool,
) -> i32 {
    if ctx.is_null() || out_token.is_null() || out_is_eos.is_null() {
        return -1;
    }

    let context = &mut *ctx;

    let req = DecodeStepRequest {
        input_token_id: input_token,
        sequence_index,
        temperature,
    };

    let step_res = if context.decode_target == "cpu" {
        match context.cpu_worker.dispatch_decode_step(
            req,
            wait_syncobj_fd,
            wait_timeline_point,
            signal_timeline_point,
        ) {
            Ok(r) => r,
            Err(_) => return -2,
        }
    } else if context.decode_target == "gpu" {
        match context.cpu_worker.dispatch_decode_step(
            req,
            wait_syncobj_fd,
            wait_timeline_point,
            signal_timeline_point,
        ) {
            Ok(r) => r,
            Err(_) => return -2,
        }
    } else {
        match context.decode_engine.dispatch_decode_step(
            req.clone(),
            wait_syncobj_fd,
            wait_timeline_point,
            signal_timeline_point,
        ) {
            Ok(r) => r,
            Err(_) => match context.cpu_worker.dispatch_decode_step(
                req,
                wait_syncobj_fd,
                wait_timeline_point,
                signal_timeline_point,
            ) {
                Ok(r) => r,
                Err(_) => return -2,
            },
        }
    };

    *out_token = step_res.output_token_id;
    *out_is_eos = step_res.is_eos;
    0
}

/// Configure stage routing targets (prefill: "gpu"|"cpu"|"npu", decode: "npu"|"gpu"|"cpu").
#[no_mangle]
pub unsafe extern "C" fn apu_backend_set_stage_routing(
    ctx: *mut ApuBackendContext,
    prefill: *const c_char,
    decode: *const c_char,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let context = &mut *ctx;
    if !prefill.is_null() {
        if let Ok(s) = CStr::from_ptr(prefill).to_str() {
            context.prefill_target = s.to_lowercase();
        }
    }
    if !decode.is_null() {
        if let Ok(s) = CStr::from_ptr(decode).to_str() {
            context.decode_target = s.to_lowercase();
        }
    }
    0
}

/// Returns 1 if running on fallback mock drivers, 0 if physical AMD APU silicon.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_is_mock(ctx: *const ApuBackendContext) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let context = &*ctx;
    if context.backend.is_mock() {
        1
    } else {
        0
    }
}

/// Returns 1 if model container has an embedded XCLBIN hardware graph, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_has_embedded_xclbin(ctx: *const ApuBackendContext) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let context = &*ctx;
    if context.model.has_embedded_xclbin() {
        1
    } else {
        0
    }
}

/// Query model architecture string (e.g. "phi3", "llama", "qwen2").
#[no_mangle]
pub unsafe extern "C" fn apu_backend_get_architecture(
    ctx: *const ApuBackendContext,
    out_arch: *mut c_char,
    max_len: usize,
) -> i32 {
    if ctx.is_null() || out_arch.is_null() || max_len == 0 {
        return -1;
    }
    let context = &*ctx;
    let arch_bytes = context.model.header.arch_name.as_bytes();
    let copy_len = arch_bytes.len().min(max_len - 1);
    std::ptr::copy_nonoverlapping(arch_bytes.as_ptr() as *const c_char, out_arch, copy_len);
    *out_arch.add(copy_len) = 0;
    0
}

/// Query embedded XCLBIN size in bytes.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_get_xclbin_size(ctx: *const ApuBackendContext) -> usize {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.model.xclbin_bytes().map(|b| b.len()).unwrap_or(0)
}

/// Query model weights payload size in bytes.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_get_payload_size(ctx: *const ApuBackendContext) -> usize {
    if ctx.is_null() {
        return 0;
    }
    let context = &*ctx;
    context.model.tensor_payload().len()
}

/// Enable speculative drafting on the active context.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_enable_speculative(
    ctx: *mut ApuBackendContext,
    draft_k: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let context = &mut *ctx;
    context.speculative = Some(SpeculativeOrchestrator::new(draft_k));
    0
}

/// Dispatch a speculative decoding step.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_dispatch_speculative_step(
    ctx: *mut ApuBackendContext,
    current_token: u32,
    _sequence_index: usize,
    out_tokens: *mut u32,
    max_tokens: usize,
    out_accepted_count: *mut usize,
    out_is_eos: *mut bool,
) -> i32 {
    if ctx.is_null() || out_tokens.is_null() || out_accepted_count.is_null() || out_is_eos.is_null() || max_tokens == 0 {
        return -1;
    }
    let context = &mut *ctx;
    let kv_handle = match context.shared_kv.as_ref() {
        Some(sb) => sb.handle(),
        None => return -2,
    };
    let orchestrator = match context.speculative.as_mut() {
        Some(o) => o,
        None => return -3,
    };
    let eos_id = 128001;
    let res = match orchestrator.step(current_token, &context.decode_engine, kv_handle, eos_id) {
        Ok(r) => r,
        Err(_) => return -4,
    };
    let count = res.accepted_tokens.len().min(max_tokens);
    for i in 0..count {
        *out_tokens.add(i) = res.accepted_tokens[i];
    }
    *out_accepted_count = count;
    *out_is_eos = res.is_eos;
    0
}

/// Configure dynamic KV-cache attention pruning (SnapKV / Sliding Window).
#[no_mangle]
pub unsafe extern "C" fn apu_backend_configure_kv_pruning(
    ctx: *mut ApuBackendContext,
    max_context_window: usize,
    sink_tokens: usize,
    keep_recent_tokens: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let context = &mut *ctx;
    context.kv_pruner = Some(DynamicKvPruner::new(KvPruningConfig {
        max_context_window,
        sink_tokens,
        keep_recent_tokens,
    }));
    0
}

/// Configure 256-bit UMA bus and 2MB huge-page memory optimizations.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_apply_strix_halo_tuning(
    ctx: *mut ApuBackendContext,
    enable_2mb_hugepages: bool,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let context = &mut *ctx;
    let optimizer = StrixHaloMemoryOptimizer::new(StrixHaloConfig {
        enable_2mb_hugepages,
        ..Default::default()
    });
    if let Some(ref mut sb) = context.shared_kv {
        if let Some(ptr) = sb.as_mut_ptr() {
            let _ = optimizer.advise_hugepages(ptr, sb.len());
        }
    }
    context.strix_optimizer = Some(optimizer);
    0
}

/// Create a custom XCLBIN hardware binary for a model and write it to disk.
///
/// target_arch can be "npu1" (AIE2 / Phoenix / Hawk Point) or "npu2" (AIE2P / Strix Point / Gorgon / Krackan / Strix Halo).
/// If NULL, defaults to "npu2".
#[no_mangle]
pub unsafe extern "C" fn apu_backend_create_xclbin_file(
    model_path: *const c_char,
    target_arch: *const c_char,
    out_xclbin_path: *const c_char,
) -> i32 {
    if model_path.is_null() || out_xclbin_path.is_null() {
        return -1;
    }

    let model_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let out_str = match CStr::from_ptr(out_xclbin_path).to_str() {
        Ok(s) => s,
        Err(_) => return -3,
    };

    let target_str = if !target_arch.is_null() {
        CStr::from_ptr(target_arch).to_str().unwrap_or("npu2")
    } else {
        "npu2"
    };
    let target = TargetHardware::from_str_loose(target_str).unwrap_or(TargetHardware::Npu2Aie2p);

    let format = if target_str.contains("mimic") {
        XclbinFormat::MimicBuiltin
    } else {
        XclbinFormat::Enhanced
    };

    let p = Path::new(model_str);
    let topo = if p.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("gguf")) {
        match ModelGraphTopology::from_gguf(p) {
            Ok(t) => t,
            Err(_) => return -4,
        }
    } else {
        match ModelGraphTopology::from_q4nx(p) {
            Ok(t) => t,
            Err(_) => return -5,
        }
    };

    let builder = XclbinBuilder::new(target, topo).with_format(format);
    if builder.build_standalone(Path::new(out_str)).is_err() {
        return -6;
    }

    0
}

/// Synthesize a custom XCLBIN hardware binary with explicit format specification.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_create_xclbin_file_formatted(
    model_path: *const c_char,
    target_arch: *const c_char,
    format_str: *const c_char,
    out_xclbin_path: *const c_char,
) -> i32 {
    if model_path.is_null() || out_xclbin_path.is_null() {
        return -1;
    }

    let model_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let out_str = match CStr::from_ptr(out_xclbin_path).to_str() {
        Ok(s) => s,
        Err(_) => return -3,
    };

    let target_str = if !target_arch.is_null() {
        CStr::from_ptr(target_arch).to_str().unwrap_or("npu2")
    } else {
        "npu2"
    };
    let target = TargetHardware::from_str_loose(target_str).unwrap_or(TargetHardware::Npu2Aie2p);

    let format = if !format_str.is_null() {
        if let Ok(f_str) = CStr::from_ptr(format_str).to_str() {
            XclbinFormat::from_str_loose(f_str).unwrap_or(XclbinFormat::Enhanced)
        } else {
            XclbinFormat::Enhanced
        }
    } else if target_str.contains("mimic") {
        XclbinFormat::MimicBuiltin
    } else {
        XclbinFormat::Enhanced
    };

    let p = Path::new(model_str);
    let topo = if p.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("gguf")) {
        match ModelGraphTopology::from_gguf(p) {
            Ok(t) => t,
            Err(_) => return -4,
        }
    } else {
        match ModelGraphTopology::from_q4nx(p) {
            Ok(t) => t,
            Err(_) => return -5,
        }
    };

    let builder = XclbinBuilder::new(target, topo).with_format(format);
    if builder.build_standalone(Path::new(out_str)).is_err() {
        return -6;
    }

    0
}

/// Create a custom XCLBIN hardware binary and embed it directly into the `.q4nx` container header.
///
/// If out_q4nx_path is NULL, updates model_path in-place (or writes <model>.q4nx if model_path was .gguf).
#[no_mangle]
pub unsafe extern "C" fn apu_backend_create_xclbin_embedded(
    model_path: *const c_char,
    target_arch: *const c_char,
    out_q4nx_path: *const c_char,
) -> i32 {
    apu_backend_create_xclbin_embedded_formatted(model_path, target_arch, std::ptr::null(), out_q4nx_path)
}

/// Create a custom XCLBIN with explicit format and embed it into the `.q4nx` container header.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_create_xclbin_embedded_formatted(
    model_path: *const c_char,
    target_arch: *const c_char,
    format_str: *const c_char,
    out_q4nx_path: *const c_char,
) -> i32 {
    if model_path.is_null() {
        return -1;
    }

    let model_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };

    let out_opt: Option<&Path> = if !out_q4nx_path.is_null() {
        match CStr::from_ptr(out_q4nx_path).to_str() {
            Ok(s) if !s.is_empty() => Some(Path::new(s)),
            _ => None,
        }
    } else {
        None
    };

    let target_str = if !target_arch.is_null() {
        CStr::from_ptr(target_arch).to_str().unwrap_or("npu2")
    } else {
        "npu2"
    };
    let target = TargetHardware::from_str_loose(target_str).unwrap_or(TargetHardware::Npu2Aie2p);

    let format = if !format_str.is_null() {
        if let Ok(f_str) = CStr::from_ptr(format_str).to_str() {
            XclbinFormat::from_str_loose(f_str).unwrap_or(XclbinFormat::Enhanced)
        } else {
            XclbinFormat::Enhanced
        }
    } else if target_str.contains("mimic") {
        XclbinFormat::MimicBuiltin
    } else {
        XclbinFormat::Enhanced
    };

    let p = Path::new(model_str);
    let topo = if p.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("gguf")) {
        match ModelGraphTopology::from_gguf(p) {
            Ok(t) => t,
            Err(_) => return -3,
        }
    } else {
        match ModelGraphTopology::from_q4nx(p) {
            Ok(t) => t,
            Err(_) => return -4,
        }
    };

    let builder = XclbinBuilder::new(target, topo).with_format(format);
    if builder.build_embedded(p, out_opt).is_err() {
        return -5;
    }

    0
}

/// Set Key-Value cache quantization mode (0=FP16, 1=INT8, 2=INT4, 3=Auto).
#[no_mangle]
pub unsafe extern "C" fn apu_backend_set_kv_quant_type(ctx: *mut ApuBackendContext, quant_type: i32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let mode = match quant_type {
        0 => crate::memory::kv_quant::KvCacheQuantType::Fp16,
        1 => crate::memory::kv_quant::KvCacheQuantType::Int8,
        2 => crate::memory::kv_quant::KvCacheQuantType::Int4,
        3 => crate::memory::kv_quant::KvCacheQuantType::Auto,
        _ => return -2,
    };
    (*ctx).kv_quant_type = mode;
    0
}

/// Configure on-chip SRAM router matrix ($W_{\text{gate}}$) pinning for MoE models.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_set_router_sram_pinning(
    ctx: *mut ApuBackendContext,
    enabled: i32,
    limit_mb: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    (*ctx).router_sram_enabled = enabled != 0;
    (*ctx).router_sram_limit_mb = if limit_mb > 0 { limit_mb } else { 32 };
    0
}

/// Free context and release all underlying devices and buffers.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_free(ctx: *mut ApuBackendContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// Run AMD Ryzen AI APU hardware diagnostics and print report to stdout.
#[no_mangle]
pub extern "C" fn apu_backend_doctor() -> i32 {
    crate::doctor::print_doctor_report();
    0
}

/// Run APU model manager (convert, stamp, info, list).
#[no_mangle]
pub unsafe extern "C" fn apu_backend_model(argc: i32, argv: *const *const c_char) -> i32 {
    if argc <= 0 || argv.is_null() {
        return crate::model_cli::run_model_cli(vec!["apu-model".to_string()]);
    }
    let mut args = Vec::with_capacity(argc as usize);
    for i in 0..argc {
        let ptr = *argv.offset(i as isize);
        if ptr.is_null() {
            continue;
        }
        if let Ok(s) = CStr::from_ptr(ptr).to_str() {
            args.push(s.to_string());
        }
    }
    crate::model_cli::run_model_cli(args)
}

/// Run XCLBIN hardware graph synthesizer.
#[no_mangle]
pub unsafe extern "C" fn apu_backend_synth(argc: i32, argv: *const *const c_char) -> i32 {
    if argc <= 0 || argv.is_null() {
        return crate::synth_cli::run_synth_cli(vec!["apu-synth".to_string()]);
    }
    let mut args = Vec::with_capacity(argc as usize);
    for i in 0..argc {
        let ptr = *argv.offset(i as isize);
        if ptr.is_null() {
            continue;
        }
        if let Ok(s) = CStr::from_ptr(ptr).to_str() {
            args.push(s.to_string());
        }
    }
    crate::synth_cli::run_synth_cli(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_c_abi_lifecycle_and_execution() {
        let temp_dir = std::env::temp_dir();
        let model_file = temp_dir.join("c_abi_test_model.q4nx");

        // Create dummy .q4nx with embedded xclbin
        let hyperparams = ModelHyperparameters {
            hidden_dim: 2048,
            num_heads: 16,
            num_kv_heads: 4,
            num_layers: 24,
            vocab_size: 32000,
            context_length: 4096,
        };
        let dummy_xclbin = b"\x7FELF_MOCK_XCLBIN";
        let dummy_weights = vec![0x11u8; 2048];

        Q4nxModel::create_container(
            &model_file,
            "llama",
            hyperparams,
            Some(dummy_xclbin),
            &dummy_weights,
        )
        .expect("Create test q4nx");

        let c_model_path = CString::new(model_file.to_str().unwrap()).unwrap();

        unsafe {
            let mut ctx: *mut ApuBackendContext = std::ptr::null_mut();

            // 1. Load model
            let rc = apu_backend_load_model(c_model_path.as_ptr(), std::ptr::null(), &mut ctx);
            assert_eq!(rc, 0, "apu_backend_load_model must succeed");
            assert!(!ctx.is_null());

            // 2. Query hyperparams
            let mut hp = ApuModelHyperparams {
                hidden_dim: 0,
                num_heads: 0,
                num_kv_heads: 0,
                num_layers: 0,
                vocab_size: 0,
                context_length: 0,
            };
            let rc_hp = apu_backend_get_hyperparams(ctx, &mut hp);
            assert_eq!(rc_hp, 0);
            assert_eq!(hp.hidden_dim, 2048);
            assert_eq!(hp.num_layers, 24);

            // 3. Allocate shared KV
            let mut dmabuf_fd = -1;
            let rc_kv = apu_backend_allocate_shared_kv(ctx, 4 * 1024 * 1024, &mut dmabuf_fd);
            assert_eq!(rc_kv, 0);
            assert!(dmabuf_fd >= 0, "dmabuf_fd must be valid");

            // 4. Prefill pass
            let tokens = [1u32, 256, 1024, 4096];
            let mut t0 = 0u32;
            let rc_prefill = apu_backend_dispatch_prefill(
                ctx,
                tokens.as_ptr(),
                tokens.len(),
                -1,
                1,
                &mut t0,
            );
            assert_eq!(rc_prefill, 0);
            assert!(t0 > 0);

            // 5. Decode step
            let mut next_tok = 0u32;
            let mut is_eos = false;
            let rc_decode = apu_backend_dispatch_decode_step(
                ctx,
                t0,
                4,
                0.0,
                -1,
                1,
                2,
                &mut next_tok,
                &mut is_eos,
            );
            assert_eq!(rc_decode, 0);
            assert!(next_tok > 0);

            // 6. Test speculative drafting ABI
            assert_eq!(apu_backend_enable_speculative(ctx, 4), 0);
            let mut spec_tokens = [0u32; 8];
            let mut accepted_count = 0usize;
            let mut spec_eos = false;
            let rc_spec = apu_backend_dispatch_speculative_step(
                ctx,
                next_tok,
                5,
                spec_tokens.as_mut_ptr(),
                8,
                &mut accepted_count,
                &mut spec_eos,
            );
            assert_eq!(rc_spec, 0);
            assert!(accepted_count > 0);

            // 7. Test KV pruning configuration ABI
            assert_eq!(apu_backend_configure_kv_pruning(ctx, 4096, 8, 1024), 0);

            // 8. Test Strix Halo tuning ABI
            assert_eq!(apu_backend_apply_strix_halo_tuning(ctx, true), 0);

            // 9. Free context
            apu_backend_free(ctx);
        }

        // Cleanup
        let _ = std::fs::remove_file(model_file);
    }

    #[test]
    fn test_c_abi_custom_xclbin_synthesis() {
        let temp_dir = std::env::temp_dir();
        let model_file = temp_dir.join("c_abi_xclbin_test.q4nx");
        let out_xclbin = temp_dir.join("c_abi_synth.xclbin");
        let out_embedded = temp_dir.join("c_abi_embedded.q4nx");

        let hyperparams = ModelHyperparameters {
            hidden_dim: 2048,
            num_heads: 8,
            num_kv_heads: 2,
            num_layers: 28,
            vocab_size: 131072,
            context_length: 4096,
        };

        Q4nxModel::create_container(
            &model_file,
            "spark2_5",
            hyperparams,
            None,
            &vec![0x33u8; 1024],
        )
        .expect("Create test q4nx");

        let c_model_path = CString::new(model_file.to_str().unwrap()).unwrap();
        let c_arch = CString::new("npu2").unwrap();
        let c_out_xclbin = CString::new(out_xclbin.to_str().unwrap()).unwrap();
        let c_out_embedded = CString::new(out_embedded.to_str().unwrap()).unwrap();

        unsafe {
            // 1. Create standalone XCLBIN
            let rc1 = apu_backend_create_xclbin_file(
                c_model_path.as_ptr(),
                c_arch.as_ptr(),
                c_out_xclbin.as_ptr(),
            );
            assert_eq!(rc1, 0, "Standalone XCLBIN synthesis via C ABI failed");
            assert!(out_xclbin.is_file());

            // 2. Embed XCLBIN into Q4NX container
            let rc2 = apu_backend_create_xclbin_embedded(
                c_model_path.as_ptr(),
                c_arch.as_ptr(),
                c_out_embedded.as_ptr(),
            );
            assert_eq!(rc2, 0, "Embedded XCLBIN synthesis via C ABI failed");
            assert!(out_embedded.is_file());

            let model = Q4nxModel::open(&out_embedded).expect("Open stamped container");
            assert!(model.has_embedded_xclbin());
        }

        let _ = std::fs::remove_file(model_file);
        let _ = std::fs::remove_file(out_xclbin);
        let _ = std::fs::remove_file(out_embedded);
    }
}
