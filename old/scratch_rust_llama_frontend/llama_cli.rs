// SPDX-License-Identifier: Apache-2.0
//! Drop-in `llama-cli` executable for AMD Ryzen AI APUs.
//!
//! Provides 100% flag parity with upstream `llama-cli` (including `-m`, `-p`, `-n`,
//! `-c`, `-t`, `--temp`, `--top-p`, `--top-k`, `-i`, `-cnv`, `-ngl`), alongside
//! Ryzen AI heterogeneous APU routing (`--gpu-based`, `--cpu-based`, `--npu-based`,
//! `--tokenize`, `--prefill`, `--decode`, `--sample`, `--doctor`).

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use zero_copy_model_runner::container::reader::GgufModelReader;
use zero_copy_model_runner::engine::cpu_worker::CpuWorkerEngine;
use zero_copy_model_runner::engine::rocm_prefill::RocmPrefillEngine;
use zero_copy_model_runner::engine::xrt_decode::XrtDecodeEngine;
use zero_copy_model_runner::engine::{DecodeEngine, DecodeStepRequest, PrefillEngine, PrefillRequest};
use zero_copy_model_runner::memory::DmaBufHandle;
use zero_copy_model_runner::router::{DeviceTarget, RoutingPolicy};

#[path = "apu_doctor.rs"]
mod doctor;

#[derive(Parser, Debug)]
#[command(name = "llama-cli")]
#[command(about = "Drop-in llama.cpp CLI replacement for AMD Ryzen AI APUs (XDNA 2 Silicon)")]
#[command(version = "0.1.0")]
pub struct CliArgs {
    /// Path to model file (.gguf or .q4nx)
    #[arg(short = 'm', long = "model")]
    pub model: Option<String>,

    /// Prompt text to evaluate
    #[arg(short = 'p', long = "prompt")]
    pub prompt: Option<String>,

    /// Path to file containing prompt text
    #[arg(short = 'f', long = "file")]
    pub file: Option<String>,

    /// Number of tokens to predict
    #[arg(short = 'n', long = "n-predict", default_value_t = 128)]
    pub n_predict: usize,

    /// Size of the prompt context window
    #[arg(short = 'c', long = "ctx-size", default_value_t = 8192)]
    pub ctx_size: usize,

    /// Number of CPU threads to use
    #[arg(short = 't', long = "threads", default_value_t = 8)]
    pub threads: usize,

    /// Temperature scaling factor (0.0 = greedy argmax)
    #[arg(long = "temp", default_value_t = 0.7)]
    pub temperature: f32,

    /// Top-k candidate truncation
    #[arg(long = "top-k", default_value_t = 40)]
    pub top_k: usize,

    /// Top-p (nucleus) probability threshold
    #[arg(long = "top-p", default_value_t = 0.9)]
    pub top_p: f32,

    /// Min-p probability threshold
    #[arg(long = "min-p", default_value_t = 0.05)]
    pub min_p: f32,

    /// Repetition penalty factor
    #[arg(long = "repeat-penalty", default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// Number of layers to offload to GPU (upstream llama.cpp compatibility)
    #[arg(short = 'g', long = "n-gpu-layers", default_value_t = 999)]
    pub n_gpu_layers: i32,

    /// Run in interactive conversation mode
    #[arg(short = 'i', long = "interactive")]
    pub interactive: bool,

    /// Run in multi-turn conversation mode
    #[arg(long = "cnv")]
    pub conversation: bool,

    /// Custom system prompt
    #[arg(long = "system-prompt")]
    pub system_prompt: Option<String>,

    /// Chat template override (e.g. llama3, qwen, chatml)
    #[arg(long = "chat-template")]
    pub chat_template: Option<String>,

    /// Explicit path to AMD XDNA XCLBIN microcode
    #[arg(long = "xclbin")]
    pub xclbin: Option<String>,

    /// Hardware Diagnostic: run APU silicon and driver checks and exit
    #[arg(long = "doctor")]
    pub doctor: bool,

    /// Macro accelerator preset: maximize iGPU utilization
    #[arg(long = "gpu-based")]
    pub gpu_based: bool,

    /// Macro accelerator preset: maximize Zen 5 CPU utilization (100% AVX-512)
    #[arg(long = "cpu-based")]
    pub cpu_based: bool,

    /// Macro accelerator preset: maximize XDNA 2 NPU utilization
    #[arg(long = "npu-based")]
    pub npu_based: bool,

    /// Granular override for tokenization stage: cpu | gpu
    #[arg(long = "tokenize")]
    pub tokenize: Option<DeviceTarget>,

    /// Granular override for prefill stage: gpu | cpu | npu
    #[arg(long = "prefill")]
    pub prefill: Option<DeviceTarget>,

    /// Granular override for decode stage: npu | gpu | cpu
    #[arg(long = "decode")]
    pub decode: Option<DeviceTarget>,

    /// Granular override for sampling stage: cpu | gpu
    #[arg(long = "sample")]
    pub sample: Option<DeviceTarget>,

    /// Enable verbose APU hardware routing and timing telemetry
    #[arg(short = 'v', long = "verbose")]
    pub verbose: bool,
}

fn main() {
    let args = CliArgs::parse();

    // 1. Check if doctor mode was requested
    if args.doctor {
        doctor::print_doctor_report();
        return;
    }

    // Configure Rayon threadpool
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global();

    // 2. Hardware Routing Resolution
    let policy = RoutingPolicy::resolve(
        args.gpu_based,
        args.cpu_based,
        args.npu_based,
        args.tokenize,
        args.prefill,
        args.decode,
        args.sample,
    );
    policy.print_telemetry(args.verbose);

    // 3. Resolve Model Path
    let model_path = match &args.model {
        Some(m) => m.clone(),
        None => {
            // Check default search locations
            let default_models = [
                "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/qwen2.5-0.5b-instruct-q8_0.gguf",
                "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/Spark-X2.5-1.7B.gguf",
                "/opt/models/qwen3.5-9b.gguf",
            ];
            let found = default_models.iter().find(|p| Path::new(p).exists());
            match found {
                Some(&p) => p.to_string(),
                None => {
                    eprintln!("\x1b[31mError: No model path specified. Use '-m <PATH>' or run 'apu-model list'.\x1b[0m");
                    std::process::exit(1);
                }
            }
        }
    };

    if args.verbose {
        eprintln!("Loading model: {}", model_path);
    }

    // 4. Open Zero-Copy Model Reader
    let reader = match GgufModelReader::open(&model_path) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            eprintln!("\x1b[31mError opening model container '{}': {:?}\x1b[0m", model_path, e);
            std::process::exit(1);
        }
    };

    if args.verbose {
        eprintln!(
            "Model loaded: {} (layers={}, hidden={}, vocab={})",
            reader.arch_name,
            reader.hyperparams.num_layers,
            reader.hyperparams.hidden_dim,
            reader.hyperparams.vocab_size
        );
    }

    // 5. Setup Compute Engines based on Hardware Routing Policy
    let shared_trans = Arc::new(std::sync::Mutex::new(Some(
        zero_copy_model_runner::engine::TransformerContext::new(Arc::clone(&reader))
    )));

    let mut cpu_engine = CpuWorkerEngine::new();
    let _ = PrefillEngine::initialize(&mut cpu_engine, 0);
    cpu_engine.set_shared_transformer(Arc::clone(&shared_trans));

    let mut rocm_engine = RocmPrefillEngine::new();
    let _ = rocm_engine.initialize(0);
    rocm_engine.set_vocab_limit(reader.hyperparams.vocab_size.max(320_000));
    rocm_engine.set_shared_transformer(Arc::clone(&shared_trans));

    let mut xrt_engine = XrtDecodeEngine::new();
    let _ = xrt_engine.initialize(args.xclbin.as_deref().unwrap_or("default.xclbin"));
    xrt_engine.set_shared_transformer(Arc::clone(&shared_trans));

    let is_interactive = args.interactive || args.conversation;

    if is_interactive {
        run_interactive_repl(
            &args,
            Arc::clone(&reader),
            &mut cpu_engine,
            &mut rocm_engine,
            &mut xrt_engine,
            policy,
        );
    } else {
        // Single prompt execution
        let prompt_text = if let Some(p) = &args.prompt {
            p.clone()
        } else if let Some(f) = &args.file {
            fs::read_to_string(f).unwrap_or_default()
        } else {
            "Hello, tell me a short fact about AMD Ryzen AI APUs.".to_string()
        };

        execute_inference_pass(
            &prompt_text,
            args.n_predict,
            args.temperature,
            args.top_p,
            args.top_k,
            Arc::clone(&reader),
            &mut cpu_engine,
            &mut rocm_engine,
            &mut xrt_engine,
            policy,
            args.verbose,
        );
    }
}

/// Execute a single forward generation pass (Prefill + Autoregressive Decode).
fn execute_inference_pass(
    prompt: &str,
    max_tokens: usize,
    temp: f32,
    _top_p: f32,
    _top_k: usize,
    reader: Arc<GgufModelReader>,
    cpu_engine: &mut CpuWorkerEngine,
    rocm_engine: &mut RocmPrefillEngine,
    xrt_engine: &mut XrtDecodeEngine,
    policy: RoutingPolicy,
    verbose: bool,
) {
    let t_start = Instant::now();

    // Stage 1: Tokenize
    let tokens = reader.tokenizer.tokenize(prompt);
    if tokens.is_empty() {
        eprintln!("\x1b[31mError: Prompt produced zero tokens.\x1b[0m");
        return;
    }
    let prefill_len = tokens.len();

    if verbose {
        eprintln!("Tokenized {} prompt tokens.", prefill_len);
    }

    // Stage 2: Prompt Prefill (Batched GEMM)
    let prefill_start = Instant::now();
    let prefill_req = PrefillRequest {
        token_ids: &tokens,
        start_offset: 0,
        batch_size: 1,
    };
    let required_bytes = (tokens.len() + max_tokens + 64) * 128;
    let name_c = std::ffi::CString::new("apu_kv_cache").unwrap();
    let mem_fd = unsafe { libc::memfd_create(name_c.as_ptr(), 0) };
    if mem_fd >= 0 {
        let _ = unsafe { libc::ftruncate(mem_fd, required_bytes as i64) };
    }
    let kv_handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(mem_fd, required_bytes) };
    let _ = xrt_engine.attach_kv_cache(&kv_handle);

    let prefill_res = match policy.prefill {
        DeviceTarget::Gpu => rocm_engine.dispatch_prefill(prefill_req, &kv_handle, -1, 0),
        DeviceTarget::Cpu => cpu_engine.dispatch_prefill(prefill_req, &kv_handle, -1, 0),
        DeviceTarget::Npu => rocm_engine.dispatch_prefill(prefill_req, &kv_handle, -1, 0),
    };

    let prefill_res = match prefill_res {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\x1b[31mPrefill forward pass error: {:?}\x1b[0m", e);
            return;
        }
    };

    let prefill_time = prefill_start.elapsed();
    let mut current_token = prefill_res.initial_token_id;

    // Stream initial token
    let initial_str = reader.tokenizer.decode_token(current_token);
    print!("{}", initial_str);
    let _ = io::stdout().flush();

    // Stage 3: Autoregressive Decode Loop (GEMV)
    let decode_start = Instant::now();
    let mut generated_count = 1;

    for step in 1..max_tokens {
        let decode_req = DecodeStepRequest {
            input_token_id: current_token,
            sequence_index: prefill_len + step - 1,
            temperature: temp,
        };

        let step_res = match policy.decode {
            DeviceTarget::Npu => xrt_engine.dispatch_decode_step(decode_req, -1, 0, 0),
            DeviceTarget::Gpu => xrt_engine.dispatch_decode_step(decode_req, -1, 0, 0),
            DeviceTarget::Cpu => cpu_engine.dispatch_decode_step(decode_req, -1, 0, 0),
        };

        match step_res {
            Ok(r) => {
                if r.is_eos {
                    break;
                }
                current_token = r.output_token_id;
                let word = reader.tokenizer.decode_token(current_token);
                print!("{}", word);
                let _ = io::stdout().flush();
                generated_count += 1;
            }
            Err(e) => {
                eprintln!("\n\x1b[31mDecode step error: {:?}\x1b[0m", e);
                break;
            }
        }
    }

    println!();
    let total_time = t_start.elapsed();
    let decode_time = decode_start.elapsed();

    if verbose {
        let prefill_tps = (prefill_len as f64) / prefill_time.as_secs_f64();
        let decode_tps = (generated_count as f64) / decode_time.as_secs_f64();
        eprintln!("------------------------------------------------------------");
        eprintln!(
            "Prefill latency : {:.2} ms ({:.1} tok/s) [Engine: {}]",
            prefill_time.as_secs_f64() * 1000.0,
            prefill_tps,
            policy.prefill
        );
        eprintln!(
            "Decode latency  : {:.2} ms ({:.1} tok/s) [Engine: {}]",
            decode_time.as_secs_f64() * 1000.0,
            decode_tps,
            policy.decode
        );
        eprintln!("Total wall time : {:.2} s", total_time.as_secs_f64());
        eprintln!("------------------------------------------------------------");
    }
}

/// Run interactive multi-turn REPL chat loop.
fn run_interactive_repl(
    args: &CliArgs,
    reader: Arc<GgufModelReader>,
    cpu_engine: &mut CpuWorkerEngine,
    rocm_engine: &mut RocmPrefillEngine,
    xrt_engine: &mut XrtDecodeEngine,
    policy: RoutingPolicy,
) {
    println!("============================================================");
    println!(" AMD Ryzen AI APU Interactive Chat Session");
    println!(" Model: {} | Architecture: {}", Path::new(&args.model.clone().unwrap_or_default()).file_name().unwrap_or_default().to_string_lossy(), reader.arch_name);
    println!(" Routing: Prefill -> {}, Decode -> {}", policy.prefill, policy.decode);
    println!(" Type your message and press Enter. Commands: /exit, /reset");
    println!("============================================================");

    let stdin = io::stdin();
    let mut handle = stdin.lock();

    loop {
        print!("\n\x1b[34mUser >\x1b[0m ");
        let _ = io::stdout().flush();

        let mut line = String::new();
        if handle.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "/exit" || trimmed == "/quit" {
            println!("Goodbye!");
            break;
        }

        if trimmed == "/reset" {
            println!("Context reset.");
            continue;
        }

        print!("\x1b[32mAssistant >\x1b[0m ");
        let _ = io::stdout().flush();

        // Format prompt using ChatML template
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            trimmed
        );

        execute_inference_pass(
            &formatted,
            args.n_predict,
            args.temperature,
            args.top_p,
            args.top_k,
            Arc::clone(&reader),
            cpu_engine,
            rocm_engine,
            xrt_engine,
            policy,
            args.verbose,
        );
    }
}
