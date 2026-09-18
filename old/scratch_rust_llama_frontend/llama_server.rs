// SPDX-License-Identifier: Apache-2.0
//! Drop-in `llama-server` HTTP Daemon with OpenAI API Parity for AMD Ryzen AI APUs.
//!
//! Provides OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
//! and `/health` endpoints with Server-Sent Events (SSE) token streaming.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use zero_copy_model_runner::container::reader::GgufModelReader;
use zero_copy_model_runner::engine::cpu_worker::CpuWorkerEngine;
use zero_copy_model_runner::engine::rocm_prefill::RocmPrefillEngine;
use zero_copy_model_runner::engine::xrt_decode::XrtDecodeEngine;
use zero_copy_model_runner::engine::{DecodeEngine, DecodeStepRequest, PrefillEngine, PrefillRequest};
use zero_copy_model_runner::memory::DmaBufHandle;
use zero_copy_model_runner::router::{DeviceTarget, RoutingPolicy};

#[derive(Parser, Debug)]
#[command(name = "llama-server")]
#[command(about = "Drop-in llama-server HTTP daemon for AMD Ryzen AI APUs (OpenAI API Compatible)")]
#[command(version = "0.1.0")]
struct ServerArgs {
    /// Path to model file (.gguf or .q4nx)
    #[arg(short = 'm', long = "model")]
    model: Option<String>,

    /// Host IP address to bind to
    #[arg(long = "host", default_value = "127.0.0.1")]
    host: String,

    /// Port number to listen on
    #[arg(long = "port", default_value_t = 8080)]
    port: u16,

    /// Size of the prompt context window
    #[arg(short = 'c', long = "ctx-size", default_value_t = 8192)]
    ctx_size: usize,

    /// Number of CPU threads
    #[arg(short = 't', long = "threads", default_value_t = 8)]
    threads: usize,

    /// Macro accelerator preset: maximize iGPU utilization
    #[arg(long = "gpu-based")]
    gpu_based: bool,

    /// Macro accelerator preset: maximize Zen 5 CPU utilization (100% AVX-512)
    #[arg(long = "cpu-based")]
    cpu_based: bool,

    /// Macro accelerator preset: maximize XDNA 2 NPU utilization
    #[arg(long = "npu-based")]
    npu_based: bool,

    /// Granular override for tokenization stage
    #[arg(long = "tokenize")]
    tokenize: Option<DeviceTarget>,

    /// Granular override for prefill stage
    #[arg(long = "prefill")]
    prefill: Option<DeviceTarget>,

    /// Granular override for decode stage
    #[arg(long = "decode")]
    decode: Option<DeviceTarget>,

    /// Granular override for sampling stage
    #[arg(long = "sample")]
    sample: Option<DeviceTarget>,

    /// Enable verbose APU routing telemetry
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

/// Shared application state across HTTP request workers.
struct AppState {
    reader: Arc<GgufModelReader>,
    model_name: String,
    cpu_engine: Mutex<CpuWorkerEngine>,
    rocm_engine: Mutex<RocmPrefillEngine>,
    xrt_engine: Mutex<XrtDecodeEngine>,
    policy: RoutingPolicy,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    #[allow(dead_code)]
    prompt: Option<String>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageOut,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ChatMessageOut {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    index: usize,
    delta: StreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[tokio::main]
async fn main() {
    let args = ServerArgs::parse();

    // Hardware routing
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

    // Resolve model path
    let model_path = match &args.model {
        Some(m) => m.clone(),
        None => {
            let default_models = [
                "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/qwen2.5-0.5b-instruct-q8_0.gguf",
                "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/Spark-X2.5-1.7B.gguf",
            ];
            let found = default_models.iter().find(|p| Path::new(p).exists());
            match found {
                Some(&p) => p.to_string(),
                None => {
                    eprintln!("Error: No model path specified. Use '-m <PATH>'");
                    std::process::exit(1);
                }
            }
        }
    };

    println!("Loading model: {}", model_path);
    let reader = Arc::new(GgufModelReader::open(&model_path).expect("Failed to open model"));
    let model_name = Path::new(&model_path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

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
    let _ = xrt_engine.initialize("default.xclbin");
    xrt_engine.set_shared_transformer(Arc::clone(&shared_trans));

    let state = Arc::new(AppState {
        reader,
        model_name: model_name.clone(),
        cpu_engine: Mutex::new(cpu_engine),
        rocm_engine: Mutex::new(rocm_engine),
        xrt_engine: Mutex::new(xrt_engine),
        policy,
    });

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/", get(webui_handler))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("Invalid address");
    println!("============================================================");
    println!(" AMD Ryzen AI APU Drop-In llama-server Online!");
    println!(" Listening on: http://{}", addr);
    println!(" OpenAI Endpoint: http://{}/v1/chat/completions", addr);
    println!(" Web UI:          http://{}", addr);
    println!("============================================================");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "silicon": "AMD Ryzen AI (XDNA 2 Silicon)",
        "runtime": "zero-copy-model-runner"
    }))
}

async fn models_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": 1700000000,
            "owned_by": "amd-apu",
            "architecture": state.reader.arch_name
        }]
    }))
}

async fn webui_handler(State(state): State<Arc<AppState>>) -> Html<String> {
    Html(format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AMD Ryzen AI APU Runner</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #ef4444; }}
        .badge {{ background: #1e293b; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.85rem; color: #38bdf8; border: 1px solid #334155; }}
        #chatbox {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; height: 450px; overflow-y: auto; padding: 1rem; margin-top: 1rem; }}
        .msg {{ margin-bottom: 0.8rem; line-height: 1.5; }}
        .user {{ color: #60a5fa; font-weight: bold; }}
        .assistant {{ color: #4ade80; font-weight: bold; }}
        .input-row {{ display: flex; gap: 0.5rem; margin-top: 1rem; }}
        input[type="text"] {{ flex: 1; background: #1e293b; border: 1px solid #334155; color: white; padding: 0.75rem; border-radius: 6px; }}
        button {{ background: #ef4444; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; font-weight: bold; }}
        button:hover {{ background: #dc2626; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AMD Ryzen AI APU Inference Server</h1>
        <div>
            <span class="badge">Model: {}</span>
            <span class="badge">Prefill: {}</span>
            <span class="badge">Decode: {}</span>
            <span class="badge">Silicon: XDNA 2</span>
        </div>
        <div id="chatbox"></div>
        <div class="input-row">
            <input type="text" id="userInput" placeholder="Ask anything to the Ryzen AI APU..." onkeydown="if(event.key==='Enter') sendPrompt()"/>
            <button onclick="sendPrompt()">Send</button>
        </div>
    </div>
    <script>
        async function sendPrompt() {{
            const input = document.getElementById('userInput');
            const box = document.getElementById('chatbox');
            const prompt = input.value.trim();
            if (!prompt) return;
            box.innerHTML += `<div class="msg"><span class="user">User:</span> ${{prompt}}</div>`;
            input.value = '';
            box.scrollTop = box.scrollHeight;

            const res = await fetch('/v1/chat/completions', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ messages: [{{ role: 'user', content: prompt }}], max_tokens: 256 }})
            }});
            const data = await res.json();
            const reply = data.choices[0].message.content;
            box.innerHTML += `<div class="msg"><span class="assistant">APU:</span> ${{reply}}</div>`;
            box.scrollTop = box.scrollHeight;
        }}
    </script>
</body>
</html>"#,
        state.model_name, state.policy.prefill, state.policy.decode
    ))
}

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Reconstruct prompt from messages
    let mut prompt = String::new();
    for msg in &payload.messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");

    let tokens = state.reader.tokenizer.tokenize(&prompt);
    let prefill_len = tokens.len();
    let prefill_req = PrefillRequest {
        token_ids: &tokens,
        start_offset: 0,
        batch_size: 1,
    };
    let required_bytes = (tokens.len() + payload.max_tokens + 64) * 128;
    let name_c = std::ffi::CString::new("apu_kv_server").unwrap();
    let mem_fd = unsafe { libc::memfd_create(name_c.as_ptr(), 0) };
    if mem_fd >= 0 {
        let _ = unsafe { libc::ftruncate(mem_fd, required_bytes as i64) };
    }
    let kv_handle = unsafe { DmaBufHandle::from_raw_fd_unchecked(mem_fd, required_bytes) };
    {
        let mut xrt = state.xrt_engine.lock().await;
        let _ = xrt.attach_kv_cache(&kv_handle);
    }

    // Execute prefill
    let initial_token = {
        match state.policy.prefill {
            DeviceTarget::Gpu => {
                let engine = state.rocm_engine.lock().await;
                engine.dispatch_prefill(prefill_req, &kv_handle, -1, 0).map(|r| r.initial_token_id)
            }
            DeviceTarget::Cpu => {
                let engine = state.cpu_engine.lock().await;
                engine.dispatch_prefill(prefill_req, &kv_handle, -1, 0).map(|r| r.initial_token_id)
            }
            DeviceTarget::Npu => {
                let engine = state.rocm_engine.lock().await;
                engine.dispatch_prefill(prefill_req, &kv_handle, -1, 0).map(|r| r.initial_token_id)
            }
        }
    };

    let mut current_token = match initial_token {
        Ok(t) => t,
        Err(e) => {
            return Json(serde_json::json!({ "error": format!("{:?}", e) })).into_response();
        }
    };

    if payload.stream {
        // SSE Streaming
        let initial_str = state.reader.tokenizer.decode_token(current_token);
        let mut generated_words = vec![initial_str];

        for step in 1..payload.max_tokens {
            let req = DecodeStepRequest {
                input_token_id: current_token,
                sequence_index: prefill_len + step - 1,
                temperature: payload.temperature,
            };

            let res = match state.policy.decode {
                DeviceTarget::Npu | DeviceTarget::Gpu => {
                    let engine = state.xrt_engine.lock().await;
                    engine.dispatch_decode_step(req, -1, 0, 0)
                }
                DeviceTarget::Cpu => {
                    let engine = state.cpu_engine.lock().await;
                    engine.dispatch_decode_step(req, -1, 0, 0)
                }
            };

            if let Ok(step_res) = res {
                if step_res.is_eos {
                    break;
                }
                current_token = step_res.output_token_id;
                generated_words.push(state.reader.tokenizer.decode_token(current_token));
            } else {
                break;
            }
        }

        let model_str = state.model_name.clone();
        let sse_stream = tokio_stream::iter(generated_words.into_iter().enumerate().map(move |(idx, word)| {
            let chunk = StreamChunk {
                id: format!("chatcmpl-{}", now),
                object: "chat.completion.chunk".into(),
                created: now,
                model: model_str.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        role: if idx == 0 { Some("assistant".into()) } else { None },
                        content: Some(word),
                    },
                    finish_reason: None,
                }],
            };
            Ok::<_, std::convert::Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()))
        }));

        Sse::new(sse_stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        // Non-streaming response
        let mut full_text = state.reader.tokenizer.decode_token(current_token);

        for step in 1..payload.max_tokens {
            let req = DecodeStepRequest {
                input_token_id: current_token,
                sequence_index: prefill_len + step - 1,
                temperature: payload.temperature,
            };

            let res = match state.policy.decode {
                DeviceTarget::Npu | DeviceTarget::Gpu => {
                    let engine = state.xrt_engine.lock().await;
                    engine.dispatch_decode_step(req, -1, 0, 0)
                }
                DeviceTarget::Cpu => {
                    let engine = state.cpu_engine.lock().await;
                    engine.dispatch_decode_step(req, -1, 0, 0)
                }
            };

            if let Ok(step_res) = res {
                if step_res.is_eos {
                    break;
                }
                current_token = step_res.output_token_id;
                full_text.push_str(&state.reader.tokenizer.decode_token(current_token));
            } else {
                break;
            }
        }

        Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", now),
            object: "chat.completion".into(),
            created: now,
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".into(),
                    content: full_text,
                },
                finish_reason: "stop".into(),
            }],
        }).into_response()
    }
}

async fn completions_handler(
    state: State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    chat_completions_handler(state, Json(payload)).await
}
