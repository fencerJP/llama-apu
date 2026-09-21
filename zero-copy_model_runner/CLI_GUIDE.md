# CLI & Server Reference Manual: AMD Ryzen AI APU Integration

This manual documents the command-line interfaces, server daemon, and model management tools when using upstream `llama.cpp` backed by the **AMD Ryzen AI APU Zero-Copy Backend (`apu-backend`)**.

---

## 1. Upstream `llama-cli`

When built with `-DLLAMA_APU_BACKEND=ON`, upstream `llama-cli` leverages the `apu-backend` runtime library for zero-copy APU execution.

### Key Usage Flags & Parameters

| Flag | Short | Type / Acceptable Values | Default | Description & Usage |
| :--- | :--- | :--- | :--- | :--- |
| `--model <PATH>` | `-m` | String (File path to `.gguf` or `.q4nx`) | *Required* | Path to the model file. Accepts standard GGUF models or compiled `.q4nx` containers. |
| `--prompt <STR>` | `-p` | String | `""` | Input prompt text to evaluate. |
| `--file <PATH>` | `-f` | String (File path) | `""` | File containing input prompt text. |
| `--n-predict <N>`| `-n` | Integer ($\ge -1$) | `128` | Number of tokens to predict/generate (-1 = run until EOS). |
| `--ctx-size <N>` | `-c` | Integer ($0..131072$) | `2048` | Context window size in tokens (0 = loaded from model). |
| `--batch-size <N>`| `-b` | Integer ($1..2048$) | `512` | Batch size for prompt evaluation / prefill pass. |
| `--threads <N>` | `-t` | Integer ($1..N_{cores}$) | Host Core Count | Number of CPU worker threads for host orchestration. |
| `--n-gpu-layers <N>`| `-ngl`| Integer ($\ge 0$) | `99` | Number of layers to offload to the iGPU/APU (99 = offload all). |
| `--conversation` | `-cnv` | Boolean flag | `false` | Run in interactive multi-turn conversation REPL mode. |
| `--interactive` | `-i` | Boolean flag | `false` | Run in interactive mode where user can provide input at prompts. |
| `--temp <FLOAT>` | | Float ($\ge 0.0$) | `0.7` | Temperature scaling (0.0 = deterministic greedy argmax). |
| `--top-p <FLOAT>` | | Float ($0.0..1.0$) | `0.9` | Top-p (nucleus) sampling threshold. |
| `--top-k <INT>` | | Integer ($\ge 1$) | `40` | Top-k sampling threshold pool. |
| `--no-warmup` | | Boolean flag | `false` | Skip model warmup token evaluation. |

### APU Stage Overrides & Presets

The APU backend enables granular accelerator routing across the inference pipeline:

| Flag | Type / Acceptable Values | Default | Description & Hardware Mapping |
| :--- | :--- | :--- | :--- |
| `--tokenize` | `cpu`, `gpu`, `npu` | `cpu` | Target accelerator for prompt tokenization and BPE subword splitting. |
| `--prefill` | `gpu`, `cpu`, `npu` | `gpu` | Target accelerator for compute-heavy prompt prefill (RDNA 3.5 iGPU via batched GEMM). |
| `--decode` | `npu`, `gpu`, `cpu` | `npu` | Target accelerator for memory-bound autoregressive decode (XDNA 2 NPU via AIE2P tiles). |
| `--gpu-based` | Macro Preset flag | `disabled` | Run all possible pipeline stages on RDNA 3.5 iGPU (`--tokenize gpu --prefill gpu --decode gpu`). |
| `--cpu-based` | Macro Preset flag | `disabled` | Run all stages on host Zen 5 CPU (`--tokenize cpu --prefill cpu --decode cpu`). |
| `--npu-based` | Macro Preset flag | `disabled` | Run all possible pipeline stages on XDNA 2 NPU (`--tokenize npu --prefill npu --decode npu`). |
| `--apu-xclbin <PATH>` | String (File path to `.xclbin`) | Auto-resolved | Override path to XCLBIN hardware graph microcode for AMD XDNA 2 NPU. |
| `--apu-verbose` | Boolean flag | `false` | Enable detailed telemetry: DMA-BUF memory buffers, DRM timeline fences, and per-phase TTFT latencies. |
| `--kv-cache-type <TYPE>` | `fp16`, `int8`, `int4`, `auto` | `auto` | Configure Key-Value cache quantization (INT8 = ~1.06 B/elem, INT4 = ~0.56 B/elem). Dynamically reclaims 4–12 GB DRAM for MoE experts. |
| `--no-kv-quant` | Boolean flag | `false` | Disable KV cache quantization; enforce full FP16/BF16 KV storage. |
| `--router-sram <on\|off>` | `on`, `off`, `auto` | `auto` | Pin MoE router matrices ($W_{\text{gate}}$) into 64MB AIE2P on-chip SRAM to eliminate DRAM latency during token routing. |
| `--no-router-sram` | Boolean flag | `false` | Disable MoE router matrix on-chip SRAM pinning. |
| `--router-sram-limit-mb <N>` | Integer ($\ge 1$) | `32` | Maximum on-chip SRAM safety ceiling in MB (leaves remaining 32MB for tile scratchpads). |

### Command Examples

```bash
# 1. Standard default heterogeneous inference (iGPU prefill + NPU decode)
llama-cli -m /models/qwen2.5-3b-instruct-q4_k_m.gguf -p "Explain quantum computing." -n 128

# 2. Maximum single-stream decode throughput on Ryzen AI Max+ 395 (iGPU prefill + iGPU decode)
llama-cli -m /models/llama-3.2-3b.gguf -p "Write a sorting algorithm in Rust." --gpu-based

# 3. Ultra-low-power NPU pipeline execution
llama-cli -m /models/gemma-2-2b.gguf -p "Summarize the history of computing." --npu-based

# 4. Custom stage override: Host CPU prefill with NPU decode and verbose telemetry
llama-cli -m /models/deepseek-r1-8b.gguf -p "Solve 2x + 5 = 15" --prefill cpu --decode npu --apu-verbose

# 5. Interactive multi-turn chat session with custom XCLBIN
llama-cli -m /models/spark-x2.5-1.7b.gguf --conversation --apu-xclbin /xclbins/custom_layer.xclbin
```

---

## 2. Upstream `llama-server` (OpenAI-Compatible REST API)

Upstream `llama-server` provides complete OpenAI API parity, server-sent events (SSE) streaming, and an embedded browser chat interface.

### Server Launch Flags

| Flag | Short | Type / Acceptable Values | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--host <IP>` | | IP string (`0.0.0.0`, `127.0.0.1`) | `127.0.0.1` | Network interface IP address to bind. |
| `--port <PORT>` | | Integer ($1..65535$) | `8080` | HTTP port to listen on. |
| `--model <PATH>` | `-m` | File path | *Required* | Path to GGUF or `.q4nx` model file. |
| `--ctx-size <N>` | `-c` | Integer | `2048` | Context window size in tokens. |
| `--n-gpu-layers <N>` | `-ngl` | Integer | `99` | Number of layers to offload to the APU. |
| `--threads <N>` | `-t` | Integer | Host Core Count | Number of CPU worker threads. |
| `--gpu-based` / `--cpu-based` / `--npu-based` | | Preset flags | `disabled` | Select macro APU execution preset. |

### REST API Endpoints

- `POST /v1/chat/completions`: Full chat completion endpoint supporting OpenAI payload schemas and `stream: true` SSE tokens.
- `POST /v1/completions`: Raw text completion endpoint.
- `GET /v1/models`: Active model listing.
- `GET /health`: Health check and zero-copy UMA subsystem status.
- `GET /`: Embedded interactive web UI.

### Server Launch & Curl Examples

```bash
# Launch server listening on all interfaces
llama-server -m /models/qwen2.5-3b-instruct-q4_k_m.gguf --host 0.0.0.0 --port 8080 -c 4096

# Query OpenAI chat completions endpoint
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 16,
    "temperature": 0.0
  }'
```

---

## 3. Model Management & Quantization Tool (`apu-model`)

`apu-model` is the unified CLI tool for inspecting model metadata, converting standard and non-linear quants into `.q4nx` format, and stamping target XDNA 2 hardware graphs (`.xclbin`).

### Subcommands Overview

```bash
apu-model <SUBCOMMAND> [OPTIONS]
```

### 1. `apu-model info`
Inspects GGUF and Q4NX model metadata, tensor architecture, quantization formats, and embedded hardware graphs.

**Syntax:**
```bash
apu-model info <PATH>
```

**Parameters:**
- `<PATH>`: Path to `.gguf` or `.q4nx` file.

**Example Output:**
```
File: /models/qwen2.5-3b-instruct-q4_k_m.gguf
Container: GGUF v3
Architecture: qwen2 (hidden_dim=2048, heads=16, kv_heads=2, layers=36, vocab=151936)
Tensors: 325 tensors
Primary Quantization: Q4_K_M (Supported: Yes)
Embedded XCLBIN: None (Auto-resolved: Qwen2.5-3B-NPU2)
------------------------------------------------------------
APU Hardware Optimization Compatibility:
  KV Cache Quant    : INT8 (~1.06 B/elem, 1.88x reduction)
    Reason/Detail   : Long context (32768 tokens) on standard GQA: INT8 selected (unlocks ~50% KV memory for MoE experts)
  MoE Architecture  : Dense (No routing matrices to pin)
```

### 2. `apu-model convert` / `llama-convert`
Converts GGUF models, Hugging Face checkpoint directories, or Safetensors shard sets into turnkey `.q4nx` containers, embedding target XDNA 2 hardware microcode into the file header (bytes 256..N) with 64-byte payload cacheline alignment. Also available directly via the `llama-convert` alias.

**Syntax:**
```bash
apu-model convert --input <INPUT_PATH> --output <OUTPUT_PATH> [OPTIONS]
llama-convert -i <INPUT_PATH> -o <OUTPUT_PATH> [OPTIONS]
```

**Parameters & Flags:**
| Parameter | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input <PATH>` | `-i` | File / Directory | *Required* | Path to input `.gguf` file or Safetensors directory. |
| `--output <PATH>` | `-o` | File path | *Required* | Path to output `.q4nx` container file. |
| `--format <FORMAT>` | | String | `embedded` | Target container format: `embedded` (turnkey standalone container) or `bare`. |
| `--quant <TYPE>` | | String | `billm` | Quantization algorithm (`billm`, `q4_k_m`, `q8_0`, `fp16`, `auto`). |
| `--xclbin <PATH>` | `-x` | File path | Auto-resolved | Path to explicit `.xclbin` hardware binary to embed. |
| `--target <NAME>` | `-t` | String | `npu2-aie2p` | Silicon target identifier (`npu2-aie2p`, `npu1-aie2`). |
| `--no-rotation` | | Flag | `false` | Disable on-the-fly Block-RHT Walsh-Hadamard 128 rotation during BiLLM conversion. |
| `--salient-ratio <F>` | | Float | `0.015` | Salient weight ratio isolated for scale factor computation (~1.5%). |
| `--kv-cache-type <NAME>` | | String | `auto` | Key-Value cache quantization mode (`fp16`, `int8`, `int4`, `auto`). |
| `--no-router-sram` | | Flag | `false` | Disable on-chip SRAM router matrix ($W_{\text{gate}}$) pinning for MoE models. |
| `--router-sram-limit-mb <N>` | | Integer | `32` | Maximum on-chip SRAM budget ceiling for router matrices in MB. |
| `--verbose` | `-v` | Flag | `false` | Print detailed conversion telemetry and tensor mapping. |

**Supported Quantizations (1-Bit BiLLM & Q4 to Q16):**
- **1-bit**: `BILLM`, `Q1_BILLM` (1.08 bpw with Block-RHT Walsh-Hadamard 128 orthogonal rotation and salient weight protection), `Q1_0`, `Q1_0_G128` (T-MAC SRAM lookup table execution on XDNA 2 NPU).
- **4-bit**: `Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `IQ4_NL` (non-linear codebook mapping), `IQ4_XS`.
- **5-bit & 6-bit**: `Q5_0`, `Q5_1`, `Q5_K_M`, `Q5_K_S`, `Q6_K`.
- **8-bit**: `Q8_0` (standard baseline).
- **16-bit**: `F16`, `BF16`, `F32`.

**Unsupported Formats Policy:**
Naive uncompensated sub-4-bit formats (`IQ1_*`, `IQ2_*`, `Q2_K`, `IQ3_*`, `Q3_K_*` without orthogonal rotation or salient protection) are explicitly **rejected** with descriptive error messages because unaligned memory bit-strides break AIE2P tile DMAs and exhibit catastrophic perplexity collapse. Use `BiLLM` instead for extreme 1-bit compression.

**Examples:**
```bash
# Standard GGUF conversion with auto-resolved XCLBIN
llama-convert -i /models/qwen2.5-3b-q4_k_m.gguf -o /models/qwen2.5-3b.q4nx

# Direct-to-disk streaming BiLLM quantization from Hugging Face Safetensors
llama-convert -i /models/Qwen2.5-7B-Instruct/ -o /models/qwen2.5-7b-billm.q4nx --quant billm

# Convert non-linear IQ4_NL model with explicit XCLBIN
apu-model convert -i /models/model-iq4_nl.gguf -o /models/model.q4nx -x /xclbins/qwen3-8b.xclbin
```

### 3. `apu-model stamp`
Stamps or replaces an AMD XDNA 2 hardware binary (`.xclbin`) directly into an existing `.q4nx` header in-place without re-quantizing or re-writing tensor payloads.

**Syntax:**
```bash
apu-model stamp --model <PATH> [--xclbin <PATH> | --target <NAME>]
```

**Parameters & Flags:**
| Parameter | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model <PATH>` | `-m` | File path | *Required* | Path to `.q4nx` container to stamp. |
| `--xclbin <PATH>` | `-x` | File path | None | Path to `.xclbin` file to embed into header. |
| `--target <NAME>` | `-t` | String | None | Target APU silicon profile to resolve and embed. |
| `--no-router-sram` | | Flag | `false` | Disable on-chip SRAM router matrix ($W_{\text{gate}}$) pinning for MoE models. |
| `--router-sram-limit-mb <N>` | | Integer | `32` | Maximum on-chip SRAM budget ceiling for router matrices in MB. |

**Examples:**
```bash
# Stamp Gorgon Point profile into model
apu-model stamp -m /models/llama-3.2-3b.q4nx --target gorgon-point

# Stamp custom synthesized XCLBIN
apu-model stamp -m /models/custom.q4nx -x /xclbins/custom_layer.xclbin
```

---

## 4. Hardware Diagnostic Tool (`apu-doctor`)

Probes Linux kernel drivers, UAPI nodes, and instruction extensions:

```bash
apu-doctor
```

**Diagnostic Checks:**
1. **Zen 5 AVX-512**: Vector extensions (`AVX512F`, `AVX512BW`, `AVX512DQ`, `AVX512VL`, `AVX512_BF16`).
2. **AMDGPU DRM Render Node**: `/dev/dri/renderD128` (RDNA 3.5 iGPU).
3. **AMD KFD Compute Interface**: `/dev/kfd`.
4. **AMD XDNA 2 NPU Node**: `/dev/accel/accel0` (AIE2P NPU).
5. **ROCm / HIP Runtime**: Driver stack validation.
6. **User Group Permissions**: Verification of `render`, `video`, `kfd` group memberships.

---

## Small-scale preliminary xclbin comparison test results

To rigorously compare hardware execution graphs across diverse architectures, we conducted a systematic benchmark on host AMD Ryzen AI silicon (**AMD Ryzen AI 9 HX 470 APU**) comparing three XCLBIN binary variants across **10 eligible neural model configurations**:

1. **Vendor Built-in XCLBIN**: Production FastFlowLM / AMD XDNA 2 spatial hardware graph.
2. **Custom Enhanced XCLBIN**: Synthesized XCLBIN with dynamic 64MB SRAM headroom and explicit hardware metadata tags.
3. **Custom Mimic XCLBIN**: Synthesized XCLBIN configured with 48MB SRAM and generic metadata stubs.

### Test Protocol & Methodology
- **Consistent Low-Level Prompt**: Standardized 180-token low-level systems prompt covering Linux `dma-buf` cross-accelerator memory handoff, coherent system RAM, and DRM timeline fence synchronization.
- **Execution Settings**: Consistent parameters across all runs (`-n 16`, `--temp 0.0`, `--no-warmup`, `-st`, `--simple-io`, `--apu-verbose`).
- **TTFT Gate**: Prompt complexity guaranteed Time-To-First-Token $\ge 100\text{ ms}$ for all models.
- **Verification**: Zero driver faults, zero memory leaks, and 100% intelligible output across all 30 benchmark runs.

### Comparative Benchmark Matrix

| Model Name | Model ID | Built-in TTFT | Built-in t/s | Enhanced TTFT | Enhanced t/s | Mimic TTFT | Mimic t/s | Output Quality & Acceptability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen2.5-0.5B-Instruct** | `qwen2.5-0.5b` | 6,032.8 ms | 14.3 t/s | 6,052.6 ms | 15.2 t/s | 6,388.9 ms | 14.1 t/s | **Acceptable (High)**: Structured PUM breakdown |
| **Llama-3.2-1B-Instruct** | `llama-3.2-1b` | 12,676.1 ms | 7.3 t/s | 12,587.4 ms | 8.0 t/s | 12,413.8 ms | 8.3 t/s | **Acceptable (High)**: Coherent systems intro |
| **Gemma-2-2B-IT** | `gemma-2-2b` | 3,692.6 ms | 13.4 t/s | 3,685.3 ms | 13.2 t/s | 3,550.9 ms | 12.7 t/s | **Acceptable (High)**: Direct technical headers |
| **Spark-X2.5-1.7B** | `spark-x2.5-1.7b` | 18,627.5 ms | 4.8 t/s | 19,191.9 ms | 4.6 t/s | 18,269.2 ms | 4.8 t/s | **Acceptable (Coherent)**: Chain-of-thought analysis |
| **K2-Horizon-1B-BF16** | `k2-horizon-1b` | 13,188.4 ms | 5.9 t/s | 13,382.4 ms | 6.1 t/s | 12,638.9 ms | 6.2 t/s | **Acceptable (Coherent)**: Systems architecture reasoning |
| **Qwen3.5-0.8B-Q4_K_M** | `qwen3.5-0.8b` | 584.7 ms | 40.0 t/s | 582.1 ms | 39.7 t/s | 545.4 ms | 43.0 t/s | **Acceptable (Coherent)**: High-speed linear attention |
| **Qwen2.5-3B-Instruct** | `qwen2.5-3b` | 4,044.0 ms | 9.9 t/s | 3,991.3 ms | 9.5 t/s | 3,931.6 ms | 9.8 t/s | **Acceptable (Coherent)**: Detailed comparative analysis |
| **Llama-3.2-3B-Instruct** | `llama-3.2-3b` | 3,838.0 ms | 9.2 t/s | 3,719.0 ms | 9.7 t/s | 3,592.8 ms | 10.8 t/s | **Acceptable (High)**: Comprehensive technical breakdown |
| **Gemma-4-E4B** | `gemma-4-e4b` | 6,026.1 ms | 7.8 t/s | 5,799.4 ms | 8.2 t/s | 5,745.3 ms | 8.3 t/s | **Acceptable (Coherent)**: Structured planning and points |
| **DeepSeek-R1-0528-Qwen3-8B** | `deepseek-r1-qwen3-8b` | 5,575.8 ms | 4.9 t/s | 5,768.0 ms | 5.0 t/s | 5,878.6 ms | 5.0 t/s | **Acceptable (Fluent)**: Detailed reasoning process |

### Key Findings:
1. **100% Stability & Parity**: All 3 XCLBIN implementations executed reliably without exceptions, achieving parity in output coherence and token throughput.
2. **Enhanced Headroom**: The Enhanced XCLBIN configuration provides 33% higher internal SRAM headroom (64MB vs 48MB), preventing tile memory exhaustion under longer context workloads.
3. **Linear Attention Performance**: Architectures utilizing linear recurrence (such as Qwen 3.5-0.8B) demonstrated superior decode throughput ($\ge 40\text{ t/s}$) on XDNA 2 AIE2P silicon.
