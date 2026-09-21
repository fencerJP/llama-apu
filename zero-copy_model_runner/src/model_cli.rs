// SPDX-License-Identifier: Apache-2.0
//! Automated Model Manager and XCLBIN Stamper Module (`model_cli`).
//!
//! Provides inspection, XCLBIN microcode stamping, conversion, and validation
//! for GGUF and `.q4nx` models on AMD Ryzen AI APUs, including CSA2 dynamic KV cache
//! evaluation and MoE router matrix on-chip SRAM pinning planning.

use std::fs;
use std::path::{Path, PathBuf};
use clap::{Parser, Subcommand};

use crate::container::converter::load_or_convert_model;
use crate::container::reader::GgufModelReader;
use crate::container::resolver::resolve_xclbin_profile;
use crate::container::xclbin_builder::{ModelGraphTopology, TargetHardware, XclbinBuilder};
use crate::container::Q4nxModel;
use crate::memory::kv_quant::{evaluate_kv_quant_compatibility, KvCacheQuantType};

#[derive(Parser)]
#[command(name = "apu-model")]
#[command(about = "AMD Ryzen AI APU Model Management and XCLBIN Stamping Utility")]
#[command(version = "0.5.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Inspect model metadata, hyperparameters, embedded XCLBIN status, and APU hardware optimization plans.
    Info {
        /// Path to .gguf or .q4nx model file.
        model_path: String,
        /// Optional KV cache quantization type to evaluate (fp16, int8, int4, auto).
        #[arg(long, default_value = "auto")]
        kv_cache_type: String,
    },
    /// Stamp or embed an XCLBIN hardware graph into a model container in-place.
    Stamp {
        /// Path to .gguf or .q4nx model file.
        model_path: String,
        /// Optional explicit path to .xclbin file.
        #[arg(long)]
        xclbin: Option<String>,
        /// Output destination path (defaults to in-place or adjacent .q4nx).
        #[arg(short, long)]
        output: Option<String>,
        /// Disable on-chip SRAM router matrix pinning for MoE models.
        #[arg(long)]
        no_router_sram: bool,
        /// Maximum on-chip SRAM budget ceiling for router matrices in megabytes (default: 32).
        #[arg(long, default_value = "32")]
        router_sram_limit_mb: usize,
    },
    /// Convert a model (GGUF file or Safetensors directory) to tile-interleaved .q4nx format with embedded XCLBIN.
    Convert {
        /// Input GGUF file or Safetensors directory path.
        #[arg(short, long)]
        input: String,
        /// Destination .q4nx output file (defaults to adjacent .q4nx).
        #[arg(short, long)]
        output: Option<String>,
        /// Container format: 'embedded' (default, standalone container) or 'bare' (omits embedded GGUF/XCLBIN).
        #[arg(long, default_value = "embedded")]
        format: String,
        /// Quantization format (billm, q4_k_m, q8_0, fp16, auto). Default is billm.
        #[arg(long, default_value = "billm")]
        quant: String,
        /// Target hardware architecture (e.g. npu2-aie2p).
        #[arg(short, long, default_value = "npu2-aie2p")]
        target: String,
        /// Optional explicit path to .xclbin file.
        #[arg(long)]
        xclbin: Option<String>,
        /// Target KV cache quantization format (fp16, int8, int4, auto).
        #[arg(long, default_value = "auto")]
        kv_cache_type: String,
        /// Disable on-chip SRAM router matrix pinning for MoE models.
        #[arg(long)]
        no_router_sram: bool,
        /// Maximum on-chip SRAM budget ceiling for router matrices in megabytes (default: 32).
        #[arg(long, default_value = "32")]
        router_sram_limit_mb: usize,
        /// Disable on-the-fly orthogonal rotation during streaming BiLLM conversion.
        #[arg(long)]
        no_rotation: bool,
        /// Salient weight ratio for BiLLM scale calculation (default: 0.015 / ~1.5%).
        #[arg(long, default_value = "0.015")]
        salient_ratio: f32,
    },
    /// List available models in standard search locations (/opt/models, local test_models).
    List,
}

/// Run model CLI with the given command line arguments.
pub fn run_model_cli<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { 1 } else { 0 };
        }
    };

    match cli.command {
        Commands::Info {
            model_path,
            kv_cache_type,
        } => {
            let p = Path::new(&model_path);
            if !p.exists() {
                eprintln!("\x1b[31mError: Model file does not exist: {}\x1b[0m", model_path);
                return 1;
            }

            println!("============================================================");
            println!(" Model Information: {}", p.file_name().unwrap_or_default().to_string_lossy());
            println!("============================================================");

            let req_kv = KvCacheQuantType::parse(&kv_cache_type).unwrap_or(KvCacheQuantType::Auto);
            let mut base_quant = "Q4_K_M".to_string();

            if model_path.ends_with(".gguf") {
                match GgufModelReader::open(p) {
                    Ok(reader) => {
                        println!(" Format            : GGUF v3");
                        println!(" Architecture      : {}", reader.arch_name);
                        println!(" Hidden Dimension  : {}", reader.hyperparams.hidden_dim);
                        println!(" Attention Heads   : {}", reader.hyperparams.num_heads);
                        println!(" KV Heads          : {}", reader.hyperparams.num_kv_heads);
                        println!(" Layers            : {}", reader.hyperparams.num_layers);
                        println!(" Vocabulary Size   : {}", reader.hyperparams.vocab_size);
                        println!(" Context Length    : {}", reader.hyperparams.context_length);
                        println!(" Total Tensors     : {}", reader.tensor_count());
                        println!(" Embedded XCLBIN   : None (Raw GGUF format)");

                        if let Some(first_tensor) = reader.tensor_list.first() {
                            base_quant = format!("{:?}", first_tensor.dtype);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to parse GGUF: {}", e);
                        return 1;
                    }
                }
            } else {
                match Q4nxModel::open(p) {
                    Ok(model) => {
                        println!(" Format            : .q4nx Unified APU Container");
                        println!(" Architecture      : {}", model.header.arch_name);
                        println!(" Hidden Dimension  : {}", model.header.hyperparams.hidden_dim);
                        println!(" Attention Heads   : {}", model.header.hyperparams.num_heads);
                        println!(" KV Heads          : {}", model.header.hyperparams.num_kv_heads);
                        println!(" Layers            : {}", model.header.hyperparams.num_layers);
                        println!(" Vocabulary Size   : {}", model.header.hyperparams.vocab_size);
                        println!(" Embedded XCLBIN   : {}", if model.has_embedded_xclbin() { "YES (Turnkey APU deployment)" } else { "NO (Bare container)" });
                        if model.has_embedded_xclbin() {
                            println!(" XCLBIN Size       : {} bytes", model.header.xclbin_size);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to parse .q4nx: {}", e);
                        return 1;
                    }
                }
            }

            // APU Optimization Analysis (KV Quantization & Router SRAM)
            if let Ok(topo) = ModelGraphTopology::from_gguf(p).or_else(|_| ModelGraphTopology::from_q4nx(p)) {
                println!("------------------------------------------------------------");
                println!(" APU Hardware Optimization Compatibility:");

                let compat = evaluate_kv_quant_compatibility(
                    &topo.arch_name,
                    topo.head_dim as usize,
                    topo.context_length as usize,
                    &base_quant,
                    req_kv,
                );
                println!(" KV Cache Quant    : {}", compat.recommended_type);
                println!("   Reason/Detail   : {}", compat.reason);

                let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, topo.clone());
                let router_plan = builder.plan_router_sram();

                if topo.num_experts > 0 {
                    println!(" MoE Architecture  : Active ({} total experts)", topo.num_experts);
                    println!(" Router Matrix SRAM: {} KB pinned of {} KB total ({}/{} layers in on-chip SRAM)",
                        router_plan.pinned_sram_bytes / 1024,
                        router_plan.total_router_bytes / 1024,
                        router_plan.pinned_layer_count,
                        router_plan.total_layers,
                    );
                    if router_plan.sram_exhaustion_prevented {
                        println!("   Safety Ceiling  : ENFORCED (32 MB max allocated to prevent tile buffer exhaustion)");
                    } else {
                        println!("   Safety Ceiling  : Within 32 MB limit (100% router matrices pinned)");
                    }
                } else {
                    println!(" MoE Architecture  : Dense (No routing matrices to pin)");
                }
            }

            println!("============================================================");
            0
        }
        Commands::Stamp {
            model_path,
            xclbin,
            output,
            no_router_sram,
            router_sram_limit_mb,
        } => {
            let p = Path::new(&model_path);
            let filename = match p.file_name() {
                Some(f) => f.to_string_lossy().to_string(),
                None => {
                    eprintln!("Invalid model path");
                    return 1;
                }
            };

            println!("Resolving XCLBIN profile for: {}", filename);
            let xclbin_path = xclbin.as_deref().map(Path::new);
            let xclbin_bytes = match resolve_xclbin_profile(&filename, xclbin_path, None) {
                Ok(profile) => {
                    println!("Matched profile: {}", profile.name);
                    match profile.read_bytes() {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("Failed to read XCLBIN: {}", e);
                            return 1;
                        }
                    }
                }
                Err(_) => {
                    println!("Synthesizing tailored AIE2P XCLBIN hardware graph...");
                    let topo = match ModelGraphTopology::from_gguf(p).or_else(|_| ModelGraphTopology::from_q4nx(p)) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Failed to parse model topology: {:?}", e);
                            return 1;
                        }
                    };
                    let max_bytes = router_sram_limit_mb.max(1) * 1024 * 1024;
                    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, topo)
                        .with_router_sram(!no_router_sram, Some(max_bytes));
                    match builder.generate_xclbin_bytes() {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("Failed to build XCLBIN: {:?}", e);
                            return 1;
                        }
                    }
                }
            };

            println!("Stamping container (XCLBIN size: {} bytes)...", xclbin_bytes.len());
            let out_dest = output.map(PathBuf::from);
            if model_path.ends_with(".q4nx") {
                let mut m = match Q4nxModel::open(p) {
                    Ok(model) => model,
                    Err(e) => {
                        eprintln!("Failed to open .q4nx container: {}", e);
                        return 1;
                    }
                };
                if let Err(e) = m.stamp_xclbin(&xclbin_bytes, out_dest.as_deref()) {
                    eprintln!("Failed to stamp XCLBIN: {}", e);
                    return 1;
                }
                println!("\x1b[32mSuccessfully stamped XCLBIN into .q4nx container!\x1b[0m");
            } else {
                let target = out_dest.unwrap_or_else(|| p.with_extension("q4nx"));
                if let Err(e) = load_or_convert_model(p, xclbin_path) {
                    eprintln!("Failed to convert and stamp: {}", e);
                    return 1;
                }
                println!("\x1b[32mSuccessfully generated and stamped: {}\x1b[0m", target.display());
            }
            0
        }
        Commands::Convert {
            input,
            output,
            format,
            quant,
            target,
            xclbin,
            kv_cache_type: _,
            no_router_sram,
            router_sram_limit_mb,
            no_rotation,
            salient_ratio,
        } => {
            let p = Path::new(&input);
            if !p.exists() {
                eprintln!("\x1b[31mError: Input model path does not exist: {}\x1b[0m", input);
                return 1;
            }

            let is_bare = format.eq_ignore_ascii_case("bare");
            if is_bare {
                eprintln!("\n\x1b[33m================================================================================");
                eprintln!("[WARNING] Non-standard format selected: '--format bare'.");
                eprintln!("Bare .q4nx containers omit the embedded GGUF metadata / turnkey XCLBIN layer.");
                eprintln!("This container will NOT be directly loadable by `llama-apu` / `apu-run`");
                eprintln!("without an adjacent companion .gguf file!");
                eprintln!("================================================================================\x1b[0m\n");
            }

            println!("============================================================");
            println!(" Converting Model to .q4nx Container");
            println!(" Input Path : {}", input);
            println!(" Target HW  : {}", target);
            println!(" Container  : {}", if is_bare { "Bare (requires companion .gguf)" } else { "Embedded GGUF/Q4NX (Default)" });
            println!(" Quantization: {}", quant.to_uppercase());
            if quant.eq_ignore_ascii_case("billm") {
                println!(" Orthogonal Rot. : {}", if no_rotation { "DISABLED" } else { "ENABLED (Block-RHT Walsh-Hadamard 128)" });
                println!(" Saliency Ratio  : {:.1}% isolated for scale factor", salient_ratio * 100.0);
            }
            println!("============================================================");

            let is_dir = p.is_dir();
            let is_safetensors_file = p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("safetensors")).unwrap_or(false);

            if is_dir || is_safetensors_file {
                let model_dir = if is_dir { p } else { p.parent().unwrap_or(p) };
                let out_path = output
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        let name = model_dir.file_name().unwrap_or_default().to_string_lossy();
                        model_dir.with_file_name(format!("{}.q4nx", name))
                    });

                println!("Detected Safetensors model directory: {}", model_dir.display());
                println!("Streaming and quantizing weights directly to disk...");

                // Execute the streaming quantizer engine
                let mut cmd = std::process::Command::new("python3");
                cmd.arg("old/converter/convert_to_billm.py")
                    .arg("--model-id").arg(model_dir)
                    .arg("--output").arg(&out_path)
                    .arg("--format").arg(if is_bare { "bare" } else { "embedded" })
                    .arg("--quant").arg(&quant)
                    .arg("--salient-ratio").arg(salient_ratio.to_string());

                if no_rotation {
                    cmd.arg("--no-rotation");
                }
                if let Some(x) = &xclbin {
                    cmd.arg("--xclbin").arg(x);
                }

                match cmd.status() {
                    Ok(status) if status.success() => {
                        println!("\x1b[32mSuccessfully converted model to .q4nx container: {}\x1b[0m", out_path.display());
                        return 0;
                    }
                    Ok(status) => {
                        eprintln!("\x1b[31mError: Streaming quantization failed with status: {}\x1b[0m", status);
                        return 1;
                    }
                    Err(e) => {
                        eprintln!("\x1b[31mError launching conversion process: {}\x1b[0m", e);
                        return 1;
                    }
                }
            }

            println!("Inspecting input model tensors & quantization formats...");
            match GgufModelReader::open(p) {
                Ok(reader) => {
                    println!("Model architecture: {}", reader.arch_name);
                    println!("Total tensors: {}", reader.tensor_count());
                    let has_iq4 = reader.tensor_list.iter().any(|t| matches!(t.dtype, crate::container::reader::TensorDType::IQ4_NL));
                    if has_iq4 {
                        println!("Note: Input contains IQ4_NL tensors. Non-linear codebooks will be dequantized into tile-interleaved representation.");
                    }
                }
                Err(e) => {
                    eprintln!("\x1b[31mError: Model failed quantization / APU validation: {}\x1b[0m", e);
                    return 1;
                }
            }

            let filename = match p.file_name() {
                Some(f) => f.to_string_lossy().to_string(),
                None => {
                    eprintln!("Invalid input filename");
                    return 1;
                }
            };
            let xclbin_path = xclbin.as_deref().map(Path::new);
            let xclbin_bytes = match resolve_xclbin_profile(&filename, xclbin_path, None) {
                Ok(profile) => {
                    println!("Matched hardware profile: {}", profile.name);
                    match profile.read_bytes() {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("Failed to read matched XCLBIN: {}", e);
                            return 1;
                        }
                    }
                }
                Err(_) => {
                    println!("Synthesizing tailored AIE2P XCLBIN hardware graph for target '{}'...", target);
                    let target_hw = match target.to_lowercase().as_str() {
                        "npu2-aie2p" | "aie2p" | "xdna2" => TargetHardware::Npu2Aie2p,
                        other => {
                            eprintln!("\x1b[33mWarning: Unknown target hardware '{}', defaulting to NPU2 (AIE2P)\x1b[0m", other);
                            TargetHardware::Npu2Aie2p
                        }
                    };
                    let topo = match ModelGraphTopology::from_gguf(p) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Failed to parse GGUF model topology: {:?}", e);
                            return 1;
                        }
                    };
                    let max_bytes = router_sram_limit_mb.max(1) * 1024 * 1024;
                    let builder = XclbinBuilder::new(target_hw, topo)
                        .with_router_sram(!no_router_sram, Some(max_bytes));
                    match builder.generate_xclbin_bytes() {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("Failed to synthesize XCLBIN: {:?}", e);
                            return 1;
                        }
                    }
                }
            };

            let out_path = output
                .map(PathBuf::from)
                .unwrap_or_else(|| crate::container::converter::get_q4nx_destination_path(p));

            println!("Writing .q4nx container to {}...", out_path.display());
            if let Err(e) = crate::container::converter::convert_gguf_to_q4nx(p, &out_path, &xclbin_bytes) {
                eprintln!("\x1b[31mError converting to .q4nx: {}\x1b[0m", e);
                return 1;
            }

            println!("\x1b[32mSuccessfully converted GGUF model to tile-interleaved .q4nx container with embedded XCLBIN!\x1b[0m");
            println!("Output: {}", out_path.display());
            0
        }
        Commands::List => {
            let search_dirs = [
                "/opt/models",
                "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models",
            ];

            println!("============================================================");
            println!(" Available Models on Local System");
            println!("============================================================");
            for dir in search_dirs {
                if let Ok(entries) = fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                        if ext.eq_ignore_ascii_case("gguf") || ext.eq_ignore_ascii_case("q4nx") {
                            let sz = entry.metadata().map(|m| m.len()).unwrap_or(0);
                            let sz_mb = sz / (1024 * 1024);
                            println!(" - {:<45} ({:>5} MB) [{}]", path.file_name().unwrap().to_string_lossy(), sz_mb, dir);
                        }
                    }
                }
            }
            println!("============================================================");
            0
        }
    }
}
