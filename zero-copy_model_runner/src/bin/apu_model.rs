// SPDX-License-Identifier: Apache-2.0
//! Automated Model Manager and XCLBIN Stamper Tool (`apu-model`).
//!
//! Provides inspection, XCLBIN microcode stamping, conversion, and validation
//! for GGUF and `.q4nx` models on AMD Ryzen AI APUs.

use std::fs;
use std::path::{Path, PathBuf};
use clap::{Parser, Subcommand};

use zero_copy_model_runner::container::converter::load_or_convert_model;
use zero_copy_model_runner::container::reader::GgufModelReader;
use zero_copy_model_runner::container::resolver::resolve_xclbin_profile;
use zero_copy_model_runner::container::xclbin_builder::{ModelGraphTopology, TargetHardware, XclbinBuilder};
use zero_copy_model_runner::container::Q4nxModel;

#[derive(Parser)]
#[command(name = "apu-model")]
#[command(about = "AMD Ryzen AI APU Model Management and XCLBIN Stamping Utility")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect model metadata, hyperparameters, and embedded XCLBIN status.
    Info {
        /// Path to .gguf or .q4nx model file.
        model_path: String,
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
    },
    /// Convert a GGUF model to tile-interleaved .q4nx format with embedded XCLBIN.
    Convert {
        /// Input GGUF file.
        #[arg(short, long)]
        input: String,
        /// Destination .q4nx output file (defaults to adjacent .q4nx).
        #[arg(short, long)]
        output: Option<String>,
        /// Target hardware architecture (e.g. npu2-aie2p).
        #[arg(short, long, default_value = "npu2-aie2p")]
        target: String,
        /// Optional explicit path to .xclbin file.
        #[arg(long)]
        xclbin: Option<String>,
    },
    /// List available models in standard search locations (/opt/models, local test_models).
    List,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { model_path } => {
            let p = Path::new(&model_path);
            if !p.exists() {
                eprintln!("\x1b[31mError: Model file does not exist: {}\x1b[0m", model_path);
                std::process::exit(1);
            }

            println!("============================================================");
            println!(" Model Information: {}", p.file_name().unwrap().to_string_lossy());
            println!("============================================================");

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
                    }
                    Err(e) => eprintln!("Failed to parse GGUF: {}", e),
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
                    Err(e) => eprintln!("Failed to parse .q4nx: {}", e),
                }
            }
            println!("============================================================");
        }
        Commands::Stamp {
            model_path,
            xclbin,
            output,
        } => {
            let p = Path::new(&model_path);
            let filename = p.file_name().unwrap().to_string_lossy().to_string();

            println!("Resolving XCLBIN profile for: {}", filename);
            let xclbin_path = xclbin.as_deref().map(Path::new);
            let xclbin_bytes = match resolve_xclbin_profile(&filename, xclbin_path, None) {
                Ok(profile) => {
                    println!("Matched profile: {}", profile.name);
                    profile.read_bytes().expect("Failed to read XCLBIN")
                }
                Err(_) => {
                    println!("Synthesizing tailored AIE2P XCLBIN hardware graph...");
                    let topo = ModelGraphTopology::from_gguf(p)
                        .or_else(|_| ModelGraphTopology::from_q4nx(p))
                        .expect("Failed to parse model topology");
                    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, topo);
                    builder.generate_xclbin_bytes().expect("Failed to build XCLBIN")
                }
            };

            println!("Stamping container (XCLBIN size: {} bytes)...", xclbin_bytes.len());
            let out_dest = output.map(PathBuf::from);
            if model_path.ends_with(".q4nx") {
                let mut m = Q4nxModel::open(p).expect("Open .q4nx");
                m.stamp_xclbin(&xclbin_bytes, out_dest.as_deref()).expect("Stamp XCLBIN");
                println!("\x1b[32mSuccessfully stamped XCLBIN into .q4nx container!\x1b[0m");
            } else {
                let target = out_dest.unwrap_or_else(|| p.with_extension("q4nx"));
                load_or_convert_model(p, xclbin_path).expect("Convert and stamp");
                println!("\x1b[32mSuccessfully generated and stamped: {}\x1b[0m", target.display());
            }
        }
        Commands::Convert {
            input,
            output,
            target,
            xclbin,
        } => {
            let p = Path::new(&input);
            if !p.exists() {
                eprintln!("\x1b[31mError: Input model file does not exist: {}\x1b[0m", input);
                std::process::exit(1);
            }

            println!("============================================================");
            println!(" Converting Model to .q4nx Container");
            println!(" Input File : {}", input);
            println!(" Target HW  : {}", target);
            println!("============================================================");

            // Validate quantization support
            println!("Inspecting input model tensors & quantization formats...");
            match GgufModelReader::open(p) {
                Ok(reader) => {
                    println!("Model architecture: {}", reader.arch_name);
                    println!("Total tensors: {}", reader.tensor_count());
                    let has_iq4 = reader.tensor_list.iter().any(|t| matches!(t.dtype, zero_copy_model_runner::container::reader::TensorDType::IQ4_NL));
                    if has_iq4 {
                        println!("Note: Input contains IQ4_NL tensors. Non-linear codebooks will be dequantized into tile-interleaved representation.");
                    }
                }
                Err(e) => {
                    eprintln!("\x1b[31mError: Model failed quantization / APU validation: {}\x1b[0m", e);
                    std::process::exit(1);
                }
            }

            let filename = p.file_name().unwrap().to_string_lossy().to_string();
            let xclbin_path = xclbin.as_deref().map(Path::new);
            let xclbin_bytes = match resolve_xclbin_profile(&filename, xclbin_path, None) {
                Ok(profile) => {
                    println!("Matched hardware profile: {}", profile.name);
                    profile.read_bytes().expect("Failed to read XCLBIN")
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
                    let topo = ModelGraphTopology::from_gguf(p).expect("Failed to parse GGUF model topology");
                    let builder = XclbinBuilder::new(target_hw, topo);
                    builder.generate_xclbin_bytes().expect("Failed to synthesize XCLBIN")
                }
            };

            let out_path = output
                .map(PathBuf::from)
                .unwrap_or_else(|| zero_copy_model_runner::container::converter::get_q4nx_destination_path(p));

            println!("Writing .q4nx container to {}...", out_path.display());
            if let Err(e) = zero_copy_model_runner::container::converter::convert_gguf_to_q4nx(p, &out_path, &xclbin_bytes) {
                eprintln!("\x1b[31mError converting to .q4nx: {}\x1b[0m", e);
                std::process::exit(1);
            }

            println!("\x1b[32mSuccessfully converted GGUF model to tile-interleaved .q4nx container with embedded XCLBIN!\x1b[0m");
            println!("Output: {}", out_path.display());
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
        }
    }
}
