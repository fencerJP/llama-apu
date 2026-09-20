// SPDX-License-Identifier: Apache-2.0
//! Automated Model Manager and XCLBIN Stamper Module (`model_cli`).
//!
//! Provides inspection, XCLBIN microcode stamping, conversion, and validation
//! for GGUF and `.q4nx` models on AMD Ryzen AI APUs.

use std::fs;
use std::path::{Path, PathBuf};
use clap::{Parser, Subcommand};

use crate::container::converter::load_or_convert_model;
use crate::container::reader::GgufModelReader;
use crate::container::resolver::resolve_xclbin_profile;
use crate::container::xclbin_builder::{ModelGraphTopology, TargetHardware, XclbinBuilder};
use crate::container::Q4nxModel;

#[derive(Parser)]
#[command(name = "apu-model")]
#[command(about = "AMD Ryzen AI APU Model Management and XCLBIN Stamping Utility")]
#[command(version = "0.4.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
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

/// Run model CLI with the given command line arguments.
pub fn run_model_cli<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e);
            return 1;
        }
    };

    match cli.command {
        Commands::Info { model_path } => {
            let p = Path::new(&model_path);
            if !p.exists() {
                eprintln!("\x1b[31mError: Model file does not exist: {}\x1b[0m", model_path);
                return 1;
            }

            println!("============================================================");
            println!(" Model Information: {}", p.file_name().unwrap_or_default().to_string_lossy());
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
            println!("============================================================");
            0
        }
        Commands::Stamp {
            model_path,
            xclbin,
            output,
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
                    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, topo);
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
                match Q4nxModel::open(p) {
                    Ok(mut m) => {
                        if let Err(e) = m.stamp_xclbin(&xclbin_bytes, out_dest.as_deref()) {
                            eprintln!("Stamp XCLBIN failed: {}", e);
                            return 1;
                        }
                        println!("\x1b[32mSuccessfully stamped XCLBIN into .q4nx container!\x1b[0m");
                    }
                    Err(e) => {
                        eprintln!("Open .q4nx failed: {}", e);
                        return 1;
                    }
                }
            } else {
                let target = out_dest.unwrap_or_else(|| p.with_extension("q4nx"));
                if let Err(e) = load_or_convert_model(p, xclbin_path) {
                    eprintln!("Convert and stamp failed: {:?}", e);
                    return 1;
                }
                println!("\x1b[32mSuccessfully generated and stamped: {}\x1b[0m", target.display());
            }
            0
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
                return 1;
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
                    eprintln!("Invalid input path");
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
                            eprintln!("Failed to read XCLBIN: {}", e);
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
                    let builder = XclbinBuilder::new(target_hw, topo);
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
                            println!(" - {:<45} ({:>5} MB) [{}]", path.file_name().unwrap_or_default().to_string_lossy(), sz_mb, dir);
                        }
                    }
                }
            }
            println!("============================================================");
            0
        }
    }
}
