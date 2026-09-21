// SPDX-License-Identifier: Apache-2.0
//! Turnkey Model Converter and XCLBIN Stamper Tool (`llama-convert`).
//!
//! Provides unified conversion for GGUF files, Hugging Face checkpoint directories,
//! and Safetensors shard sets into tile-interleaved `.q4nx` containers with embedded
//! AMD XDNA 2 hardware microcode and on-the-fly BiLLM orthogonal rotations.

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    // If invoked as llama-convert directly, route transparently to convert subcommand if not provided
    if args.len() > 1 && !["convert", "info", "stamp", "list", "--help", "-h", "--version", "-V"].contains(&args[1].as_str()) {
        args.insert(1, "convert".to_string());
    }
    let code = zero_copy_model_runner::model_cli::run_model_cli(args);
    std::process::exit(code);
}
