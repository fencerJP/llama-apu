// SPDX-License-Identifier: Apache-2.0
//! XCLBIN Hardware Graph Synthesizer Module (`synth_cli`).
//!
//! Provides synthesis of tailored AIE2P XCLBIN hardware graphs for XDNA 2 NPUs.

use std::path::Path;
use crate::container::xclbin_builder::{
    ModelGraphTopology, TargetHardware, XclbinBuilder, XclbinFormat,
};

/// Run XCLBIN synthesis CLI with given command line arguments.
pub fn run_synth_cli<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let args: Vec<String> = args
        .into_iter()
        .map(|a| a.into().to_string_lossy().to_string())
        .collect();

    if args.len() < 3 {
        eprintln!("Usage: apu-synth <model_file.gguf|.q4nx> <output.xclbin> [enhanced|mimic]");
        return 1;
    }

    let model_path = Path::new(&args[1]);
    let out_xclbin = Path::new(&args[2]);
    let format = if args.len() >= 4 && args[3].to_lowercase().contains("mimic") {
        XclbinFormat::MimicBuiltin
    } else {
        XclbinFormat::Enhanced
    };

    println!("Parsing model topology from: {}", model_path.display());
    let topo = match ModelGraphTopology::from_gguf(model_path).or_else(|_| ModelGraphTopology::from_q4nx(model_path)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to parse model topology: {:?}", e);
            return 1;
        }
    };

    println!("Synthesizing {:?} XCLBIN for target NPU2 (AIE2P)...", format);
    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, topo).with_format(format);
    match builder.build_standalone(out_xclbin) {
        Ok(saved) => {
            println!("Successfully generated: {}", saved.display());
            0
        }
        Err(e) => {
            eprintln!("Failed to synthesize XCLBIN: {:?}", e);
            return 1;
        }
    }
}
