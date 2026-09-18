// SPDX-License-Identifier: Apache-2.0
use std::path::Path;
use zero_copy_model_runner::container::xclbin_builder::{
    ModelGraphTopology, TargetHardware, XclbinBuilder, XclbinFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: apu-synth <model_file.gguf|.q4nx> <output.xclbin> [enhanced|mimic]");
        std::process::exit(1);
    }

    let model_path = Path::new(&args[1]);
    let out_xclbin = Path::new(&args[2]);
    let format = if args.len() >= 4 && args[3].to_lowercase().contains("mimic") {
        XclbinFormat::MimicBuiltin
    } else {
        XclbinFormat::Enhanced
    };

    println!("Parsing model topology from: {}", model_path.display());
    let topo = ModelGraphTopology::from_gguf(model_path)
        .or_else(|_| ModelGraphTopology::from_q4nx(model_path))
        .expect("Failed to parse model topology");

    println!("Synthesizing {:?} XCLBIN for target NPU2 (AIE2P)...", format);
    let builder = XclbinBuilder::new(TargetHardware::Npu2Aie2p, topo).with_format(format);
    let saved = builder.build_standalone(out_xclbin).expect("Failed to synthesize XCLBIN");
    println!("Successfully generated: {}", saved.display());
}
