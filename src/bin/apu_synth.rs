// SPDX-License-Identifier: Apache-2.0
//! XCLBIN Hardware Graph Synthesizer Tool (`apu-synth`).

fn main() {
    let code = zero_copy_model_runner::synth_cli::run_synth_cli(std::env::args());
    std::process::exit(code);
}
