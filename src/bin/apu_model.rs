// SPDX-License-Identifier: Apache-2.0
//! Automated Model Manager and XCLBIN Stamper Tool (`apu-model`).

fn main() {
    let code = zero_copy_model_runner::model_cli::run_model_cli(std::env::args());
    std::process::exit(code);
}
