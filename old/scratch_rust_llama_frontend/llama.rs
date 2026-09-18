// SPDX-License-Identifier: Apache-2.0
//! Convenience alias binary for `llama-cli` on AMD Ryzen AI APUs.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    let current_exe = env::current_exe().unwrap_or_default();
    let target_bin = current_exe
        .parent()
        .map(|p| p.join("llama-cli"))
        .unwrap_or_else(|| PathBuf::from("llama-cli"));

    let mut cmd = Command::new(target_bin);
    if args.len() > 1 {
        cmd.args(&args[1..]);
    }

    match cmd.status() {
        Ok(status) => {
            std::process::exit(status.code().unwrap_or(0));
        }
        Err(e) => {
            eprintln!("Failed to execute llama-cli: {}", e);
            std::process::exit(1);
        }
    }
}
