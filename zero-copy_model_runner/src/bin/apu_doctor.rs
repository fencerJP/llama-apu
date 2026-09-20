// SPDX-License-Identifier: Apache-2.0
//! AMD Ryzen AI APU Hardware Diagnostic Tool (`apu-doctor`).

fn main() {
    zero_copy_model_runner::doctor::print_doctor_report();
}
