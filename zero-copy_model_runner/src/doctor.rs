// SPDX-License-Identifier: Apache-2.0
//! AMD Ryzen AI APU Hardware Diagnostic Module (`doctor`).
//!
//! Validates kernel UAPI interfaces, DRM render nodes, ROCm/HIP installations,
//! XDNA 2 NPU accelerators, and user group permissions on Linux.

use std::fs;
use std::path::Path;
use std::process::Command;

/// Diagnostic check result.
pub struct CheckResult {
    pub name: &'static str,
    pub status: bool,
    pub details: String,
    pub recommendation: Option<String>,
}

/// Run comprehensive system hardware and driver diagnosis.
pub fn run_diagnostics() -> Vec<CheckResult> {
    let mut results = Vec::new();

    // 1. Check CPU AVX-512 & Architecture
    let cpu_info = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let has_avx512 = cpu_info.contains("avx512f") || cpu_info.contains("avx512");
    let model_name = cpu_info
        .lines()
        .find(|l| l.starts_with("model name"))
        .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    results.push(CheckResult {
        name: "Zen 5 CPU & AVX-512 SIMD",
        status: has_avx512,
        details: format!("{} (AVX-512: {})", model_name, if has_avx512 { "Supported" } else { "Missing" }),
        recommendation: if has_avx512 {
            None
        } else {
            Some("AMD Zen 4 or Zen 5 CPU with AVX-512 is recommended for optimal sampler/CPU performance.".into())
        },
    });

    // 2. Check iGPU Render Node (/dev/dri/renderD128)
    let has_render128 = Path::new("/dev/dri/renderD128").exists();
    results.push(CheckResult {
        name: "AMDGPU DRM Render Node (/dev/dri/renderD128)",
        status: has_render128,
        details: if has_render128 { "Present".into() } else { "Not found".into() },
        recommendation: if has_render128 {
            None
        } else {
            Some("Ensure amdgpu kernel driver is loaded: 'modprobe amdgpu'".into())
        },
    });

    // 3. Check KFD Compute Interface (/dev/kfd)
    let has_kfd = Path::new("/dev/kfd").exists();
    results.push(CheckResult {
        name: "AMDGPU KFD Compute Node (/dev/kfd)",
        status: has_kfd,
        details: if has_kfd { "Present".into() } else { "Not found".into() },
        recommendation: if has_kfd {
            None
        } else {
            Some("Verify AMDGPU ROCm KFD kernel support is enabled.".into())
        },
    });

    // 4. Check XDNA 2 NPU Node (/dev/accel/accel0)
    let has_npu = Path::new("/dev/accel/accel0").exists();
    results.push(CheckResult {
        name: "AMD XDNA 2 NPU Node (/dev/accel/accel0)",
        status: has_npu,
        details: if has_npu {
            "Present (AMD XDNA 2 AIE2P Silicon)".into()
        } else {
            "Not found (Running in CPU or mock emulation mode)".into()
        },
        recommendation: if has_npu {
            None
        } else {
            Some("Verify the amdxdna Linux kernel driver is installed: 'lsmod | grep amdxdna'".into())
        },
    });

    // 5. Check ROCm / HIP Stack
    let has_rocm = Path::new("/opt/rocm").exists();
    let rocm_version = fs::read_to_string("/opt/rocm/.info/version")
        .or_else(|_| fs::read_to_string("/opt/rocm/version.txt"))
        .unwrap_or_else(|_| {
            if has_rocm { "Installed".into() } else { "Missing".into() }
        })
        .trim()
        .to_string();

    results.push(CheckResult {
        name: "ROCm / HIP Runtime Stack",
        status: has_rocm,
        details: format!("Path: /opt/rocm ({})", rocm_version),
        recommendation: if has_rocm {
            None
        } else {
            Some("Install AMD ROCm 6.x or 7.x: 'sudo apt install rocm-hip-sdk'".into())
        },
    });

    // 6. Check User Group Permissions (render & video)
    let id_output = Command::new("id")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let in_render = id_output.contains("render");
    let in_video = id_output.contains("video");
    let perms_ok = in_render && in_video;

    results.push(CheckResult {
        name: "User Hardware Permissions (render, video groups)",
        status: perms_ok,
        details: format!("render: {}, video: {}", if in_render { "OK" } else { "MISSING" }, if in_video { "OK" } else { "MISSING" }),
        recommendation: if perms_ok {
            None
        } else {
            Some("Add user to render and video groups: 'sudo usermod -a -G render,video $USER' and log back in.".into())
        },
    });

    results
}

/// Print formatted doctor report to console.
pub fn print_doctor_report() {
    println!("============================================================");
    println!(" AMD Ryzen AI APU Hardware & Runtime Diagnostic (Doctor)");
    println!("============================================================");

    let checks = run_diagnostics();
    let mut all_ok = true;

    for check in &checks {
        let status_str = if check.status {
            "\x1b[32m[ PASS ]\x1b[0m"
        } else {
            all_ok = false;
            "\x1b[31m[ FAIL ]\x1b[0m"
        };
        println!("{} {}: {}", status_str, check.name, check.details);
        if let Some(rec) = &check.recommendation {
            println!("         \x1b[33mRecommendation: {}\x1b[0m", rec);
        }
    }

    println!("============================================================");
    if all_ok {
        println!("\x1b[32mStatus: All checks passed. System ready for zero-copy APU inference!\x1b[0m");
    } else {
        println!("\x1b[33mStatus: Some checks failed. Zero-copy runner will fall back to CPU or mock.\x1b[0m");
    }
    println!("============================================================");
}
