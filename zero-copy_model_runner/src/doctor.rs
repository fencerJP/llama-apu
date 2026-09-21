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
        details: if has_kfd { "Present (ROCm/HIP Compute Ready)".into() } else { "Not found".into() },
        recommendation: if has_kfd {
            None
        } else {
            Some("ROCm compute requires /dev/kfd. Ensure user is in 'render' or 'kfd' group.".into())
        },
    });

    // 4. Check AMD XDNA 2 NPU Node (/dev/accel/accel0)
    let has_accel0 = Path::new("/dev/accel/accel0").exists();
    results.push(CheckResult {
        name: "AMD XDNA 2 NPU Node (/dev/accel/accel0)",
        status: has_accel0,
        details: if has_accel0 { "Present (amdxdna driver active)".into() } else { "Not found (Fallback to UAPI Mock)".into() },
        recommendation: if has_accel0 {
            None
        } else {
            Some("NPU requires amdxdna driver. Check 'dmesg | grep -i amdxdna' or load amdxdna kernel module.".into())
        },
    });

    // 5. Check ROCm / HIP runtime libraries
    let rocm_paths = [
        "/opt/rocm/lib/libhiprtc.so",
        "/opt/rocm/lib/libamdhip64.so",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
    ];
    let has_rocm = rocm_paths.iter().any(|p| Path::new(p).exists());
    results.push(CheckResult {
        name: "ROCm / HIP Runtime Libraries",
        status: has_rocm,
        details: if has_rocm { "Installed".into() } else { "Not found in standard paths".into() },
        recommendation: if has_rocm {
            None
        } else {
            Some("Install ROCm runtime: 'sudo apt install rocm-hip-runtime' or consult https://rocm.docs.amd.com".into())
        },
    });

    // 6. Check user group permissions (render, video)
    let groups_output = Command::new("id").arg("-Gn").output().ok();
    let groups_str = groups_output
        .as_ref()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let in_render = groups_str.contains("render");
    let in_video = groups_str.contains("video");
    let perms_ok = in_render && in_video;

    results.push(CheckResult {
        name: "User Hardware Permissions",
        status: perms_ok,
        details: format!("render: {}, video: {}", if in_render { "OK" } else { "Missing" }, if in_video { "OK" } else { "Missing" }),
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
