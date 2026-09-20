// SPDX-License-Identifier: Apache-2.0
//! Hardware Routing Engine and APU Policy Selector.
//!
//! Evaluates physical AMD Ryzen AI APU silicon (Strix Point, Gorgon Point, Krackan Point,
//! Strix Halo) and resolves optimal accelerator mapping across RDNA 3.5 iGPU,
//! XDNA 2 AIE2P NPU, and Zen 5 AVX-512 Classic CPU cores.
//!
//! Supports macro accelerator presets (`--gpu-based`, `--cpu-based`, `--npu-based`)
//! and granular per-phase overrides (`--tokenize`, `--prefill`, `--decode`, `--sample`).

use std::fmt;
use std::fs;
use std::path::Path;

/// Target compute accelerator device for a single inference pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
pub enum DeviceTarget {
    #[value(name = "cpu")]
    Cpu,
    #[value(name = "gpu")]
    Gpu,
    #[value(name = "npu")]
    Npu,
}

impl fmt::Display for DeviceTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceTarget::Cpu => write!(f, "CPU (Zen 5 AVX-512)"),
            DeviceTarget::Gpu => write!(f, "iGPU (RDNA 3.5)"),
            DeviceTarget::Npu => write!(f, "NPU (XDNA 2 AIE2P)"),
        }
    }
}

/// AMD Ryzen AI Silicon Family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiliconFamily {
    StrixPoint,
    GorgonPoint,
    KrackanPoint,
    StrixHalo,
    GenericApu,
    NonApu,
}

impl fmt::Display for SiliconFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SiliconFamily::StrixPoint => write!(f, "AMD Strix Point (16 CUs RDNA 3.5, 32-tile AIE2P, 136 GB/s UMA)"),
            SiliconFamily::GorgonPoint => write!(f, "AMD Gorgon Point (16 CUs RDNA 3.5, 32-tile AIE2P, 136 GB/s UMA)"),
            SiliconFamily::KrackanPoint => write!(f, "AMD Krackan Point (8 CUs RDNA 3.5, 16-tile AIE2P, 120 GB/s UMA)"),
            SiliconFamily::StrixHalo => write!(f, "AMD Strix Halo (40 CUs RDNA 3.5, 32-tile AIE2P, 273 GB/s 256-bit UMA)"),
            SiliconFamily::GenericApu => write!(f, "AMD Ryzen AI APU (XDNA 2 Silicon)"),
            SiliconFamily::NonApu => write!(f, "Non-APU / Virtualized Host"),
        }
    }
}

/// Resolved multi-stage hardware routing policy for an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingPolicy {
    pub tokenize: DeviceTarget,
    pub prefill: DeviceTarget,
    pub decode: DeviceTarget,
    pub sample: DeviceTarget,
    pub silicon: SiliconFamily,
}

impl RoutingPolicy {
    /// Detect underlying silicon and establish ideal baseline routing policy.
    pub fn detect_silicon() -> SiliconFamily {
        let cpu_info = fs::read_to_string("/proc/cpuinfo").unwrap_or_default().to_lowercase();

        if cpu_info.contains("395") || cpu_info.contains("strix halo") {
            SiliconFamily::StrixHalo
        } else if cpu_info.contains("470") || cpu_info.contains("gorgon") {
            SiliconFamily::GorgonPoint
        } else if cpu_info.contains("370") || cpu_info.contains("365") || cpu_info.contains("strix") {
            SiliconFamily::StrixPoint
        } else if cpu_info.contains("krackan") || cpu_info.contains("kraken") {
            SiliconFamily::KrackanPoint
        } else if Path::new("/dev/accel/accel0").exists() || Path::new("/dev/kfd").exists() {
            SiliconFamily::GenericApu
        } else {
            SiliconFamily::NonApu
        }
    }

    /// Resolve policy according to silicon matrix defaults, macro flags, and granular overrides.
    pub fn resolve(
        gpu_based: bool,
        cpu_based: bool,
        npu_based: bool,
        tokenize_override: Option<DeviceTarget>,
        prefill_override: Option<DeviceTarget>,
        decode_override: Option<DeviceTarget>,
        sample_override: Option<DeviceTarget>,
    ) -> Self {
        let silicon = Self::detect_silicon();

        // Baseline silicon defaults:
        // Strix Point / Gorgon Point / Krackan Point: Prefill on iGPU, Decode on NPU.
        // Strix Halo: Prefill on iGPU, Decode on iGPU (40 CUs RDNA 3.5 on 273 GB/s bandwidth).
        // Fallback / Non-APU: CPU.
        let (mut def_tok, mut def_pref, mut def_dec, mut def_samp) = match silicon {
            SiliconFamily::StrixHalo => (
                DeviceTarget::Cpu,
                DeviceTarget::Gpu,
                DeviceTarget::Gpu,
                DeviceTarget::Cpu,
            ),
            SiliconFamily::StrixPoint | SiliconFamily::GorgonPoint | SiliconFamily::KrackanPoint | SiliconFamily::GenericApu => (
                DeviceTarget::Cpu,
                DeviceTarget::Gpu,
                DeviceTarget::Npu,
                DeviceTarget::Cpu,
            ),
            SiliconFamily::NonApu => (
                DeviceTarget::Cpu,
                DeviceTarget::Cpu,
                DeviceTarget::Cpu,
                DeviceTarget::Cpu,
            ),
        };

        // Macro accelerator presets
        if cpu_based {
            def_tok = DeviceTarget::Cpu;
            def_pref = DeviceTarget::Cpu;
            def_dec = DeviceTarget::Cpu;
            def_samp = DeviceTarget::Cpu;
        } else if gpu_based {
            def_tok = DeviceTarget::Gpu;
            def_pref = DeviceTarget::Gpu;
            def_dec = DeviceTarget::Gpu;
            def_samp = DeviceTarget::Gpu;
        } else if npu_based {
            def_tok = DeviceTarget::Cpu;
            def_pref = DeviceTarget::Gpu;
            def_dec = DeviceTarget::Npu;
            def_samp = DeviceTarget::Cpu;
        }

        // Granular step overrides take highest precedence
        let tokenize = tokenize_override.unwrap_or(def_tok);
        let prefill = prefill_override.unwrap_or(def_pref);
        let decode = decode_override.unwrap_or(def_dec);
        let sample = sample_override.unwrap_or(def_samp);

        Self {
            tokenize,
            prefill,
            decode,
            sample,
            silicon,
        }
    }

    /// Print routing telemetry and configuration report.
    pub fn print_telemetry(&self, verbose: bool) {
        if verbose {
            eprintln!("============================================================");
            eprintln!(" AMD Ryzen AI APU Zero-Copy Hardware Routing Telemetry");
            eprintln!("============================================================");
            eprintln!(" Silicon Architecture : {}", self.silicon);
            eprintln!(" Tokenize Stage       : {}", self.tokenize);
            eprintln!(" Prefill Stage (GEMM) : {}", self.prefill);
            eprintln!(" Decode Stage (GEMV)  : {}", self.decode);
            eprintln!(" Sampler Stage        : {}", self.sample);
            eprintln!(" Zero-Copy Bridge     : Linux DMA-BUF + AMDGPU Prime (64-byte aligned)");
            eprintln!(" Synchronization      : DRM syncobj timeline fences");
            eprintln!("============================================================");
        }
    }
}
