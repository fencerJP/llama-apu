// SPDX-License-Identifier: Apache-2.0
//! APU Topology Governor & Microarchitectural Core Affinity.
//!
//! Pinning latency-critical threads away from asymmetric core migration boundaries
//! (Zen 5 Classic vs. Zen 5c Compact) to prevent L3 cache flushes and ITL jitter.

use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TopologyError {
    #[error("Failed to parse /sys CPU topology: {0}")]
    SysfsError(#[from] std::io::Error),
    #[error("Failed to set thread affinity: {0}")]
    AffinityFailed(#[from] nix::Error),
    #[error("No suitable cores found for role: {0}")]
    NoMatchingCores(String),
}

/// Specialized worker roles in the heterogeneous APU pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// Latency-critical tokenization and prefill feeder thread.
    /// Pin exclusively to Zen 5 Classic cores (large L3 cache, full AVX-512).
    Feeder,
    /// Low-power asynchronous event wait thread (/dev/accel0 event polling).
    /// Pin to Zen 5c Compact cores to allow Classic cores to enter deep C-states.
    EventPoller,
    /// High-throughput HTTP API and SSE network dispatch worker.
    NetworkWorker,
}

/// Core microarchitecture classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMicroarchitecture {
    /// AMD Zen 5 Classic: Full frequency, full L3 cache slice (e.g. 16MB/CCX), dual-pipe AVX-512.
    Zen5Classic,
    /// AMD Zen 5c Compact: Compact cell library, shared compact L3 (e.g. 8MB), power-optimized.
    Zen5cCompact,
    /// Standard symmetric x86_64 core.
    GenericX86,
}

/// Details of an individual logical CPU core discovered in the system.
#[derive(Debug, Clone)]
pub struct CpuCoreInfo {
    pub cpu_id: usize,
    pub ccx_id: usize,
    pub max_frequency_khz: u64,
    pub l3_cache_size_kb: usize,
    pub microarchitecture: CoreMicroarchitecture,
}

/// Governor maintaining the APU topology map and enforcing thread affinity policies.
#[derive(Debug)]
pub struct ApuTopologyGovernor {
    cores: Vec<CpuCoreInfo>,
    zen5_classic_cores: Vec<usize>,
    zen5c_compact_cores: Vec<usize>,
}

impl ApuTopologyGovernor {
    /// Probe the Linux sysfs hierarchy (`/sys/devices/system/cpu/`) to detect
    /// heterogeneous core asymmetric clusters (frequency, L3 cache, package layout).
    pub fn probe_system() -> Result<Self, TopologyError> {
        let mut cores = Vec::new();
        let cpu_dir = Path::new("/sys/devices/system/cpu");

        let mut cpu_id = 0;
        loop {
            let cpu_path = cpu_dir.join(format!("cpu{}", cpu_id));
            if !cpu_path.exists() {
                break;
            }

            // Read maximum scaling frequency
            let max_freq_path = cpu_path.join("cpufreq/cpuinfo_max_freq");
            let max_frequency_khz = if max_freq_path.exists() {
                fs::read_to_string(max_freq_path)
                    .unwrap_or_default()
                    .trim()
                    .parse::<u64>()
                    .unwrap_or(0)
            } else {
                0
            };

            // Read L3 cache size from cache/index3
            let l3_size_path = cpu_path.join("cache/index3/size");
            let l3_cache_size_kb = if l3_size_path.exists() {
                let size_str = fs::read_to_string(l3_size_path).unwrap_or_default();
                Self::parse_size_kb(size_str.trim())
            } else {
                0
            };

            // Read core/package ID
            let package_id_path = cpu_path.join("topology/physical_package_id");
            let ccx_id = if package_id_path.exists() {
                fs::read_to_string(package_id_path)
                    .unwrap_or_default()
                    .trim()
                    .parse::<usize>()
                    .unwrap_or(0)
            } else {
                0
            };

            // Heuristic classification:
            // Zen 5 Classic cores feature higher max boost frequency and dedicated L3 slices,
            // while Zen 5c cores feature reduced max clock (typically 3.3-3.7 GHz vs 5.0+ GHz).
            let microarchitecture = if max_frequency_khz > 4_500_000 {
                CoreMicroarchitecture::Zen5Classic
            } else if max_frequency_khz > 0 && max_frequency_khz <= 3_800_000 {
                CoreMicroarchitecture::Zen5cCompact
            } else if cpu_id < 4 {
                // Typical Strix Point configuration: first 4 cores are Zen 5, remaining are Zen 5c
                CoreMicroarchitecture::Zen5Classic
            } else {
                CoreMicroarchitecture::Zen5cCompact
            };

            cores.push(CpuCoreInfo {
                cpu_id,
                ccx_id,
                max_frequency_khz,
                l3_cache_size_kb,
                microarchitecture,
            });

            cpu_id += 1;
        }

        if cores.is_empty() {
            // Fallback for containerized or mock environments
            cores.push(CpuCoreInfo {
                cpu_id: 0,
                ccx_id: 0,
                max_frequency_khz: 5_100_000,
                l3_cache_size_kb: 16384,
                microarchitecture: CoreMicroarchitecture::Zen5Classic,
            });
        }

        let mut zen5_classic_cores = Vec::new();
        let mut zen5c_compact_cores = Vec::new();

        for core in &cores {
            match core.microarchitecture {
                CoreMicroarchitecture::Zen5Classic => zen5_classic_cores.push(core.cpu_id),
                CoreMicroarchitecture::Zen5cCompact => zen5c_compact_cores.push(core.cpu_id),
                CoreMicroarchitecture::GenericX86 => zen5_classic_cores.push(core.cpu_id),
            }
        }

        // If no Zen 5c detected, mirror classic cores
        if zen5c_compact_cores.is_empty() {
            zen5c_compact_cores = zen5_classic_cores.clone();
        }

        Ok(Self {
            cores,
            zen5_classic_cores,
            zen5c_compact_cores,
        })
    }

    /// Pin the calling OS thread strictly to CPU cores assigned to `role`.
    pub fn pin_calling_thread(&self, role: WorkerRole) -> Result<(), TopologyError> {
        let target_cpus = match role {
            WorkerRole::Feeder => &self.zen5_classic_cores,
            WorkerRole::EventPoller => &self.zen5c_compact_cores,
            WorkerRole::NetworkWorker => &self.zen5c_compact_cores,
        };

        if target_cpus.is_empty() {
            return Err(TopologyError::NoMatchingCores(format!("{:?}", role)));
        }

        let mut cpuset = nix::sched::CpuSet::new();
        for &cpu in target_cpus {
            cpuset.set(cpu)?;
        }

        nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), &cpuset)?;
        tracing::info!(
            "Successfully pinned thread for role {:?} to CPU set: {:?}",
            role,
            target_cpus
        );

        Ok(())
    }

    /// Enumerate all detected CPU core information.
    pub fn cores(&self) -> &[CpuCoreInfo] {
        &self.cores
    }

    /// Enumerate all detected Zen 5 Classic core IDs.
    pub fn zen5_classic_cores(&self) -> &[usize] {
        &self.zen5_classic_cores
    }

    /// Enumerate all detected Zen 5c Compact core IDs.
    pub fn zen5c_compact_cores(&self) -> &[usize] {
        &self.zen5c_compact_cores
    }

    fn parse_size_kb(size_str: &str) -> usize {
        if size_str.ends_with('K') || size_str.ends_with('k') {
            size_str[..size_str.len() - 1].parse::<usize>().unwrap_or(0)
        } else if size_str.ends_with('M') || size_str.ends_with('m') {
            size_str[..size_str.len() - 1]
                .parse::<usize>()
                .map(|m| m * 1024)
                .unwrap_or(0)
        } else {
            size_str.parse::<usize>().unwrap_or(0)
        }
    }
}
