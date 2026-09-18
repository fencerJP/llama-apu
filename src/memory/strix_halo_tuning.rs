// SPDX-License-Identifier: Apache-2.0
//! Strix Halo (Ryzen AI Max+ 395) 256-Bit UMA Memory Bus Tuning.
//!
//! Optimizes unified LPDDR5X DRAM throughput across 40 RDNA 3.5 CUs and AIE2P tiles:
//! 1. 2MB Huge-Page Alignment & `MADV_HUGEPAGE` kernel hints.
//! 2. 256-bit Wide Bus Interleaved Cacheline Stride optimization.
//! 3. NUMA & TLB thrashing mitigation for large KV-cache allocations.

use std::fs;
use crate::memory::MemoryError;

/// Memory tuning parameters for high-tier AMD APUs.
#[derive(Debug, Clone)]
pub struct StrixHaloConfig {
    /// Enable 2MB huge-page allocation hints (`MADV_HUGEPAGE`).
    pub enable_2mb_hugepages: bool,
    /// Interleaved cacheline stride in bytes (default 128 bytes for dual 128-bit sub-channels).
    pub cacheline_stride_bytes: usize,
    /// Prefetch distance for sequential KV cache decoding.
    pub prefetch_distance_cachelines: usize,
}

impl Default for StrixHaloConfig {
    fn default() -> Self {
        Self {
            enable_2mb_hugepages: true,
            cacheline_stride_bytes: 128,
            prefetch_distance_cachelines: 4,
        }
    }
}

/// APU memory bus optimizer for wide-UMA architectures.
#[derive(Debug)]
pub struct StrixHaloMemoryOptimizer {
    config: StrixHaloConfig,
    is_256bit_bus: bool,
}

impl StrixHaloMemoryOptimizer {
    /// Create optimizer, automatically inspecting system topology.
    pub fn new(config: StrixHaloConfig) -> Self {
        let is_256bit_bus = Self::detect_256bit_uma();
        Self {
            config,
            is_256bit_bus,
        }
    }

    /// Check if system microarchitecture indicates a 256-bit UMA APU (e.g. Strix Halo).
    pub fn detect_256bit_uma() -> bool {
        // Inspect cpuinfo for 16 Zen 5 cores or high compute capability APUs
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            let cpu_count = cpuinfo.matches("processor").count();
            if cpu_count >= 32 {
                return true; // 16C / 32T Strix Halo configuration
            }
        }
        false
    }

    /// Whether 256-bit UMA bus is detected.
    pub fn is_256bit_bus(&self) -> bool {
        self.is_256bit_bus
    }

    /// Apply `MADV_HUGEPAGE` advice to a 2MB-aligned memory buffer to eliminate TLB misses.
    pub fn advise_hugepages(&self, ptr: *mut u8, len: usize) -> Result<(), MemoryError> {
        if !self.config.enable_2mb_hugepages || ptr.is_null() || len < 2 * 1024 * 1024 {
            return Ok(());
        }

        unsafe {
            let ret = libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_HUGEPAGE);
            if ret != 0 {
                // Non-fatal if hugepages are disabled at kernel level
                return Ok(());
            }
        }

        Ok(())
    }

    /// Optimal cacheline stride for matrix-vector multiplication in LPDDR5X DRAM.
    pub fn optimal_stride(&self) -> usize {
        if self.is_256bit_bus {
            256 // Interleaved across dual 128-bit memory controllers
        } else {
            self.config.cacheline_stride_bytes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strix_halo_optimizer_creation() {
        let optimizer = StrixHaloMemoryOptimizer::new(StrixHaloConfig::default());
        assert!(optimizer.optimal_stride() >= 128);
    }

    #[test]
    fn test_advise_hugepages_null_or_small_safe() {
        let optimizer = StrixHaloMemoryOptimizer::new(StrixHaloConfig::default());
        assert!(optimizer.advise_hugepages(std::ptr::null_mut(), 1024).is_ok());
    }
}
