// SPDX-License-Identifier: Apache-2.0
//! Persistent `.imatrix` Storage, External Provenance Protection & Calibration Management.
//!
//! Handles adjacent `<model>.imatrix.gguf` files with strict provenance safety:
//! external/third-party imatrices remain read-only; native ones are updateable online.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

pub const LLAMA_APU_PROVENANCE_TAG: &str = "llama-apu-v1";

/// Provenance status of an imatrix file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImatrixProvenance {
    /// Created by llama-apu with full online write/update permission.
    NativeUpdateable,
    /// Third-party or pre-existing external file: strictly immutable/read-only.
    ExternalReadOnly,
    /// No imatrix file found on disk.
    NotPresent,
}

/// In-memory imatrix frequency storage and persistence manager.
#[derive(Debug, Clone)]
pub struct ImatrixStore {
    /// Path to adjacent imatrix file.
    file_path: PathBuf,
    /// Provenance flag.
    pub provenance: ImatrixProvenance,
    /// Layer-wise expert activation frequency counts.
    pub layer_counts: HashMap<usize, Vec<u64>>,
    /// Unsaved dirty in-memory updates.
    has_dirty_updates: bool,
}

impl ImatrixStore {
    /// Discover or resolve imatrix configuration for a given model path.
    pub fn open_or_discover<P: AsRef<Path>>(model_path: P) -> Self {
        let p = model_path.as_ref();
        let imatrix_path = Self::adjacent_imatrix_path(p);

        if !imatrix_path.exists() {
            return Self {
                file_path: imatrix_path,
                provenance: ImatrixProvenance::NotPresent,
                layer_counts: HashMap::new(),
                has_dirty_updates: false,
            };
        }

        // Check provenance metadata
        let (provenance, counts) = Self::read_imatrix_file(&imatrix_path);

        Self {
            file_path: imatrix_path,
            provenance,
            layer_counts: counts,
            has_dirty_updates: false,
        }
    }

    /// Compute adjacent imatrix path (e.g. `model.gguf` -> `model.imatrix.gguf` or `model.gguf.imatrix.gguf`).
    pub fn adjacent_imatrix_path<P: AsRef<Path>>(model_path: P) -> PathBuf {
        let p = model_path.as_ref();
        let file_str = p.to_string_lossy();
        if file_str.ends_with(".gguf") {
            PathBuf::from(format!("{}.imatrix.gguf", &file_str[..file_str.len() - 5]))
        } else if file_str.ends_with(".q4nx") {
            PathBuf::from(format!("{}.imatrix.gguf", &file_str[..file_str.len() - 5]))
        } else {
            PathBuf::from(format!("{}.imatrix.gguf", file_str))
        }
    }

    /// Check if this imatrix can be written to.
    pub fn is_updateable(&self) -> bool {
        self.provenance == ImatrixProvenance::NativeUpdateable || self.provenance == ImatrixProvenance::NotPresent
    }

    /// Read an imatrix file and determine provenance.
    fn read_imatrix_file(path: &Path) -> (ImatrixProvenance, HashMap<usize, Vec<u64>>) {
        let mut counts = HashMap::new();
        if let Ok(mut f) = File::open(path) {
            let mut buf = Vec::new();
            if f.read_to_end(&mut buf).is_ok() {
                // Check if our provenance tag is present in the binary/text metadata
                let provenance = if buf.windows(LLAMA_APU_PROVENANCE_TAG.len()).any(|w| w == LLAMA_APU_PROVENANCE_TAG.as_bytes()) {
                    ImatrixProvenance::NativeUpdateable
                } else {
                    ImatrixProvenance::ExternalReadOnly
                };

                // Simple JSON/binary payload parse fallback
                if let Ok(json_str) = String::from_utf8(buf) {
                    if let Ok(parsed) = serde_json::from_str::<HashMap<String, Vec<u64>>>(&json_str) {
                        for (k, v) in parsed {
                            if let Ok(layer) = k.parse::<usize>() {
                                counts.insert(layer, v);
                            }
                        }
                    }
                }
                return (provenance, counts);
            }
        }
        (ImatrixProvenance::ExternalReadOnly, counts)
    }

    /// Create or initialize a new native updateable `.imatrix.gguf` from analytical priors.
    pub fn create_native_imatrix(
        &mut self,
        layer_priors: HashMap<usize, Vec<f32>>,
    ) -> Result<(), io::Error> {
        self.layer_counts.clear();

        for (layer, priors) in layer_priors {
            let counts: Vec<u64> = priors.iter().map(|&p| (p * 100_000.0) as u64).collect();
            self.layer_counts.insert(layer, counts);
        }

        self.provenance = ImatrixProvenance::NativeUpdateable;
        self.has_dirty_updates = true;
        self.save_to_disk()
    }

    /// Record a live expert activation event in-memory.
    pub fn record_activation(&mut self, layer_idx: usize, expert_idx: usize) {
        if self.provenance == ImatrixProvenance::ExternalReadOnly {
            // Strict safety: never mutate external third-party imatrix priors on disk
            return;
        }

        let counts = self.layer_counts.entry(layer_idx).or_default();
        if expert_idx >= counts.len() {
            counts.resize(expert_idx + 1, 0);
        }
        counts[expert_idx] = counts[expert_idx].saturating_add(1);
        self.has_dirty_updates = true;
    }

    /// Persist in-memory counts to disk if updateable and dirty.
    pub fn save_to_disk(&mut self) -> Result<(), io::Error> {
        if !self.has_dirty_updates || self.provenance == ImatrixProvenance::ExternalReadOnly {
            return Ok(());
        }

        let mut out_map: HashMap<String, Vec<u64>> = HashMap::new();
        for (k, v) in &self.layer_counts {
            out_map.insert(k.to_string(), v.clone());
        }

        let json_body = serde_json::to_string_pretty(&out_map).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Include provenance tag header
        let mut file = File::create(&self.file_path)?;
        writeln!(file, "# PROVENANCE: {}", LLAMA_APU_PROVENANCE_TAG)?;
        file.write_all(json_body.as_bytes())?;
        self.has_dirty_updates = false;
        Ok(())
    }
}
