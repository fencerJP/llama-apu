// SPDX-License-Identifier: Apache-2.0
//! Unified `.q4nx` Binary Container Format with Embedded XCLBIN Header.
//!
//! Stores compiled AMD XDNA hardware microcode (`.xclbin`) directly inside
//! the model file header by default alongside 64-byte aligned quantized weight tensors.

pub mod converter;
pub mod reader;
pub mod resolver;
pub mod xclbin_builder;

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use thiserror::Error;

pub use converter::load_or_convert_model;
pub use reader::{f16_to_f32, GgufModelReader, ModelTokenizer, TensorDType, TensorDescriptor, TensorView};
pub use resolver::{discover_xclbin_profiles, resolve_xclbin_profile, XclbinProfile};
pub use xclbin_builder::{ModelGraphTopology, RouterSramPlan, TargetHardware, XclbinBuilder, XclbinFormat};

/// Magic bytes for the `.q4nx` container: 'Q', '4', 'N', 'X' (0x584E3451 in little-endian).
pub const Q4NX_MAGIC: [u8; 4] = [b'Q', b'4', b'N', b'X'];
pub const Q4NX_CURRENT_VERSION: u32 = 1;
pub const HEADER_FIXED_SIZE: usize = 256;

#[derive(Error, Debug)]
pub enum ContainerError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid Q4NX magic bytes: expected {expected:?}, found {found:?}")]
    InvalidMagic { expected: [u8; 4], found: [u8; 4] },
    #[error("Unsupported Q4NX container version: {0}")]
    UnsupportedVersion(u32),
    #[error("Buffer alignment error: payload must be 64-byte aligned (offset: {offset})")]
    AlignmentError { offset: u64 },
    #[error("Corrupted container header: {0}")]
    CorruptedHeader(String),
    #[error("Unsupported quantization format: {0}")]
    UnsupportedQuantization(String),
}

/// Model structural hyperparameters stored in the `.q4nx` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelHyperparameters {
    pub hidden_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
    pub vocab_size: u32,
    pub context_length: u32,
}

/// In-memory representation of a parsed `.q4nx` header.
#[derive(Debug, Clone)]
pub struct Q4nxHeader {
    pub version: u32,
    pub arch_name: String,
    pub hyperparams: ModelHyperparameters,
    pub xclbin_offset: u64,
    pub xclbin_size: u64,
    pub tensor_table_offset: u64,
    pub tensor_table_count: u64,
    pub payload_offset: u64,
    pub payload_size: u64,
}

/// A parsed, memory-mapped or open `.q4nx` model container.
pub struct Q4nxModel {
    pub header: Q4nxHeader,
    file_path: String,
    xclbin_data: Vec<u8>,
    payload_data: Vec<u8>,
}

impl Q4nxModel {
    /// Open and validate a `.q4nx` container from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ContainerError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut file = File::open(&path)?;

        let mut header_buf = [0u8; HEADER_FIXED_SIZE];
        file.read_exact(&mut header_buf)?;

        // Check magic
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&header_buf[0..4]);
        if magic != Q4NX_MAGIC {
            let header_len = u64::from_le_bytes(header_buf[0..8].try_into().unwrap());
            if header_len > 0 && header_len < 100_000_000 && header_buf[8] == b'{' {
                return Self::open_safetensors_q4nx(&mut file, header_len, &path_str);
            }
            return Err(ContainerError::InvalidMagic {
                expected: Q4NX_MAGIC,
                found: magic,
            });
        }

        let version = u32::from_le_bytes(header_buf[4..8].try_into().unwrap());
        if version != Q4NX_CURRENT_VERSION {
            return Err(ContainerError::UnsupportedVersion(version));
        }

        // Parse architecture string (32 bytes null-terminated)
        let arch_bytes = &header_buf[8..40];
        let arch_len = arch_bytes.iter().position(|&b| b == 0).unwrap_or(32);
        let arch_name = String::from_utf8_lossy(&arch_bytes[..arch_len]).to_string();

        let hidden_dim = u32::from_le_bytes(header_buf[40..44].try_into().unwrap());
        let num_heads = u32::from_le_bytes(header_buf[44..48].try_into().unwrap());
        let num_kv_heads = u32::from_le_bytes(header_buf[48..52].try_into().unwrap());
        let num_layers = u32::from_le_bytes(header_buf[52..56].try_into().unwrap());
        let vocab_size = u32::from_le_bytes(header_buf[56..60].try_into().unwrap());
        let context_length = u32::from_le_bytes(header_buf[60..64].try_into().unwrap());

        let xclbin_offset = u64::from_le_bytes(header_buf[64..72].try_into().unwrap());
        let xclbin_size = u64::from_le_bytes(header_buf[72..80].try_into().unwrap());
        let tensor_table_offset = u64::from_le_bytes(header_buf[80..88].try_into().unwrap());
        let tensor_table_count = u64::from_le_bytes(header_buf[88..96].try_into().unwrap());
        let payload_offset = u64::from_le_bytes(header_buf[96..104].try_into().unwrap());
        let payload_size = u64::from_le_bytes(header_buf[104..112].try_into().unwrap());

        if payload_offset % 64 != 0 {
            return Err(ContainerError::AlignmentError {
                offset: payload_offset,
            });
        }

        // Read embedded XCLBIN if present
        let mut xclbin_data = Vec::with_capacity(xclbin_size as usize);
        if xclbin_size > 0 {
            file.seek(SeekFrom::Start(xclbin_offset))?;
            let mut xclbin_buf = vec![0u8; xclbin_size as usize];
            file.read_exact(&mut xclbin_buf)?;
            xclbin_data = xclbin_buf;
        }

        // Read sample of payload data for inspection (up to 64KB or payload_size)
        file.seek(SeekFrom::Start(payload_offset))?;
        let sample_len = (payload_size as usize).min(65536);
        let mut payload_data = vec![0u8; sample_len];
        file.read_exact(&mut payload_data)?;

        Ok(Self {
            header: Q4nxHeader {
                version,
                arch_name,
                hyperparams: ModelHyperparameters {
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    num_layers,
                    vocab_size,
                    context_length,
                },
                xclbin_offset,
                xclbin_size,
                tensor_table_offset,
                tensor_table_count,
                payload_offset,
                payload_size,
            },
            file_path: path_str,
            xclbin_data,
            payload_data,
        })
    }

    /// Access the embedded XCLBIN binary blob.
    pub fn xclbin_bytes(&self) -> Option<&[u8]> {
        if self.xclbin_data.is_empty() {
            None
        } else {
            Some(&self.xclbin_data)
        }
    }

    /// Access the 64-byte aligned tile-interleaved weight tensor payload.
    pub fn tensor_payload(&self) -> &[u8] {
        &self.payload_data
    }

    /// Check if this `.q4nx` container has an embedded XCLBIN.
    pub fn has_embedded_xclbin(&self) -> bool {
        self.header.xclbin_size > 0 && !self.xclbin_data.is_empty()
    }

    /// Stamp or replace the embedded XCLBIN binary in this `.q4nx` container in-place on disk.
    pub fn stamp_xclbin<P: AsRef<Path>>(&mut self, new_xclbin: &[u8], target_path: Option<P>) -> Result<(), ContainerError> {
        let out_path = target_path
            .as_ref()
            .map(|p| p.as_ref().to_string_lossy().to_string())
            .unwrap_or_else(|| self.file_path.clone());

        let temp_out = if out_path == self.file_path {
            format!("{}.tmp_stamp", out_path)
        } else {
            out_path.clone()
        };

        {
            let mut source_file = File::open(&self.file_path)?;
            source_file.seek(SeekFrom::Start(self.header.payload_offset))?;

            let mut out_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&temp_out)?;

            let mut header_buf = [0u8; HEADER_FIXED_SIZE];
            header_buf[0..4].copy_from_slice(&Q4NX_MAGIC);
            header_buf[4..8].copy_from_slice(&Q4NX_CURRENT_VERSION.to_le_bytes());
            let arch_bytes = self.header.arch_name.as_bytes();
            let copy_len = arch_bytes.len().min(31);
            header_buf[8..8 + copy_len].copy_from_slice(&arch_bytes[..copy_len]);

            header_buf[40..44].copy_from_slice(&self.header.hyperparams.hidden_dim.to_le_bytes());
            header_buf[44..48].copy_from_slice(&self.header.hyperparams.num_heads.to_le_bytes());
            header_buf[48..52].copy_from_slice(&self.header.hyperparams.num_kv_heads.to_le_bytes());
            header_buf[52..56].copy_from_slice(&self.header.hyperparams.num_layers.to_le_bytes());
            header_buf[56..60].copy_from_slice(&self.header.hyperparams.vocab_size.to_le_bytes());
            header_buf[60..64].copy_from_slice(&self.header.hyperparams.context_length.to_le_bytes());

            let xclbin_offset = HEADER_FIXED_SIZE as u64;
            let xclbin_size = new_xclbin.len() as u64;
            let raw_payload_offset = xclbin_offset + xclbin_size;
            let payload_offset = (raw_payload_offset + 63) & !63;

            header_buf[64..72].copy_from_slice(&xclbin_offset.to_le_bytes());
            header_buf[72..80].copy_from_slice(&xclbin_size.to_le_bytes());
            header_buf[96..104].copy_from_slice(&payload_offset.to_le_bytes());
            header_buf[104..112].copy_from_slice(&self.header.payload_size.to_le_bytes());

            out_file.write_all(&header_buf)?;
            out_file.write_all(new_xclbin)?;

            let padding_needed = (payload_offset - raw_payload_offset) as usize;
            if padding_needed > 0 {
                out_file.write_all(&vec![0u8; padding_needed])?;
            }

            io::copy(&mut source_file, &mut out_file)?;
            out_file.flush()?;
        }

        if out_path == self.file_path {
            std::fs::rename(&temp_out, &out_path)?;
        }

        self.xclbin_data = new_xclbin.to_vec();
        self.header.xclbin_size = new_xclbin.len() as u64;
        self.file_path = out_path;
        Ok(())
    }

    /// Build and serialize a unified `.q4nx` container to disk with embedded XCLBIN by default.
    pub fn create_container<P: AsRef<Path>>(
        output_path: P,
        arch_name: &str,
        hyperparams: ModelHyperparameters,
        xclbin_data: Option<&[u8]>,
        payload_data: &[u8],
    ) -> Result<(), ContainerError> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output_path)?;

        let mut header_buf = [0u8; HEADER_FIXED_SIZE];

        // Magic
        header_buf[0..4].copy_from_slice(&Q4NX_MAGIC);
        // Version
        header_buf[4..8].copy_from_slice(&Q4NX_CURRENT_VERSION.to_le_bytes());

        // Architecture Name
        let arch_bytes = arch_name.as_bytes();
        let copy_len = arch_bytes.len().min(31);
        header_buf[8..8 + copy_len].copy_from_slice(&arch_bytes[..copy_len]);

        // Hyperparameters
        header_buf[40..44].copy_from_slice(&hyperparams.hidden_dim.to_le_bytes());
        header_buf[44..48].copy_from_slice(&hyperparams.num_heads.to_le_bytes());
        header_buf[48..52].copy_from_slice(&hyperparams.num_kv_heads.to_le_bytes());
        header_buf[52..56].copy_from_slice(&hyperparams.num_layers.to_le_bytes());
        header_buf[56..60].copy_from_slice(&hyperparams.vocab_size.to_le_bytes());
        header_buf[60..64].copy_from_slice(&hyperparams.context_length.to_le_bytes());

        let xclbin_bytes = xclbin_data.unwrap_or(&[]);
        let xclbin_size = xclbin_bytes.len() as u64;
        let xclbin_offset = HEADER_FIXED_SIZE as u64;

        // Calculate 64-byte aligned payload offset after XCLBIN
        let raw_payload_offset = xclbin_offset + xclbin_size;
        let payload_offset = (raw_payload_offset + 63) & !63; // 64-byte alignment
        let payload_size = payload_data.len() as u64;

        header_buf[64..72].copy_from_slice(&xclbin_offset.to_le_bytes());
        header_buf[72..80].copy_from_slice(&xclbin_size.to_le_bytes());
        header_buf[80..88].copy_from_slice(&0u64.to_le_bytes()); // tensor table offset
        header_buf[88..96].copy_from_slice(&0u64.to_le_bytes()); // tensor count
        header_buf[96..104].copy_from_slice(&payload_offset.to_le_bytes());
        header_buf[104..112].copy_from_slice(&payload_size.to_le_bytes());

        // Write header
        file.write_all(&header_buf)?;

        // Write XCLBIN if present
        if xclbin_size > 0 {
            file.write_all(xclbin_bytes)?;
        }

        // Pad to 64-byte alignment
        let current_pos = xclbin_offset + xclbin_size;
        if current_pos < payload_offset {
            let padding = vec![0u8; (payload_offset - current_pos) as usize];
            file.write_all(&padding)?;
        }

        // Write tensor payload
        file.write_all(payload_data)?;
        file.flush()?;

        Ok(())
    }

    /// Open and parse a FastFlowLM SafeTensors-format `.q4nx` container.
    fn open_safetensors_q4nx(
        file: &mut File,
        header_len: u64,
        path_str: &str,
    ) -> Result<Self, ContainerError> {
        let p = Path::new(path_str);

        // Try reading config.json if present in the same directory
        let config_path = p.parent().map(|dir| dir.join("config.json"));
        let mut arch_name = "llama".to_string();
        let mut hp = ModelHyperparameters::default();

        if let Some(cfg_path) = config_path.filter(|cp| cp.is_file()) {
            if let Ok(cfg_text) = std::fs::read_to_string(&cfg_path) {
                if let Some(model_type) = extract_json_str(&cfg_text, "model_type") {
                    arch_name = model_type;
                }
                hp.hidden_dim = extract_json_u32(&cfg_text, "hidden_size").unwrap_or(4096);
                hp.num_heads = extract_json_u32(&cfg_text, "num_attention_heads").unwrap_or(32);
                hp.num_kv_heads = extract_json_u32(&cfg_text, "num_key_value_heads").unwrap_or(hp.num_heads);
                hp.num_layers = extract_json_u32(&cfg_text, "num_hidden_layers").unwrap_or(32);
                hp.vocab_size = extract_json_u32(&cfg_text, "vocab_size").unwrap_or(128000);
                hp.context_length = extract_json_u32(&cfg_text, "max_position_embeddings").unwrap_or(8192);
            }
        } else {
            // Read first chunk of header JSON to inspect keys
            file.seek(SeekFrom::Start(8))?;
            let mut buf = vec![0u8; (header_len as usize).min(65536)];
            let _ = file.read(&mut buf);
            let header_str = String::from_utf8_lossy(&buf);
            let filename = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            arch_name = if filename.contains("qwen") || header_str.contains("qwen") {
                "qwen2".to_string()
            } else if filename.contains("gemma") || header_str.contains("gemma") {
                "gemma2".to_string()
            } else {
                "llama".to_string()
            };
            hp.hidden_dim = 4096;
            hp.num_heads = 32;
            hp.num_kv_heads = 8;
            hp.num_layers = 32;
            hp.vocab_size = 128000;
            hp.context_length = 8192;
        }

        let total_file_len = file.metadata()?.len();
        let payload_offset = 8 + header_len;
        let payload_size = total_file_len.saturating_sub(payload_offset);

        // Check if an external layer.xclbin exists in the same directory
        let mut xclbin_data = Vec::new();
        if let Some(parent) = p.parent() {
            let companion_xclbin = parent.join("layer.xclbin");
            if companion_xclbin.is_file() {
                if let Ok(bytes) = std::fs::read(&companion_xclbin) {
                    xclbin_data = bytes;
                }
            }
        }
        let xclbin_size = xclbin_data.len() as u64;

        // Read sample of payload data for inspection (up to 4KB or payload_size)
        file.seek(SeekFrom::Start(payload_offset))?;
        let sample_len = (payload_size as usize).min(4096);
        let mut payload_data = vec![0u8; sample_len];
        file.read_exact(&mut payload_data)?;

        Ok(Self {
            header: Q4nxHeader {
                version: 0,
                arch_name,
                hyperparams: hp,
                xclbin_offset: 0,
                xclbin_size,
                tensor_table_offset: 8,
                tensor_table_count: 0,
                payload_offset,
                payload_size,
            },
            file_path: path_str.to_string(),
            xclbin_data,
            payload_data,
        })
    }
}

fn extract_json_u32(text: &str, key: &str) -> Option<u32> {
    let key_pattern = format!("\"{}\"", key);
    let idx = text.find(&key_pattern)?;
    let after_key = &text[idx + key_pattern.len()..];
    let colon_idx = after_key.find(':')?;
    let val_str = after_key[colon_idx + 1..].trim_start();
    let end_idx = val_str
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(val_str.len());
    val_str[..end_idx].parse::<u32>().ok()
}

fn extract_json_str(text: &str, key: &str) -> Option<String> {
    let key_pattern = format!("\"{}\"", key);
    let idx = text.find(&key_pattern)?;
    let after_key = &text[idx + key_pattern.len()..];
    let colon_idx = after_key.find(':')?;
    let val_str = after_key[colon_idx + 1..].trim_start();
    let first_quote = val_str.find('"')?;
    let rest = &val_str[first_quote + 1..];
    let end_quote = rest.find('"')?;
    Some(rest[..end_quote].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4nx_container_roundtrip_with_embedded_xclbin() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_model_with_xclbin.q4nx");

        let arch = "llama";
        let hyperparams = ModelHyperparameters {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 32,
            vocab_size: 128000,
            context_length: 8192,
        };

        let dummy_xclbin = b"\x7FELF_XCLBIN_MOCK_HARDWARE_GRAPH_DATA";
        let dummy_weights = vec![0xABu8; 4096]; // 4KB of dummy weights

        // 1. Create container
        Q4nxModel::create_container(
            &test_file,
            arch,
            hyperparams,
            Some(dummy_xclbin),
            &dummy_weights,
        )
        .expect("Failed to create container");

        // 2. Open container
        let model = Q4nxModel::open(&test_file).expect("Failed to open container");

        assert_eq!(model.header.arch_name, "llama");
        assert_eq!(model.header.hyperparams.hidden_dim, 4096);
        assert_eq!(model.header.hyperparams.num_layers, 32);
        assert_eq!(model.header.payload_offset % 64, 0); // Strict 64-byte alignment
        assert!(model.has_embedded_xclbin());
        assert_eq!(model.xclbin_bytes(), Some(&dummy_xclbin[..]));
        assert_eq!(model.tensor_payload(), &dummy_weights[..]);

        // Cleanup
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_q4nx_bare_stamping_with_xclbin() {
        let temp_dir = std::env::temp_dir();
        let bare_file = temp_dir.join("test_bare_model.q4nx");
        let stamped_file = temp_dir.join("test_stamped_model.q4nx");

        let hyperparams = ModelHyperparameters::default();
        let dummy_weights = vec![0x42u8; 512];

        // Create bare container without xclbin
        Q4nxModel::create_container(
            &bare_file,
            "qwen2",
            hyperparams,
            None,
            &dummy_weights,
        )
        .expect("Failed to create bare container");

        let mut model = Q4nxModel::open(&bare_file).expect("Failed to open bare container");
        assert!(!model.has_embedded_xclbin());

        // In-place stamp with XCLBIN
        let resolved_xclbin = b"RESOLVED_AIE2P_HARDWARE_GRAPH";
        model
            .stamp_xclbin(resolved_xclbin, Some(&stamped_file))
            .expect("Failed to stamp XCLBIN");

        // Verify stamped file
        let stamped_model = Q4nxModel::open(&stamped_file).expect("Failed to open stamped container");
        assert!(stamped_model.has_embedded_xclbin());
        assert_eq!(stamped_model.xclbin_bytes(), Some(&resolved_xclbin[..]));
        assert_eq!(stamped_model.tensor_payload(), &dummy_weights[..]);

        // Cleanup
        let _ = std::fs::remove_file(bare_file);
        let _ = std::fs::remove_file(stamped_file);
    }
}
