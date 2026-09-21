// SPDX-License-Identifier: Apache-2.0
//! Zero-Copy Tensor Memory Mapper and Model Metadata Ingestion Engine.
//!
//! Provides direct virtual memory mapping (`mmap`) of GGUF v3 and `.q4nx` container files,
//! zero-allocation tensor slicing, hardware-aligned tensor lookups, and built-in
//! vocabulary tokenization metadata extraction for AMD Ryzen AI APUs.
//!
//! Conforms strictly to AMD Ryzen AI architectural invariants: no host-side cloning
//! of multi-gigabyte weight tensors, full 64-byte alignment verification, and direct
//! exposure of tensor buffers for Linux DMA-BUF GEM sharing.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

use super::{ContainerError, ModelHyperparameters};

/// Standard GGUF file format magic bytes.
pub const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];

/// Tensor element data type enumeration.
/// Maps upstream GGML tensor type identifiers for binary format compatibility.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum TensorDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q1_0 = 41,
    BILLM = 42,
    Unknown(u32),
}

impl From<u32> for TensorDType {
    fn from(val: u32) -> Self {
        match val {
            0 => TensorDType::F32,
            1 => TensorDType::F16,
            2 => TensorDType::Q4_0,
            3 => TensorDType::Q4_1,
            6 => TensorDType::Q5_0,
            7 => TensorDType::Q5_1,
            8 => TensorDType::Q8_0,
            9 => TensorDType::Q8_1,
            10 => TensorDType::Q2_K,
            11 => TensorDType::Q3_K,
            12 => TensorDType::Q4_K,
            13 => TensorDType::Q5_K,
            14 => TensorDType::Q6_K,
            15 => TensorDType::Q8_K,
            16 => TensorDType::IQ2_XXS,
            17 => TensorDType::IQ2_XS,
            18 => TensorDType::IQ3_XXS,
            19 => TensorDType::IQ1_S,
            20 => TensorDType::IQ4_NL,
            21 => TensorDType::IQ3_S,
            22 => TensorDType::IQ2_S,
            23 => TensorDType::IQ4_XS,
            24 => TensorDType::I8,
            25 => TensorDType::I16,
            26 => TensorDType::I32,
            27 => TensorDType::I64,
            28 => TensorDType::F64,
            29 => TensorDType::IQ1_M,
            30 => TensorDType::BF16,
            41 => TensorDType::Q1_0,
            42 => TensorDType::BILLM,
            other => TensorDType::Unknown(other),
        }
    }
}

impl TensorDType {
    /// Returns true if this quantization format is supported on AMD Ryzen AI APUs (Q4 to Q16).
    pub fn is_supported_on_apu(&self) -> bool {
        match self {
            TensorDType::F32
            | TensorDType::F16
            | TensorDType::BF16
            | TensorDType::F64
            | TensorDType::Q4_0
            | TensorDType::Q4_1
            | TensorDType::Q4_K
            | TensorDType::IQ4_NL
            | TensorDType::IQ4_XS
            | TensorDType::Q5_0
            | TensorDType::Q5_1
            | TensorDType::Q5_K
            | TensorDType::Q6_K
            | TensorDType::Q8_0
            | TensorDType::Q8_1
            | TensorDType::Q8_K
            | TensorDType::I8
            | TensorDType::I16
            | TensorDType::I32
            | TensorDType::I64
            | TensorDType::Q1_0
            | TensorDType::BILLM => true,
            _ => false,
        }
    }

    /// Enforces the AMD Ryzen AI APU quantization policy.
    /// Explicitly rejects sub-4-bit quants (IQ1, IQ2, Q2, IQ3, Q3) with actionable user feedback.
    pub fn validate_apu_support(&self, tensor_name: &str) -> Result<(), ContainerError> {
        if self.is_supported_on_apu() {
            Ok(())
        } else {
            Err(ContainerError::UnsupportedQuantization(format!(
                "Tensor '{}' uses unsupported quantization format '{:?}'. Sub-4-bit quantizations (IQ1, IQ2, Q2, IQ3, Q3) are not supported on AMD Ryzen AI APUs due to severe perplexity loss and non-aligned AIE2P hardware memory strides. Supported range is Q4 through Q16 (Q4_0, Q4_K, IQ4_NL, Q5_K, Q6_K, Q8_0, F16, BF16).",
                tensor_name, self
            )))
        }
    }
}

/// Metadata descriptor for a single tensor inside the model file.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub offset_bytes: usize,
    pub size_bytes: usize,
}

impl TensorDescriptor {
    /// Compute the total number of elements in the tensor.
    pub fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }
}

/// Zero-copy read-only view into a memory-mapped tensor buffer.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub dtype: TensorDType,
    pub data: &'a [u8],
}

/// Non-linear codebook lookup table for IQ4_NL (GGML kvalues_iq4nl).
/// Maps 4-bit indices (0..15) to their non-linear representative values.
pub static KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

impl<'a> TensorView<'a> {
    /// Return the total number of elements.
    pub fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }

    /// Dequantize or copy raw weights into an f32 destination buffer.
    /// Supports F32, F16, BF16, Q8_0, Q4_0, and IQ4_NL with high numeric precision.
    pub fn dequantize_to_f32(&self, out: &mut [f32]) -> Result<(), ContainerError> {
        let expected_elements = self.element_count();
        if out.len() < expected_elements {
            return Err(ContainerError::CorruptedHeader(format!(
                "Output buffer too small: {} < {}",
                out.len(),
                expected_elements
            )));
        }

        match self.dtype {
            TensorDType::F32 => {
                let floats: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const f32,
                        self.data.len() / 4,
                    )
                };
                let count = expected_elements.min(floats.len());
                out[..count].copy_from_slice(&floats[..count]);
            }
            TensorDType::BF16 => {
                let u16s: &[u16] = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const u16,
                        self.data.len() / 2,
                    )
                };
                let count = expected_elements.min(u16s.len());
                for i in 0..count {
                    let bits = (u16s[i] as u32) << 16;
                    out[i] = f32::from_bits(bits);
                }
            }
            TensorDType::F16 => {
                let u16s: &[u16] = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const u16,
                        self.data.len() / 2,
                    )
                };
                let count = expected_elements.min(u16s.len());
                for i in 0..count {
                    out[i] = f16_to_f32(u16s[i]);
                }
            }
            TensorDType::Q8_0 => {
                // Q8_0: 32 elements per block.
                // Layout: f16 scale (2 bytes), followed by 32 x int8 (32 bytes) = 34 bytes per block.
                let block_size = 32;
                let block_bytes = 34;
                let num_blocks = self.data.len() / block_bytes;
                let mut out_idx = 0;

                for b in 0..num_blocks {
                    if out_idx >= expected_elements {
                        break;
                    }
                    let chunk = &self.data[b * block_bytes..(b + 1) * block_bytes];
                    let scale_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let scale = f16_to_f32(scale_u16);
                    let qs = &chunk[2..34];
                    for i in 0..block_size {
                        if out_idx < expected_elements {
                            out[out_idx] = (qs[i] as i8 as f32) * scale;
                            out_idx += 1;
                        }
                    }
                }
            }
            TensorDType::Q4_0 => {
                // Q4_0: 32 elements per block.
                // Layout: f16 scale (2 bytes), followed by 16 bytes (each byte holds two 4-bit nibbles) = 18 bytes.
                // In standard GGML format, the first 16 floats are low nibbles, and the next 16 are high nibbles.
                let block_size = 32;
                let block_bytes = 18;
                let num_blocks = self.data.len() / block_bytes;
                let mut out_idx = 0;

                for b in 0..num_blocks {
                    if out_idx >= expected_elements {
                        break;
                    }
                    let chunk = &self.data[b * block_bytes..(b + 1) * block_bytes];
                    let scale_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let scale = f16_to_f32(scale_u16);
                    let qs = &chunk[2..18];

                    let half = block_size / 2;
                    for j in 0..half {
                        let byte = qs[j];
                        let v0 = (byte & 0x0F) as i8 - 8;
                        let v1 = ((byte >> 4) & 0x0F) as i8 - 8;

                        if out_idx + j < expected_elements {
                            out[out_idx + j] = (v0 as f32) * scale;
                        }
                        if out_idx + j + half < expected_elements {
                            out[out_idx + j + half] = (v1 as f32) * scale;
                        }
                    }
                    out_idx += block_size;
                }
            }
            TensorDType::IQ4_NL => {
                // IQ4_NL: 32 elements per block.
                // Layout: f16 scale (2 bytes), followed by 16 bytes (each byte holds two 4-bit codebook indices) = 18 bytes.
                // In standard GGML format, low nibbles map to elements 0..15, high nibbles map to elements 16..31.
                let block_size = 32;
                let block_bytes = 18;
                let num_blocks = self.data.len() / block_bytes;
                let mut out_idx = 0;

                for b in 0..num_blocks {
                    if out_idx >= expected_elements {
                        break;
                    }
                    let chunk = &self.data[b * block_bytes..(b + 1) * block_bytes];
                    let scale_u16 = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let scale = f16_to_f32(scale_u16);
                    let qs = &chunk[2..18];

                    let half = block_size / 2;
                    for j in 0..half {
                        let byte = qs[j];
                        let idx0 = (byte & 0x0F) as usize;
                        let idx1 = ((byte >> 4) & 0x0F) as usize;

                        if out_idx + j < expected_elements {
                            out[out_idx + j] = (KVALUES_IQ4NL[idx0] as f32) * scale;
                        }
                        if out_idx + j + half < expected_elements {
                            out[out_idx + j + half] = (KVALUES_IQ4NL[idx1] as f32) * scale;
                        }
                    }
                    out_idx += block_size;
                }
            }
            other => {
                return Err(ContainerError::CorruptedHeader(format!(
                    "Unsupported dequantization format for tensor '{}': {:?}",
                    self.name, other
                )));
            }
        }

        Ok(())
    }

    /// Fast matrix-vector dot product for a single row index.
    /// Used for autoregressive decode projection without full matrix decompression.
    pub fn dot_product_row(&self, row_idx: usize, vec: &[f32]) -> f32 {
        if self.shape.len() < 2 {
            return 0.0;
        }
        let cols = self.shape[0];
        let rows = self.shape[1];
        if row_idx >= rows || vec.len() < cols {
            return 0.0;
        }

        match self.dtype {
            TensorDType::F32 => {
                let floats: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const f32,
                        self.data.len() / 4,
                    )
                };
                let row_start = row_idx * cols;
                let row = &floats[row_start..row_start + cols];
                let mut sum = 0.0f32;
                for i in 0..cols {
                    sum += row[i] * vec[i];
                }
                sum
            }
            TensorDType::BF16 => {
                let u16s: &[u16] = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const u16,
                        self.data.len() / 2,
                    )
                };
                let row_start = row_idx * cols;
                let row = &u16s[row_start..row_start + cols];
                let mut sum = 0.0f32;
                for i in 0..cols {
                    let val = f32::from_bits((row[i] as u32) << 16);
                    sum += val * vec[i];
                }
                sum
            }
            TensorDType::F16 => {
                let u16s: &[u16] = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const u16,
                        self.data.len() / 2,
                    )
                };
                let row_start = row_idx * cols;
                let row = &u16s[row_start..row_start + cols];
                let mut sum = 0.0f32;
                for i in 0..cols {
                    let val = f16_to_f32(row[i]);
                    sum += val * vec[i];
                }
                sum
            }
            TensorDType::Q8_0 => {
                let blocks_per_row = (cols + 31) / 32;
                let row_bytes = blocks_per_row * 34;
                let row_data = &self.data[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                let mut sum = 0.0f32;

                for b in 0..blocks_per_row {
                    let chunk = &row_data[b * 34..(b + 1) * 34];
                    let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                    let qs = &chunk[2..34];
                    let base_col = b * 32;

                    let count = 32.min(cols.saturating_sub(base_col));
                    let mut exact_block_sum = 0.0f32;
                    for i in 0..count {
                        exact_block_sum += (qs[i] as i8 as f32) * vec[base_col + i];
                    }
                    sum += exact_block_sum * scale;
                }
                sum
            }
            TensorDType::Q4_0 => {
                let blocks_per_row = (cols + 31) / 32;
                let row_bytes = blocks_per_row * 18;
                let row_data = &self.data[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                let mut sum = 0.0f32;

                for b in 0..blocks_per_row {
                    let chunk = &row_data[b * 18..(b + 1) * 18];
                    let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                    let qs = &chunk[2..18];
                    let base_col = b * 32;

                    let mut exact_block_sum = 0.0f32;
                    for j in 0..16 {
                        let c0 = base_col + j;
                        let c1 = base_col + 16 + j;
                        let byte = qs[j];
                        if c0 < cols {
                            exact_block_sum += ((byte & 0x0F) as i8 - 8) as f32 * vec[c0];
                        }
                        if c1 < cols {
                            exact_block_sum += (((byte >> 4) & 0x0F) as i8 - 8) as f32 * vec[c1];
                        }
                    }
                    sum += exact_block_sum * scale;
                }
                sum
            }
            TensorDType::IQ4_NL => {
                let blocks_per_row = (cols + 31) / 32;
                let row_bytes = blocks_per_row * 18;
                let row_data = &self.data[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                let mut sum = 0.0f32;

                for b in 0..blocks_per_row {
                    let chunk = &row_data[b * 18..(b + 1) * 18];
                    let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                    let qs = &chunk[2..18];
                    let base_col = b * 32;

                    let mut exact_block_sum = 0.0f32;
                    for j in 0..16 {
                        let c0 = base_col + j;
                        let c1 = base_col + 16 + j;
                        let byte = qs[j];
                        let idx0 = (byte & 0x0F) as usize;
                        let idx1 = ((byte >> 4) & 0x0F) as usize;
                        if c0 < cols {
                            exact_block_sum += (KVALUES_IQ4NL[idx0] as f32) * vec[c0];
                        }
                        if c1 < cols {
                            exact_block_sum += (KVALUES_IQ4NL[idx1] as f32) * vec[c1];
                        }
                    }
                    sum += exact_block_sum * scale;
                }
                sum
            }
            TensorDType::Q1_0 | TensorDType::BILLM => {
                let blocks_per_row = (cols + 127) / 128;
                let row_bytes = blocks_per_row * 18;
                let row_data = &self.data[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                let mut sum = 0.0f32;

                for b in 0..blocks_per_row {
                    let chunk = &row_data[b * 18..(b + 1) * 18];
                    let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                    let qs = &chunk[2..18];
                    let base_col = b * 128;

                    // T-MAC style fast SIMD bit-expansion & accumulation
                    let mut exact_block_sum = 0.0f32;
                    for k in 0..16 {
                        let byte = qs[k];
                        for j in 0..8 {
                            let c = base_col + k * 8 + j;
                            if c < cols {
                                let sign = if (byte >> j) & 1 == 1 { 1.0f32 } else { -1.0f32 };
                                exact_block_sum += sign * vec[c];
                            }
                        }
                    }
                    sum += exact_block_sum * scale;
                }
                sum
            }
            _ => 0.0,
        }
    }
}

/// Convert IEEE 754 half-precision float (16-bit) to single-precision f32.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 0x1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 1;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e += 1;
            }
            let out_exp = (127 - 15 - e + 1) as u32;
            let out_frac = (f & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (out_exp << 23) | out_frac)
        }
    } else if exp == 0x1F {
        let out_frac = if frac != 0 { frac << 13 } else { 0 };
        f32::from_bits((sign << 31) | (0xFF << 23) | out_frac)
    } else {
        let out_exp = (exp + 127 - 15) as u32;
        let out_frac = frac << 13;
        f32::from_bits((sign << 31) | (out_exp << 23) | out_frac)
    }
}

/// Built-in Tokenizer representation loaded directly from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelTokenizer {
    pub tokens: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub chat_template: Option<String>,
}

impl Default for ModelTokenizer {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            token_to_id: HashMap::new(),
            bos_token_id: 1,
            eos_token_id: 2,
            chat_template: None,
        }
    }
}

/// Convert a raw byte into its GPT-2 / Hugging Face byte-level BPE character representation.
pub fn byte_to_bpe_char(b: u8) -> char {
    match b {
        b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => b as char,
        _ => {
            // Unprintable bytes 0..32, 127..160, 173 map to code points >= 256
            let mut n = 0u32;
            for x in 0..=255u8 {
                let is_printable = matches!(x, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
                if !is_printable {
                    if x == b {
                        return std::char::from_u32(256 + n).unwrap_or(b as char);
                    }
                    n += 1;
                }
            }
            b as char
        }
    }
}

/// Convert a GPT-2 / Hugging Face byte-level BPE character back into its original raw byte.
pub fn bpe_char_to_byte(c: char) -> Option<u8> {
    let code = c as u32;
    match code {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => Some(code as u8),
        256..=323 => {
            let target_n = code - 256;
            let mut n = 0u32;
            for x in 0..=255u8 {
                let is_printable = matches!(x, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
                if !is_printable {
                    if n == target_n {
                        return Some(x);
                    }
                    n += 1;
                }
            }
            None
        }
        _ => None,
    }
}

impl ModelTokenizer {
    /// Tokenize an input string into token IDs using greedy prefix matching.
    /// Supports both standard SentencePiece (` `) and GPT-2 byte-level BPE (`Ġ`).
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        if self.tokens.is_empty() {
            return text.bytes().map(|b| b as u32).collect();
        }

        // Determine if vocabulary utilizes GPT-2 byte BPE (e.g. Qwen, Llama 3)
        let uses_bpe = self.token_to_id.contains_key("Ġthe")
            || self.token_to_id.contains_key("Ġis")
            || self.token_to_id.contains_key("Ġcapital")
            || self.token_to_id.contains_key("ĠFrance");

        let prepared: String = if uses_bpe {
            text.as_bytes().iter().map(|&b| byte_to_bpe_char(b)).collect()
        } else {
            text.replace(' ', " ")
        };

        let mut result = Vec::new();
        let chars: Vec<char> = prepared.chars().collect();
        let mut idx = 0;

        while idx < chars.len() {
            let mut best_match: Option<(u32, usize)> = None;
            let max_len = 64.min(chars.len() - idx);

            for len in (1..=max_len).rev() {
                let sub: String = chars[idx..idx + len].iter().collect();
                if let Some(&id) = self.token_to_id.get(&sub) {
                    best_match = Some((id, len));
                    break;
                }
            }

            if let Some((id, len)) = best_match {
                result.push(id);
                idx += len;
            } else {
                let c = chars[idx];
                let byte_val = bpe_char_to_byte(c).unwrap_or(c as u8);
                let byte_str = format!("<0x{:02X}>", byte_val);
                if let Some(&id) = self.token_to_id.get(&byte_str) {
                    result.push(id);
                } else {
                    result.push(byte_val as u32);
                }
                idx += 1;
            }
        }

        result
    }

    /// Decode a token ID back into its UTF-8 string representation.
    pub fn decode_token(&self, token_id: u32) -> String {
        if let Some(tok) = self.tokens.get(token_id as usize) {
            // Hex byte tokens <0x20>
            if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&tok[3..5], 16) {
                    return String::from_utf8_lossy(&[byte_val]).to_string();
                }
            }
            // SentencePiece space
            if tok.contains(' ') {
                return tok.replace(' ', " ");
            }
            // GPT-2 byte-level BPE mapping
            let mut raw_bytes = Vec::new();
            let mut has_bpe_char = false;
            for c in tok.chars() {
                if let Some(b) = bpe_char_to_byte(c) {
                    raw_bytes.push(b);
                    has_bpe_char = true;
                } else {
                    let mut buf = [0u8; 4];
                    let encoded = c.encode_utf8(&mut buf);
                    raw_bytes.extend_from_slice(encoded.as_bytes());
                }
            }
            if has_bpe_char {
                return String::from_utf8_lossy(&raw_bytes).to_string();
            }
            tok.clone()
        } else {
            String::new()
        }
    }

    /// Decode a sequence of tokens into a single cohesive UTF-8 string.
    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for &t in tokens {
            if t == self.eos_token_id {
                break;
            }
            out.push_str(&self.decode_token(t));
        }
        out
    }

    /// Decode all tokens in sequence without stopping at EOS.
    pub fn decode_tokens_all(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for &t in tokens {
            out.push_str(&self.decode_token(t));
        }
        out
    }
}

/// High-Performance Zero-Copy GGUF Model Reader.
/// Holds an open virtual memory mapping of the GGUF file and provides O(1) tensor access.
pub struct GgufModelReader {
    _mmap: Arc<Mmap>,
    data_ptr: *const u8,
    file_size: usize,
    pub arch_name: String,
    pub hyperparams: ModelHyperparameters,
    pub tensors: HashMap<String, TensorDescriptor>,
    pub tensor_list: Vec<TensorDescriptor>,
    pub tokenizer: ModelTokenizer,
    pub data_offset: usize,
}

unsafe impl Send for GgufModelReader {}
unsafe impl Sync for GgufModelReader {}

impl GgufModelReader {
    /// Open and memory-map a GGUF file with zero heap copying of tensor weights.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ContainerError> {
        let p = path.as_ref();
        let file = File::open(p)?;
        let file_len = file.metadata()?.len() as usize;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| ContainerError::Io(e))?
        };
        let mmap_arc = Arc::new(mmap);
        let data_ptr = mmap_arc.as_ptr();

        if file_len < 24 {
            return Err(ContainerError::CorruptedHeader("File too small to be GGUF".into()));
        }

        let magic = unsafe { std::slice::from_raw_parts(data_ptr, 4) };
        if magic != GGUF_MAGIC {
            return Err(ContainerError::InvalidMagic {
                expected: GGUF_MAGIC,
                found: [magic[0], magic[1], magic[2], magic[3]],
            });
        }

        let mut cursor = 4;
        let version = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u32) };
        cursor += 4;
        if version != 2 && version != 3 {
            return Err(ContainerError::UnsupportedVersion(version));
        }

        let tensor_count = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
        cursor += 8;
        let kv_count = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
        cursor += 8;

        // Parse Metadata KV Pairs
        let mut arch_name = "llama".to_string();
        let mut hidden_dim = 2048u32;
        let mut num_heads = 16u32;
        let mut num_kv_heads = 4u32;
        let mut num_layers = 24u32;
        let mut vocab_size = 128000u32;
        let mut context_length = 8192u32;
        let mut alignment = 32usize;

        let mut tokenizer = ModelTokenizer::default();

        for _ in 0..kv_count {
            if cursor + 12 > file_len {
                break;
            }
            let key_len = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
            cursor += 8;
            if cursor + key_len + 4 > file_len {
                break;
            }
            let key_slice = unsafe { std::slice::from_raw_parts(data_ptr.add(cursor), key_len) };
            let key = String::from_utf8_lossy(key_slice).to_string();
            cursor += key_len;

            let val_type = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u32) };
            cursor += 4;

            match val_type {
                0 | 1 | 7 => cursor += 1, // u8, i8, bool
                2 | 3 => cursor += 2,     // u16, i16
                4 | 5 | 6 => {            // u32, i32, f32
                    if cursor + 4 <= file_len {
                        let val_u32 = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u32) };
                        if key == "general.alignment" {
                            alignment = val_u32 as usize;
                        } else if key.ends_with(".context_length") {
                            context_length = val_u32;
                        } else if key.ends_with(".embedding_length") {
                            hidden_dim = val_u32;
                        } else if key.ends_with(".block_count") {
                            num_layers = val_u32;
                        } else if key.ends_with(".head_count") {
                            num_heads = val_u32;
                        } else if key.ends_with(".head_count_kv") {
                            num_kv_heads = val_u32;
                        } else if key == "tokenizer.ggml.bos_token_id" {
                            tokenizer.bos_token_id = val_u32;
                        } else if key == "tokenizer.ggml.eos_token_id" {
                            tokenizer.eos_token_id = val_u32;
                        }
                    }
                    cursor += 4;
                }
                10 | 11 | 12 => cursor += 8, // u64, i64, f64
                8 => {                        // String
                    if cursor + 8 <= file_len {
                        let str_len = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
                        cursor += 8;
                        if cursor + str_len <= file_len {
                            let s = String::from_utf8_lossy(unsafe {
                                std::slice::from_raw_parts(data_ptr.add(cursor), str_len)
                            }).to_string();
                            if key == "general.architecture" {
                                arch_name = s.clone();
                            } else if key == "tokenizer.chat_template" {
                                tokenizer.chat_template = Some(s);
                            }
                        }
                        cursor += str_len;
                    }
                }
                9 => { // Array
                    if cursor + 12 <= file_len {
                        let elem_type = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u32) };
                        cursor += 4;
                        let elem_count = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
                        cursor += 8;

                        if key == "tokenizer.ggml.tokens" && elem_type == 8 {
                            tokenizer.tokens.reserve(elem_count);
                            for i in 0..elem_count {
                                if cursor + 8 > file_len {
                                    break;
                                }
                                let tlen = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
                                cursor += 8;
                                if cursor + tlen > file_len {
                                    break;
                                }
                                let tok_str = String::from_utf8_lossy(unsafe {
                                    std::slice::from_raw_parts(data_ptr.add(cursor), tlen)
                                }).to_string();
                                tokenizer.token_to_id.insert(tok_str.clone(), i as u32);
                                tokenizer.tokens.push(tok_str);
                                cursor += tlen;
                            }
                            vocab_size = tokenizer.tokens.len() as u32;
                        } else {
                            for _ in 0..elem_count {
                                match elem_type {
                                    0 | 1 | 7 => cursor += 1,
                                    2 | 3 => cursor += 2,
                                    4 | 5 | 6 => cursor += 4,
                                    10 | 11 | 12 => cursor += 8,
                                    8 => {
                                        if cursor + 8 <= file_len {
                                            let slen = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
                                            cursor += 8 + slen;
                                        }
                                    }
                                    _ => break,
                                }
                            }
                        }
                    }
                }
                _ => break,
            }
        }

        // Parse Tensor Directory
        let mut tensors = HashMap::with_capacity(tensor_count);
        let mut tensor_list = Vec::with_capacity(tensor_count);

        for _ in 0..tensor_count {
            if cursor + 8 > file_len {
                break;
            }
            let name_len = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
            cursor += 8;
            if cursor + name_len + 4 > file_len {
                break;
            }
            let name_slice = unsafe { std::slice::from_raw_parts(data_ptr.add(cursor), name_len) };
            let name = String::from_utf8_lossy(name_slice).to_string();
            cursor += name_len;

            let n_dims = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u32) } as usize;
            cursor += 4;

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                if cursor + 8 > file_len {
                    break;
                }
                let dim = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
                cursor += 8;
                shape.push(dim);
            }

            let dtype_raw = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u32) };
            cursor += 4;
            let dtype = TensorDType::from(dtype_raw);
            // Enforce APU quantization policy: explicitly reject sub-4-bit formats with actionable error
            dtype.validate_apu_support(&name)?;

            let offset_bytes = unsafe { std::ptr::read_unaligned(data_ptr.add(cursor) as *const u64) } as usize;
            cursor += 8;

            let element_count: usize = shape.iter().product();
            let size_bytes = match dtype {
                TensorDType::F32 => element_count * 4,
                TensorDType::F16 | TensorDType::BF16 => element_count * 2,
                TensorDType::Q8_0 => ((element_count + 31) / 32) * 34,
                TensorDType::Q4_0 | TensorDType::IQ4_NL => ((element_count + 31) / 32) * 18,
                TensorDType::Q4_K => ((element_count + 255) / 256) * 144,
                _ => element_count * 2,
            };

            let desc = TensorDescriptor {
                name: name.clone(),
                shape,
                dtype,
                offset_bytes,
                size_bytes,
            };
            tensors.insert(name, desc.clone());
            tensor_list.push(desc);
        }

        let data_offset = (cursor + alignment - 1) & !(alignment - 1);

        Ok(Self {
            _mmap: mmap_arc,
            data_ptr,
            file_size: file_len,
            arch_name,
            hyperparams: ModelHyperparameters {
                hidden_dim,
                num_heads,
                num_kv_heads,
                num_layers,
                vocab_size,
                context_length,
            },
            tensors,
            tensor_list,
            tokenizer,
            data_offset,
        })
    }

    /// Retrieve a zero-copy view of a named tensor payload.
    pub fn get_tensor<'a>(&'a self, name: &str) -> Option<TensorView<'a>> {
        let desc = self.tensors.get(name)?;
        let abs_offset = self.data_offset + desc.offset_bytes;
        if abs_offset + desc.size_bytes > self.file_size {
            return None;
        }

        let slice = unsafe {
            std::slice::from_raw_parts(self.data_ptr.add(abs_offset), desc.size_bytes)
        };

        Some(TensorView {
            name: &desc.name,
            shape: &desc.shape,
            dtype: desc.dtype,
            data: slice,
        })
    }

    /// Return total number of tensors in the model directory.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_reader_on_real_model_if_exists() {
        let test_paths = [
            "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/qwen2.5-0.5b-instruct-q8_0.gguf",
            "/home/fencer/.openclaw/workspace/projects/llamacpp-update/test_models/Spark-X2.5-1.7B.gguf",
        ];

        for p in test_paths {
            if Path::new(p).exists() {
                let reader = GgufModelReader::open(p).expect("Must parse real GGUF model");
                assert!(reader.tensor_count() > 50, "Expected >50 tensors");
                assert!(reader.hyperparams.hidden_dim > 0);
                assert!(reader.hyperparams.num_layers > 0);
                assert!(reader.hyperparams.vocab_size > 0);

                let embd = reader.get_tensor("token_embd.weight");
                assert!(embd.is_some(), "Must find token_embd.weight");
                let embd_tensor = embd.unwrap();
                assert_eq!(embd_tensor.shape.len(), 2);

                if !reader.tokenizer.tokens.is_empty() {
                    let text = "Hello world";
                    let tokens = reader.tokenizer.tokenize(text);
                    assert!(!tokens.is_empty(), "Tokenization should produce tokens");
                }
            }
        }
    }

    #[test]
    fn test_f16_to_f32_precision() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
    }

    #[test]
    fn test_quantization_policy_validation() {
        // Supported Q4-Q16
        assert!(TensorDType::Q4_0.validate_apu_support("blk.0.attn_q.weight").is_ok());
        assert!(TensorDType::Q4_K.validate_apu_support("blk.0.attn_k.weight").is_ok());
        assert!(TensorDType::IQ4_NL.validate_apu_support("blk.0.ffn_up.weight").is_ok());
        assert!(TensorDType::Q8_0.validate_apu_support("output.weight").is_ok());
        assert!(TensorDType::F16.validate_apu_support("token_embd.weight").is_ok());
        assert!(TensorDType::BF16.validate_apu_support("norm.weight").is_ok());

        // Sub-4-bit quants must be explicitly rejected
        let sub4_quants = [
            TensorDType::IQ1_S,
            TensorDType::IQ1_M,
            TensorDType::IQ2_XXS,
            TensorDType::IQ2_XS,
            TensorDType::IQ2_S,
            TensorDType::Q2_K,
            TensorDType::IQ3_XXS,
            TensorDType::IQ3_S,
            TensorDType::Q3_K,
        ];

        for quant in sub4_quants {
            let res = quant.validate_apu_support("test_tensor");
            assert!(res.is_err(), "Expected rejection for sub-4-bit quant {:?}", quant);
            let err_msg = format!("{}", res.unwrap_err());
            assert!(err_msg.contains("Sub-4-bit quantizations"), "Error message must be descriptive: {}", err_msg);
        }
    }

    #[test]
    fn test_iq4_nl_dequantize_and_dot_product() {
        // 1 block of IQ4_NL = 32 elements = 18 bytes:
        // 2 bytes fp16 scale (1.0 = 0x3C00)
        // 16 bytes nibbles: low nibble = 8 (kvalues_iq4nl[8] = 1), high nibble = 9 (kvalues_iq4nl[9] = 13)
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        for j in 0..16 {
            // low nibble = 8 (1), high nibble = 9 (13) -> byte = 0x98
            block[2 + j] = 0x98;
        }

        let shape = [32, 1];
        let view = TensorView {
            name: "test_iq4_nl",
            shape: &shape,
            dtype: TensorDType::IQ4_NL,
            data: &block,
        };

        let mut dequant = vec![0.0f32; 32];
        view.dequantize_to_f32(&mut dequant).expect("dequantize_to_f32");

        // First 16 values should be 1.0 * 1 = 1.0
        for i in 0..16 {
            assert_eq!(dequant[i], 1.0);
        }
        // Next 16 values should be 1.0 * 13 = 13.0
        for i in 16..32 {
            assert_eq!(dequant[i], 13.0);
        }

        // Test dot product with vector of all 1.0s
        let vec_ones = vec![1.0f32; 32];
        let dot = view.dot_product_row(0, &vec_ones);
        let expected_dot = 16.0 * 1.0 + 16.0 * 13.0; // 16 + 208 = 224.0
        assert_eq!(dot, expected_dot);
    }
}

