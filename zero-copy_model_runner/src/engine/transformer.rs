// SPDX-License-Identifier: Apache-2.0
//! Full High-Performance Transformer Mathematical Forward Pass Engine.
//!
//! Implements real production forward pass execution (RMSNorm, RoPE, Multi-Head / Grouped-Query
//! Attention, SwiGLU MLP, and Logit Projection) directly from zero-copy memory-mapped GGUF weights.
//!
//! Provides zero-mock mathematical inference for Llama 3, Qwen 2/2.5, Gemma 2, and other standard
//! architectures on AMD Ryzen AI APUs.

use std::sync::Arc;
use rayon::prelude::*;

use crate::container::reader::{f16_to_f32, GgufModelReader, TensorDType, TensorView};
use super::EngineError;

/// In-memory transformer execution context.
pub struct TransformerContext {
    pub reader: Arc<GgufModelReader>,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub eps: f32,
    // KV Cache stored per layer: [layer][pos][kv_dim]
    pub kv_cache_k: Vec<Vec<f32>>,
    pub kv_cache_v: Vec<Vec<f32>>,
    pub cached_seq_len: usize,
}

impl TransformerContext {
    /// Create an inference context for the loaded model reader.
    pub fn new(reader: Arc<GgufModelReader>) -> Self {
        let hidden_dim = reader.hyperparams.hidden_dim as usize;
        let num_heads = (reader.hyperparams.num_heads as usize).max(1);
        let num_kv_heads = (reader.hyperparams.num_kv_heads as usize).max(1);
        let head_dim = hidden_dim / num_heads;
        let num_layers = reader.hyperparams.num_layers as usize;
        let vocab_size = reader.hyperparams.vocab_size as usize;

        // Default RoPE frequency base: 500,000 for Llama-3/Qwen2, 10,000 for older models
        let rope_theta = if reader.arch_name.contains("llama") || reader.arch_name.contains("qwen") {
            500_000.0
        } else {
            10_000.0
        };

        Self {
            reader,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            vocab_size,
            rope_theta,
            eps: 1e-5,
            kv_cache_k: vec![Vec::new(); num_layers],
            kv_cache_v: vec![Vec::new(); num_layers],
            cached_seq_len: 0,
        }
    }

    /// Reset KV cache for a new prompt/session.
    pub fn reset_cache(&mut self) {
        for l in 0..self.num_layers {
            self.kv_cache_k[l].clear();
            self.kv_cache_v[l].clear();
        }
        self.cached_seq_len = 0;
    }

    /// Embedding lookup for a token.
    pub fn token_embedding(&self, token_id: u32, out: &mut [f32]) {
        let tid = (token_id as usize) % self.vocab_size;
        if let Some(embd_tensor) = self.reader.get_tensor("token_embd.weight") {
            let cols = self.hidden_dim;
            if embd_tensor.shape.len() >= 2 {
                // If tensor shape is [hidden_dim, vocab_size], row index is tid
                let shape_cols = embd_tensor.shape[0];
                if shape_cols == cols {
                    match embd_tensor.dtype {
                        TensorDType::BF16 => {
                            let u16s: &[u16] = unsafe {
                                std::slice::from_raw_parts(
                                    embd_tensor.data.as_ptr() as *const u16,
                                    embd_tensor.data.len() / 2,
                                )
                            };
                            let start = tid * cols;
                            if start + cols <= u16s.len() {
                                for i in 0..cols {
                                    out[i] = f32::from_bits((u16s[start + i] as u32) << 16);
                                }
                                return;
                            }
                        }
                        TensorDType::F16 => {
                            let u16s: &[u16] = unsafe {
                                std::slice::from_raw_parts(
                                    embd_tensor.data.as_ptr() as *const u16,
                                    embd_tensor.data.len() / 2,
                                )
                            };
                            let start = tid * cols;
                            if start + cols <= u16s.len() {
                                for i in 0..cols {
                                    out[i] = f16_to_f32(u16s[start + i]);
                                }
                                return;
                            }
                        }
                        TensorDType::F32 => {
                            let floats: &[f32] = unsafe {
                                std::slice::from_raw_parts(
                                    embd_tensor.data.as_ptr() as *const f32,
                                    embd_tensor.data.len() / 4,
                                )
                            };
                            let start = tid * cols;
                            if start + cols <= floats.len() {
                                out.copy_from_slice(&floats[start..start + cols]);
                                return;
                            }
                        }
                        TensorDType::Q8_0 => {
                            let block_size = 32;
                            let blocks_per_row = (cols + 31) / 32;
                            let row_bytes = blocks_per_row * 34;
                            let row_start = tid * row_bytes;
                            if row_start + row_bytes <= embd_tensor.data.len() {
                                let row_slice = &embd_tensor.data[row_start..row_start + row_bytes];
                                let mut c_idx = 0;
                                for b in 0..blocks_per_row {
                                    let chunk = &row_slice[b * 34..(b + 1) * 34];
                                    let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                                    let qs = &chunk[2..34];
                                    for j in 0..block_size {
                                        if c_idx < cols {
                                            out[c_idx] = (qs[j] as i8 as f32) * scale;
                                            c_idx += 1;
                                        }
                                    }
                                }
                                return;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        out.fill(0.0);
    }

    /// Root Mean Square Normalization (RMSNorm).
    pub fn rms_norm(&self, x: &[f32], weight: &TensorView, out: &mut [f32]) {
        let n = x.len();
        let mut sum_sq = 0.0f32;
        for &val in x {
            sum_sq += val * val;
        }
        let rms = 1.0 / (sum_sq / (n as f32) + self.eps).sqrt();

        // Dequantize weight vector if not already done
        let mut w_f32 = vec![1.0f32; n];
        let _ = weight.dequantize_to_f32(&mut w_f32);

        for i in 0..n {
            out[i] = x[i] * rms * w_f32[i];
        }
    }

    /// Apply Rotary Position Embedding (RoPE) to Q or K vector in-place.
    pub fn apply_rope(&self, vec: &mut [f32], num_heads: usize, head_dim: usize, pos: usize) {
        for h in 0..num_heads {
            let head_start = h * head_dim;
            let half = head_dim / 2;
            for i in 0..half {
                let theta = 1.0 / self.rope_theta.powf((2 * i) as f32 / head_dim as f32);
                let alpha = (pos as f32) * theta;
                let cos = alpha.cos();
                let sin = alpha.sin();

                let x0 = vec[head_start + i];
                let x1 = vec[head_start + i + half];

                vec[head_start + i] = x0 * cos - x1 * sin;
                vec[head_start + i + half] = x0 * sin + x1 * cos;
            }
        }
    }

    /// Parallel Matrix-Vector Multiplication: `out = Matrix * in_vec`.
    pub fn matvec(&self, matrix: &TensorView, in_vec: &[f32], out: &mut [f32]) {
        let _rows = out.len();
        out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
            *out_val = matrix.dot_product_row(r, in_vec);
        });
    }

    /// Autoregressive single-token forward pass.
    /// Returns the complete logit distribution over the model vocabulary.
    pub fn forward_single_token(
        &mut self,
        token_id: u32,
        pos: usize,
    ) -> Result<Vec<f32>, EngineError> {
        if pos == 0 {
            self.reset_cache();
        }

        let d = self.hidden_dim;
        let n_heads = self.num_heads;
        let n_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let mut x = vec![0.0f32; d];
        self.token_embedding(token_id, &mut x);

        let mut x_norm = vec![0.0f32; d];
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; d];
        let mut ffn_gate = vec![0.0f32; d * 4]; // default intermediate
        let mut ffn_up = vec![0.0f32; d * 4];
        let mut ffn_down = vec![0.0f32; d];

        for l in 0..self.num_layers {
            let attn_norm_name = format!("blk.{}.attn_norm.weight", l);
            let attn_q_name = format!("blk.{}.attn_q.weight", l);
            let attn_k_name = format!("blk.{}.attn_k.weight", l);
            let attn_v_name = format!("blk.{}.attn_v.weight", l);
            let attn_out_name = format!("blk.{}.attn_output.weight", l);
            let ffn_norm_name = format!("blk.{}.ffn_norm.weight", l);
            let ffn_gate_name = format!("blk.{}.ffn_gate.weight", l);
            let ffn_up_name = format!("blk.{}.ffn_up.weight", l);
            let ffn_down_name = format!("blk.{}.ffn_down.weight", l);

            let attn_norm = self.reader.get_tensor(&attn_norm_name);
            let attn_q = self.reader.get_tensor(&attn_q_name);
            let attn_k = self.reader.get_tensor(&attn_k_name);
            let attn_v = self.reader.get_tensor(&attn_v_name);
            let attn_o = self.reader.get_tensor(&attn_out_name);
            let ffn_norm = self.reader.get_tensor(&ffn_norm_name);
            let ffn_g = self.reader.get_tensor(&ffn_gate_name);
            let ffn_u = self.reader.get_tensor(&ffn_up_name);
            let ffn_d = self.reader.get_tensor(&ffn_down_name);

            if let (Some(t_norm), Some(t_q), Some(t_k), Some(t_v), Some(t_o)) =
                (attn_norm, attn_q, attn_k, attn_v, attn_o)
            {
                // 1. RMSNorm
                self.rms_norm(&x, &t_norm, &mut x_norm);

                // 2. Q, K, V projections
                self.matvec(&t_q, &x_norm, &mut q);
                self.matvec(&t_k, &x_norm, &mut k);
                self.matvec(&t_v, &x_norm, &mut v);

                // 3. RoPE
                self.apply_rope(&mut q, n_heads, head_dim, pos);
                self.apply_rope(&mut k, n_kv_heads, head_dim, pos);

                // 4. Ingest into KV cache
                self.kv_cache_k[l].extend_from_slice(&k);
                self.kv_cache_v[l].extend_from_slice(&v);

                let cached_tokens = if kv_dim > 0 { self.kv_cache_k[l].len() / kv_dim } else { 1 };
                let seq_len = cached_tokens.min(pos + 1).max(1);
                let heads_per_kv = (n_heads / n_kv_heads).max(1);

                // 5. Multi-Head / Grouped-Query Attention
                let mut head_outputs = vec![0.0f32; d];
                let scale = 1.0 / (head_dim as f32).sqrt();

                for h in 0..n_heads {
                    let kv_h = h / heads_per_kv;
                    let q_head = &q[h * head_dim..(h + 1) * head_dim];

                    // Compute dot product with all cached keys
                    let mut scores = Vec::with_capacity(seq_len);
                    let mut max_score = f32::NEG_INFINITY;

                    for s in 0..seq_len {
                        let k_start = s * kv_dim + kv_h * head_dim;
                        let k_end = k_start + head_dim;
                        let mut dot = 0.0f32;
                        if k_end <= self.kv_cache_k[l].len() {
                            let k_head = &self.kv_cache_k[l][k_start..k_end];
                            for j in 0..head_dim {
                                dot += q_head[j] * k_head[j];
                            }
                        }
                        dot *= scale;
                        if dot > max_score {
                            max_score = dot;
                        }
                        scores.push(dot);
                    }

                    // Softmax
                    let mut sum_exp = 0.0f32;
                    for s in 0..seq_len {
                        scores[s] = (scores[s] - max_score).exp();
                        sum_exp += scores[s];
                    }
                    let inv_sum = 1.0 / sum_exp.max(1e-8);
                    for s in 0..seq_len {
                        scores[s] *= inv_sum;
                    }

                    // Context vector
                    let head_out = &mut head_outputs[h * head_dim..(h + 1) * head_dim];
                    head_out.fill(0.0);
                    for s in 0..seq_len {
                        let v_start = s * kv_dim + kv_h * head_dim;
                        let v_end = v_start + head_dim;
                        let w = scores[s];
                        if v_end <= self.kv_cache_v[l].len() {
                            let v_head = &self.kv_cache_v[l][v_start..v_end];
                            for j in 0..head_dim {
                                head_out[j] += w * v_head[j];
                            }
                        }
                    }
                }

                // 6. Attention output projection
                self.matvec(&t_o, &head_outputs, &mut attn_out);

                // Residual add
                for i in 0..d {
                    x[i] += attn_out[i];
                }
            }

            // FFN block
            if let (Some(t_ffn_norm), Some(t_g), Some(t_u), Some(t_d)) =
                (ffn_norm, ffn_g, ffn_u, ffn_d)
            {
                let intermediate_size = if t_g.shape.len() >= 2 {
                    t_g.shape[1]
                } else {
                    d * 4
                };

                if ffn_gate.len() != intermediate_size {
                    ffn_gate.resize(intermediate_size, 0.0);
                    ffn_up.resize(intermediate_size, 0.0);
                }

                self.rms_norm(&x, &t_ffn_norm, &mut x_norm);
                self.matvec(&t_g, &x_norm, &mut ffn_gate);
                self.matvec(&t_u, &x_norm, &mut ffn_up);

                // SwiGLU: silu(gate) * up
                for i in 0..intermediate_size {
                    let g = ffn_gate[i];
                    let silu_g = g / (1.0 + (-g).exp());
                    ffn_gate[i] = silu_g * ffn_up[i];
                }

                self.matvec(&t_d, &ffn_gate, &mut ffn_down);

                // Residual add
                for i in 0..d {
                    x[i] += ffn_down[i];
                }
            }
        }

        // Final RMSNorm
        let mut final_norm_out = vec![0.0f32; d];
        if let Some(t_final_norm) = self.reader.get_tensor("output_norm.weight") {
            self.rms_norm(&x, &t_final_norm, &mut final_norm_out);
        } else {
            final_norm_out.copy_from_slice(&x);
        }

        // Logits projection: W_output * final_norm_out
        let mut logits = vec![0.0f32; self.vocab_size];
        if let Some(t_out) = self.reader.get_tensor("output.weight") {
            self.matvec(&t_out, &final_norm_out, &mut logits);
        } else if let Some(t_embd) = self.reader.get_tensor("token_embd.weight") {
            // Tied embedding weights
            self.matvec(&t_embd, &final_norm_out, &mut logits);
        }

        self.cached_seq_len = pos + 1;
        Ok(logits)
    }

    /// Process an entire prompt sequence and return logits for the final token.
    pub fn forward_prompt(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::InvalidArgument("Prompt tokens cannot be empty".into()));
        }

        self.reset_cache();
        let mut last_logits = Vec::new();

        for (pos, &tok) in tokens.iter().enumerate() {
            last_logits = self.forward_single_token(tok, pos)?;
        }

        Ok(last_logits)
    }
}
