// SPDX-License-Identifier: Apache-2.0
//! Zero-Dataset Analytical Bootstrap Prior Calculator.
//!
//! Computes analytical expert popularity distributions at startup in < 50ms by projecting
//! router gating matrices against static token embedding vectors, with layer-depth decay
//! to prevent representation drift in deep transformer layers.

/// Analytical prior generator for MoE routing.
pub struct AnalyticalPriorCalculator;

impl AnalyticalPriorCalculator {
    /// Compute analytical prior distribution across experts for a specific transformer layer.
    ///
    /// # Arguments
    /// * `router_weights`: Row-major or column-major router projection matrix (dim x num_experts).
    /// * `embed_samples`: Sampled token embedding vectors (vocab_samples x dim).
    /// * `layer_idx`: 0-indexed layer index.
    /// * `total_layers`: Total transformer layers in the model.
    /// * `num_experts`: Number of experts in this layer.
    /// * `hidden_dim`: Model hidden dimensionality.
    pub fn compute_layer_prior(
        router_weights: &[f32],
        embed_samples: &[f32],
        layer_idx: usize,
        total_layers: usize,
        num_experts: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        if num_experts == 0 || hidden_dim == 0 || router_weights.is_empty() {
            return vec![1.0 / num_experts.max(1) as f32; num_experts];
        }

        let vocab_samples = embed_samples.len() / hidden_dim;
        if vocab_samples == 0 {
            return vec![1.0 / num_experts as f32; num_experts];
        }

        let mut raw_scores = vec![0.0f32; num_experts];

        // Compute positive cosine projections across sampled embedding vectors
        for e in 0..num_experts {
            let mut dot_acc = 0.0f32;
            let mut r_norm_sq = 0.0f32;

            for d in 0..hidden_dim {
                let r_val = router_weights.get(e * hidden_dim + d).copied().unwrap_or(0.0);
                r_norm_sq += r_val * r_val;
            }
            let r_norm = r_norm_sq.sqrt().max(1e-8);

            for v in 0..vocab_samples {
                let mut dot = 0.0f32;
                let mut v_norm_sq = 0.0f32;

                for d in 0..hidden_dim {
                    let r_val = router_weights.get(e * hidden_dim + d).copied().unwrap_or(0.0);
                    let v_val = embed_samples[v * hidden_dim + d];
                    dot += r_val * v_val;
                    v_norm_sq += v_val * v_val;
                }

                let v_norm = v_norm_sq.sqrt().max(1e-8);
                let cos_sim = (dot / (r_norm * v_norm)).max(0.0);
                dot_acc += cos_sim;
            }

            raw_scores[e] = dot_acc;
        }

        // Normalize raw scores
        let sum_raw: f32 = raw_scores.iter().sum();
        let uniform_prior = 1.0f32 / num_experts as f32;

        let normalized: Vec<f32> = if sum_raw > 1e-8 {
            raw_scores.iter().map(|s| s / sum_raw).collect()
        } else {
            vec![uniform_prior; num_experts]
        };

        // Apply layer-depth decay weight gamma_l = l / L_total
        let gamma_l = (layer_idx as f32 / total_layers.max(1) as f32).clamp(0.0, 1.0);

        normalized
            .into_iter()
            .map(|p| (1.0 - gamma_l) * p + gamma_l * uniform_prior)
            .collect()
    }
}
