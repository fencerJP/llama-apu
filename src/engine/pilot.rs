// SPDX-License-Identifier: Apache-2.0
//! Dual-Scale EWMA & PILOT Multi-Layer Lookahead Prefetch Engine.
//!
//! Implements online adaptive expert hotness tracking with topic-shift resilience,
//! hysteresis guard-bands to prevent cache ping-ponging, and multi-layer lookahead
//! prefetching with confidence thresholding.

use std::collections::HashMap;

/// Configuration for the Dual-Scale EWMA & PILOT lookahead engine.
#[derive(Debug, Clone)]
pub struct PilotConfig {
    /// Short-term EWMA decay factor (default 0.02, ~50 token window).
    pub alpha_decay: f32,
    /// Balance weight between short-term frequency and baseline prior (default 0.70).
    pub lambda_short: f32,
    /// Minimum floor probability to prevent prior erosion (default 0.01).
    pub floor_epsilon: f32,
    /// Upper hysteresis threshold for DRAM hot pool promotion.
    pub tau_high: f32,
    /// Lower hysteresis threshold for NVMe cold pool demotion.
    pub tau_low: f32,
    /// Minimum confidence threshold for issuing speculative lookahead fetches.
    pub lookahead_confidence_thresh: f32,
    /// Number of layers to look ahead (default 2).
    pub lookahead_depth: usize,
    /// Cumulative transition mass threshold for lookahead pruning (default 0.90).
    pub pilot_mass: f32,
}

impl Default for PilotConfig {
    fn default() -> Self {
        Self {
            alpha_decay: 0.02,
            lambda_short: 0.70,
            floor_epsilon: 0.01,
            tau_high: 0.08,
            tau_low: 0.03,
            lookahead_confidence_thresh: 0.80,
            lookahead_depth: 2,
            pilot_mass: 0.90,
        }
    }
}

/// Dynamic tracker for a single layer's expert routing statistics.
#[derive(Debug, Clone)]
pub struct LayerExpertTracker {
    pub num_experts: usize,
    pub short_term_freq: Vec<f32>,
    pub baseline_prior: Vec<f32>,
    pub composite_scores: Vec<f32>,
    pub is_hot_resident: Vec<bool>,
    /// Layer-to-layer Markov transition matrix: trans[from_expert * num_experts + to_expert].
    pub transition_matrix: Vec<u64>,
}

impl LayerExpertTracker {
    pub fn new(num_experts: usize, initial_prior: Vec<f32>) -> Self {
        let prior = if initial_prior.len() == num_experts {
            initial_prior
        } else {
            vec![1.0 / num_experts.max(1) as f32; num_experts]
        };

        Self {
            num_experts,
            short_term_freq: prior.clone(),
            baseline_prior: prior.clone(),
            composite_scores: prior,
            is_hot_resident: vec![false; num_experts],
            transition_matrix: vec![0; num_experts * num_experts],
        }
    }

    /// Update routing statistics after a token routing step.
    pub fn update_step(&mut self, active_experts: &[usize], config: &PilotConfig) {
        let num_active = active_experts.len().max(1) as f32;

        for e in 0..self.num_experts {
            let was_active = if active_experts.contains(&e) { 1.0 / num_active } else { 0.0 };
            // Update short-term EWMA
            self.short_term_freq[e] = (1.0 - config.alpha_decay) * self.short_term_freq[e] + config.alpha_decay * was_active;

            // Compute composite score: S_e = lambda * F_short + (1 - lambda) * F_prior with floor constraint
            let composite = config.lambda_short * self.short_term_freq[e]
                + (1.0 - config.lambda_short) * self.baseline_prior[e];

            self.composite_scores[e] = composite.max(config.floor_epsilon);
        }

        // Apply hysteresis promotion / demotion
        for e in 0..self.num_experts {
            if !self.is_hot_resident[e] && self.composite_scores[e] >= config.tau_high {
                self.is_hot_resident[e] = true;
            } else if self.is_hot_resident[e] && self.composite_scores[e] <= config.tau_low {
                self.is_hot_resident[e] = false;
            }
        }
    }

    /// Record a layer-to-layer transition from expert `from` at layer L-1 to `to` at layer L.
    pub fn record_transition(&mut self, from_expert: usize, to_expert: usize) {
        if from_expert < self.num_experts && to_expert < self.num_experts {
            let idx = from_expert * self.num_experts + to_expert;
            self.transition_matrix[idx] = self.transition_matrix[idx].saturating_add(1);
        }
    }

    /// Predict the most probable upcoming experts at this layer given the active expert at preceding layer.
    pub fn predict_lookahead_experts(&self, from_expert: usize, config: &PilotConfig) -> Vec<usize> {
        if from_expert >= self.num_experts {
            return Vec::new();
        }

        let row_start = from_expert * self.num_experts;
        let row = &self.transition_matrix[row_start..row_start + self.num_experts];
        let total_hits: u64 = row.iter().sum();

        if total_hits == 0 {
            // No transitions learned yet: fall back to top baseline composite scores
            let mut indexed: Vec<(usize, f32)> = self.composite_scores.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return indexed.into_iter().take(2).map(|(i, _)| i).collect();
        }

        let mut ranked: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .map(|(e, &hits)| (e, hits as f32 / total_hits as f32))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = Vec::new();
        let mut cum_mass = 0.0f32;

        for (e, prob) in ranked {
            if prob >= config.lookahead_confidence_thresh || cum_mass < config.pilot_mass {
                out.push(e);
                cum_mass += prob;
            }
            if cum_mass >= config.pilot_mass {
                break;
            }
        }

        out
    }
}

/// Global PILOT Lookahead Coordinator across all model layers.
#[derive(Debug, Clone)]
pub struct PilotEngine {
    pub config: PilotConfig,
    pub layer_trackers: HashMap<usize, LayerExpertTracker>,
}

impl PilotEngine {
    pub fn new(config: PilotConfig) -> Self {
        Self {
            config,
            layer_trackers: HashMap::new(),
        }
    }

    /// Register a layer with its baseline analytical prior.
    pub fn register_layer(&mut self, layer_idx: usize, num_experts: usize, prior: Vec<f32>) {
        self.layer_trackers.insert(layer_idx, LayerExpertTracker::new(num_experts, prior));
    }

    /// Process a token routing step for layer `layer_idx`.
    pub fn on_layer_routed(
        &mut self,
        layer_idx: usize,
        active_experts: &[usize],
        prev_layer_experts: Option<&[usize]>,
    ) {
        if let Some(prev) = prev_layer_experts {
            if let Some(tracker) = self.layer_trackers.get_mut(&layer_idx) {
                for &p in prev {
                    for &curr in active_experts {
                        tracker.record_transition(p, curr);
                    }
                }
            }
        }

        if let Some(tracker) = self.layer_trackers.get_mut(&layer_idx) {
            tracker.update_step(active_experts, &self.config);
        }
    }

    /// Predict the target experts to prefetch for layer `target_layer_idx`.
    pub fn predict_prefetch_targets(
        &self,
        current_layer_idx: usize,
        current_active_experts: &[usize],
    ) -> Vec<(usize, usize)> {
        let target_layer = current_layer_idx + 1;
        let mut prefetch_list = Vec::new();

        if let Some(tracker) = self.layer_trackers.get(&target_layer) {
            for &from_e in current_active_experts {
                let predicted = tracker.predict_lookahead_experts(from_e, &self.config);
                for target_e in predicted {
                    if !tracker.is_hot_resident[target_e] {
                        prefetch_list.push((target_layer, target_e));
                    }
                }
            }
        }

        prefetch_list
    }
}
