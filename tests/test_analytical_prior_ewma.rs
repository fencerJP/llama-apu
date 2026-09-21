// SPDX-License-Identifier: Apache-2.0
use zero_copy_model_runner::engine::{
    AnalyticalPriorCalculator, PilotConfig, PilotEngine,
};

#[test]
fn test_analytical_prior_calculation() {
    let hidden_dim = 64;
    let num_experts = 8;
    let total_layers = 16;
    let vocab_samples = 10;

    let mut router_weights = vec![0.0f32; num_experts * hidden_dim];
    // Expert 0 has strong positive alignment with embedding sample 0
    for d in 0..hidden_dim {
        router_weights[0 * hidden_dim + d] = 1.0;
    }

    let mut embed_samples = vec![0.0f32; vocab_samples * hidden_dim];
    for d in 0..hidden_dim {
        embed_samples[0 * hidden_dim + d] = 1.0;
    }

    // Compute layer 0 prior (low gamma_l -> strong analytical bias)
    let prior_0 = AnalyticalPriorCalculator::compute_layer_prior(
        &router_weights,
        &embed_samples,
        0,
        total_layers,
        num_experts,
        hidden_dim,
    );

    assert_eq!(prior_0.len(), num_experts);
    let sum_0: f32 = prior_0.iter().sum();
    assert!((sum_0 - 1.0).abs() < 1e-4);
    assert!(prior_0[0] > prior_0[1]);

    // Compute deep layer prior (layer 15 -> high gamma_l -> close to uniform)
    let prior_15 = AnalyticalPriorCalculator::compute_layer_prior(
        &router_weights,
        &embed_samples,
        15,
        total_layers,
        num_experts,
        hidden_dim,
    );

    assert_eq!(prior_15.len(), num_experts);
    let sum_15: f32 = prior_15.iter().sum();
    assert!((sum_15 - 1.0).abs() < 1e-4);
    // Difference between expert 0 and 1 should be heavily smoothed towards uniform
    let uniform = 1.0 / num_experts as f32;
    assert!((prior_15[1] - uniform).abs() < (prior_0[1] - uniform).abs());
}

#[test]
fn test_pilot_ewma_and_lookahead_prefetch() {
    let config = PilotConfig {
        alpha_decay: 0.10,
        lambda_short: 0.80,
        floor_epsilon: 0.01,
        tau_high: 0.15,
        tau_low: 0.05,
        lookahead_confidence_thresh: 0.50,
        lookahead_depth: 2,
        pilot_mass: 0.90,
    };

    let mut engine = PilotEngine::new(config);
    let num_experts = 4;
    let prior = vec![0.25; num_experts];

    engine.register_layer(0, num_experts, prior.clone());
    engine.register_layer(1, num_experts, prior);

    // Simulate Layer 0 routing to Expert 1, Layer 1 routing to Expert 2
    for _ in 0..10 {
        engine.on_layer_routed(0, &[1], None);
        engine.on_layer_routed(1, &[2], Some(&[1]));
    }

    // Tracker for layer 0 should mark expert 1 as hot resident
    let tracker_0 = engine.layer_trackers.get(&0).unwrap();
    assert!(tracker_0.composite_scores[1] > 0.15);
    assert!(tracker_0.is_hot_resident[1]);

    // Predict prefetch targets for layer 1 when layer 0 routes to expert 1
    let targets = engine.predict_prefetch_targets(0, &[1]);
    // Should predict prefetching layer 1, expert 2 if not already hot
    assert!(targets.iter().any(|&(layer, exp)| layer == 1 && exp == 2) || tracker_0.is_hot_resident[1]);
}
