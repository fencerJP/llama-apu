// SPDX-License-Identifier: Apache-2.0
//! High-Performance SIMD-Accelerated Token Sampler for AMD Zen 5 Classic Cores.
//!
//! Implements greedy (argmax) and multinomial sampling with temperature scaling,
//! Top-P (nucleus) filtering, and AVX-512 dual-pipe SIMD vector acceleration on
//! AMD Zen 5 execution units.

use super::EngineError;

/// Configuration parameters for token generation sampling.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature scaling factor. Values <= 0.0 trigger greedy argmax sampling.
    pub temperature: f32,
    /// Cumulative probability threshold for nucleus (Top-P) sampling (0.0 to 1.0].
    pub top_p: f32,
    /// Optional top-k candidate truncation (0 disables top-k).
    pub top_k: usize,
    /// Optional deterministic PRNG seed for reproducible test generation.
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        }
    }
}

/// Fast, deterministic 64-bit pseudo-random number generator (SplitMix64 / XorShift64).
#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x853c_49e6_748f_ea9b } else { seed };
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Sample a uniform float in the half-open interval `[0.0, 1.0)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let val = self.next_u64() >> 40; // 24 bits of mantissa
        (val as f32) / ((1u64 << 24) as f32)
    }
}

/// High-throughput token sampler executing on Zen 5 Classic cores.
#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplerConfig,
    rng: FastRng,
}

impl Sampler {
    /// Instantiate a new `Sampler` with the provided configuration.
    pub fn new(config: SamplerConfig) -> Self {
        let seed = config.seed.unwrap_or(0x1234_5678_9ABC_DEF0);
        Self {
            config,
            rng: FastRng::new(seed),
        }
    }

    /// Return the current configuration.
    #[inline]
    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Update the sampler configuration.
    pub fn set_config(&mut self, config: SamplerConfig) {
        if let Some(seed) = config.seed {
            self.rng = FastRng::new(seed);
        }
        self.config = config;
    }

    /// Re-seed the internal deterministic pseudo-random number generator.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = FastRng::new(seed);
    }

    /// Pure greedy argmax sampling: returns the index of the highest logit.
    ///
    /// Automatically leverages Zen 5 AVX-512 SIMD when available on x86_64,
    /// with transparent fallback to scalar execution.
    pub fn sample_greedy(logits: &[f32]) -> Result<u32, EngineError> {
        if logits.is_empty() {
            return Err(EngineError::InvalidArgument("Logits slice cannot be empty".into()));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Ok(unsafe { argmax_avx512(logits) });
            }
        }

        Ok(argmax_scalar(logits))
    }

    /// Sample a token ID from raw vocabulary logits according to configured temperature and top-p.
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32, EngineError> {
        self.sample_with_params(logits, self.config.temperature, self.config.top_p)
    }

    /// Sample a token ID with explicit per-request temperature and top-p overrides.
    pub fn sample_with_params(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
    ) -> Result<u32, EngineError> {
        if logits.is_empty() {
            return Err(EngineError::InvalidArgument("Logits slice cannot be empty".into()));
        }

        // Greedy decoding fast-path
        if temperature <= 0.0 || temperature.is_nan() {
            return Self::sample_greedy(logits);
        }

        let len = logits.len();
        if len == 1 {
            return Ok(0);
        }

        // 1. Find global max logit for numerical stability
        let max_val = {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    let idx = unsafe { argmax_avx512(logits) };
                    logits[idx as usize]
                } else {
                    let idx = argmax_scalar(logits);
                    logits[idx as usize]
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let idx = argmax_scalar(logits);
                logits[idx as usize]
            }
        };

        // 2. Softmax computation with temperature scaling
        let inv_temp = 1.0 / temperature;
        let mut exp_logits = Vec::with_capacity(len);
        let mut sum_exp = 0.0f32;

        for &l in logits {
            let shifted = (l - max_val) * inv_temp;
            let e = shifted.exp();
            exp_logits.push(e);
            sum_exp += e;
        }

        if sum_exp <= 0.0 || sum_exp.is_nan() {
            // Degenerate distribution: fall back to argmax
            return Self::sample_greedy(logits);
        }

        let inv_sum = 1.0 / sum_exp;
        let mut probs: Vec<(u32, f32)> = exp_logits
            .into_iter()
            .enumerate()
            .map(|(i, e)| (i as u32, e * inv_sum))
            .collect();

        // 3. Optional Top-K truncation
        if self.config.top_k > 0 && self.config.top_k < probs.len() {
            probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            probs.truncate(self.config.top_k);
        }

        // 4. Top-P (Nucleus) Filtering
        let clamped_top_p = top_p.clamp(0.0, 1.0);
        if clamped_top_p < 1.0 {
            if self.config.top_k == 0 {
                probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            let mut cumsum = 0.0f32;
            let mut cutoff = probs.len();
            for (idx, &(_, p)) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= clamped_top_p {
                    cutoff = idx + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
        }

        // Re-normalize probabilities over surviving nucleus
        let nucleus_sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if nucleus_sum <= 0.0 {
            return Ok(probs.first().map(|&(t, _)| t).unwrap_or(0));
        }

        // 5. Multinomial draw
        let mut r = self.rng.next_f32() * nucleus_sum;
        for &(token_id, prob) in &probs {
            r -= prob;
            if r <= 0.0 {
                return Ok(token_id);
            }
        }

        // Guard against precision round-off
        Ok(probs.last().map(|&(t, _)| t).unwrap_or(0))
    }
}

/// Fallback scalar argmax loop.
#[inline]
pub fn argmax_scalar(logits: &[f32]) -> u32 {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0u32;
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i as u32;
        }
    }
    max_idx
}

/// AVX-512 SIMD argmax executing across dual 512-bit Zen 5 vector pipelines.
///
/// Processes 16 32-bit floating-point values per instruction iteration, tracking
/// winning elements and indices in parallel ZMM register sets.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn argmax_avx512(logits: &[f32]) -> u32 {
    use std::arch::x86_64::*;

    let len = logits.len();
    assert!(len > 0);
    let chunks = len / 16;
    let ptr = logits.as_ptr();

    let mut global_max = f32::NEG_INFINITY;
    let mut global_idx = 0u32;

    if chunks > 0 {
        let mut v_max = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut v_idx = _mm512_set1_epi32(0);
        let v_step = _mm512_set1_epi32(16);
        let mut current_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        for c in 0..chunks {
            let v_data = _mm512_loadu_ps(ptr.add(c * 16));
            let mask = _mm512_cmp_ps_mask(v_data, v_max, _CMP_GT_OQ);
            v_max = _mm512_mask_mov_ps(v_max, mask, v_data);
            v_idx = _mm512_mask_mov_epi32(v_idx, mask, current_idx);
            current_idx = _mm512_add_epi32(current_idx, v_step);
        }

        let mut vals = [0.0f32; 16];
        let mut idxs = [0i32; 16];
        _mm512_storeu_ps(vals.as_mut_ptr(), v_max);
        _mm512_storeu_si512(idxs.as_mut_ptr() as *mut __m512i, v_idx);

        for i in 0..16 {
            if vals[i] > global_max {
                global_max = vals[i];
                global_idx = idxs[i] as u32;
            }
        }
    }

    // Process remaining tail elements (< 16)
    let tail_start = chunks * 16;
    for i in tail_start..len {
        let val = *logits.get_unchecked(i);
        if val > global_max {
            global_max = val;
            global_idx = i as u32;
        }
    }

    global_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_greedy_argmax_basic() {
        let mut logits = vec![0.0f32; 1000];
        logits[777] = 42.5;
        let selected = Sampler::sample_greedy(&logits).expect("sample_greedy failed");
        assert_eq!(selected, 777);
    }

    #[test]
    fn test_sampler_empty_logits_rejected() {
        let logits: [f32; 0] = [];
        let res = Sampler::sample_greedy(&logits);
        assert!(res.is_err());
    }

    #[test]
    fn test_sampler_temperature_zero_is_greedy() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: Some(42),
        });

        let mut logits = vec![-1.0f32; 128256];
        logits[128001] = 100.0;
        let token = sampler.sample(&logits).expect("sample failed");
        assert_eq!(token, 128001);
    }

    #[test]
    fn test_sampler_multinomial_distribution_and_seed_determinism() {
        let mut sampler1 = Sampler::new(SamplerConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 10,
            seed: Some(1337),
        });

        let mut sampler2 = Sampler::new(SamplerConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 10,
            seed: Some(1337),
        });

        let logits: Vec<f32> = (0..100).map(|i| (i as f32).sin() * 5.0).collect();

        let s1_tokens: Vec<u32> = (0..20).map(|_| sampler1.sample(&logits).unwrap()).collect();
        let s2_tokens: Vec<u32> = (0..20).map(|_| sampler2.sample(&logits).unwrap()).collect();

        assert_eq!(s1_tokens, s2_tokens, "Deterministic seeds must generate identical token streams");
    }

    #[test]
    fn test_avx512_and_scalar_equivalence() {
        let len = 128000;
        let mut logits: Vec<f32> = (0..len).map(|i| ((i * 37) % 1000) as f32 / 100.0).collect();
        logits[98765] = 10000.0; // Clear maximum

        let scalar_result = argmax_scalar(&logits);
        assert_eq!(scalar_result, 98765);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") {
            let avx512_result = unsafe { argmax_avx512(&logits) };
            assert_eq!(avx512_result, scalar_result, "AVX-512 and scalar argmax must yield identical results");
        }
    }
}
