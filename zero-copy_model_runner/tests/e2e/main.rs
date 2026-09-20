// SPDX-License-Identifier: Apache-2.0
//! Main Entry Point for Opaque-Box E2E Test Suite
//!
//! Organizes tests across 4 tiers:
//! - Tier 1: Feature Coverage (R1–R5, 25 tests)
//! - Tier 2: Boundary & Corner Cases (R1–R5, 25 tests)
//! - Tier 3: Cross-Feature Combinations (5 tests)
//! - Tier 4: Real-World Application Scenarios (5 tests)
//! Total: 60 comprehensive E2E tests

#[path = "../common/mod.rs"]
pub mod common;

mod tier1_feature_coverage;
mod tier2_boundary_corner;
mod tier3_cross_feature;
mod tier4_real_world;
