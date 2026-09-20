// SPDX-License-Identifier: Apache-2.0
//! Unit tests for BiLLM (1.08 bpw) quantization and SpinQuant precedence.

use zero_copy_model_runner::container::converter::is_quantization_supported;
use zero_copy_model_runner::container::reader::{TensorDType, TensorView};

#[test]
fn test_billm_and_q1_quantization_support() {
    // Standard supported quants
    assert!(is_quantization_supported("Q4_0"));
    assert!(is_quantization_supported("Q4_K_M"));
    assert!(is_quantization_supported("IQ4_NL"));
    assert!(is_quantization_supported("Q8_0"));
    assert!(is_quantization_supported("F16"));
    assert!(is_quantization_supported("BF16"));

    // BiLLM and 1-bit formats
    assert!(is_quantization_supported("BILLM"));
    assert!(is_quantization_supported("Q1_BILLM"));
    assert!(is_quantization_supported("Q1_0"));
    assert!(is_quantization_supported("Q1_0_G128"));

    // Legacy unaligned formats remain rejected
    assert!(!is_quantization_supported("IQ1_S"));
    assert!(!is_quantization_supported("Q2_K"));
    assert!(!is_quantization_supported("IQ3_S"));
}

#[test]
fn test_tensordtype_apu_support() {
    assert!(TensorDType::Q1_0.is_supported_on_apu());
    assert!(TensorDType::BILLM.is_supported_on_apu());
    assert!(TensorDType::Q4_0.is_supported_on_apu());
    assert!(TensorDType::Q8_0.is_supported_on_apu());
    assert!(!TensorDType::IQ1_S.is_supported_on_apu());
    assert!(!TensorDType::Q2_K.is_supported_on_apu());
}

#[test]
fn test_billm_dot_product_computation() {
    let cols = 128;
    let rows = 1;
    let shape = [cols, rows];

    // Create a mock 18-byte BiLLM block:
    // Scale: FP16 1.0 (0x3C00)
    // Signs: All 1s (0xFF repeated 16 times -> all weights are +1.0)
    let mut data = Vec::new();
    data.extend_from_slice(&0x3C00u16.to_le_bytes()); // FP16 1.0
    data.extend_from_slice(&[0xFFu8; 16]); // All bits 1

    let tensor = TensorView {
        name: "test.weight",
        shape: &shape,
        dtype: TensorDType::BILLM,
        data: &data,
    };

    let in_vec = vec![1.0f32; cols];
    let result = tensor.dot_product_row(0, &in_vec);

    // 128 weights of +1.0 multiplied by 1.0 = 128.0
    assert!((result - 128.0).abs() < 1e-3, "Expected dot product 128.0, got {}", result);

    // If signs are alternating (0xAA -> 10101010)
    let mut data_alt = Vec::new();
    data_alt.extend_from_slice(&0x3C00u16.to_le_bytes()); // FP16 1.0
    data_alt.extend_from_slice(&[0xAAu8; 16]); // Half 1s, half 0s (+1 and -1)

    let tensor_alt = TensorView {
        name: "test_alt.weight",
        shape: &shape,
        dtype: TensorDType::BILLM,
        data: &data_alt,
    };

    let result_alt = tensor_alt.dot_product_row(0, &in_vec);
    // 64 of +1.0 and 64 of -1.0 = 0.0
    assert!(result_alt.abs() < 1e-3, "Expected dot product 0.0, got {}", result_alt);
}
