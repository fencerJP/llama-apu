// SPDX-License-Identifier: Apache-2.0
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use zero_copy_model_runner::engine::{ImatrixProvenance, ImatrixStore, LLAMA_APU_PROVENANCE_TAG};

#[test]
fn test_adjacent_path_resolution() {
    let model_p = PathBuf::from("/opt/models/qwen-122b.gguf");
    let imatrix_p = ImatrixStore::adjacent_imatrix_path(&model_p);
    assert_eq!(imatrix_p, PathBuf::from("/opt/models/qwen-122b.imatrix.gguf"));

    let q4nx_p = PathBuf::from("/opt/models/neohorse.q4nx");
    let q4nx_imatrix_p = ImatrixStore::adjacent_imatrix_path(&q4nx_p);
    assert_eq!(q4nx_imatrix_p, PathBuf::from("/opt/models/neohorse.imatrix.gguf"));
}

#[test]
fn test_external_imatrix_read_only_protection() {
    let tmp_dir = std::env::temp_dir().join("test_imatrix_ext");
    let _ = fs::create_dir_all(&tmp_dir);

    let model_path = tmp_dir.join("external_model.gguf");
    let imatrix_path = tmp_dir.join("external_model.imatrix.gguf");

    // Write external imatrix WITHOUT llama-apu provenance tag
    {
        let mut f = File::create(&imatrix_path).unwrap();
        writeln!(f, "{{\"0\": [10, 20, 30]}}").unwrap();
    }

    let mut store = ImatrixStore::open_or_discover(&model_path);
    assert_eq!(store.provenance, ImatrixProvenance::ExternalReadOnly);
    assert!(!store.is_updateable());

    // Record activation should NOT mutate the read-only file
    store.record_activation(0, 0);
    assert!(store.save_to_disk().is_ok());

    let content = fs::read_to_string(&imatrix_path).unwrap();
    assert!(!content.contains(LLAMA_APU_PROVENANCE_TAG));

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_native_imatrix_creation_and_persistence() {
    let tmp_dir = std::env::temp_dir().join("test_imatrix_native");
    let _ = fs::create_dir_all(&tmp_dir);

    let model_path = tmp_dir.join("native_model.gguf");
    let mut store = ImatrixStore::open_or_discover(&model_path);
    assert_eq!(store.provenance, ImatrixProvenance::NotPresent);

    let mut priors = HashMap::new();
    priors.insert(0, vec![0.5f32, 0.3f32, 0.2f32]);

    assert!(store.create_native_imatrix(priors).is_ok());
    assert_eq!(store.provenance, ImatrixProvenance::NativeUpdateable);
    assert!(store.is_updateable());

    let imatrix_path = ImatrixStore::adjacent_imatrix_path(&model_path);
    assert!(imatrix_path.exists());

    let content = fs::read_to_string(&imatrix_path).unwrap();
    assert!(content.contains(LLAMA_APU_PROVENANCE_TAG));

    // Re-open and verify native provenance is recognized
    let reloaded = ImatrixStore::open_or_discover(&model_path);
    assert_eq!(reloaded.provenance, ImatrixProvenance::NativeUpdateable);

    let _ = fs::remove_dir_all(&tmp_dir);
}
