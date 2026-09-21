// SPDX-License-Identifier: Apache-2.0
//! XCLBIN Profile Resolver for AMD Ryzen AI NPU.
//!
//! Scans local banks of XCLBIN profiles, performs automated compatibility matching
//! based on model architecture and parameter count, and provides an interactive
//! selection fallback when ambiguous.

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

/// Default search directories for compiled XDNA XCLBIN profiles.
pub const DEFAULT_XCLBIN_SEARCH_PATHS: &[&str] = &[
    "/usr/local/share/llama-apu/xclbins",
    "/usr/local/share/apu-backend/xclbins",
    "/opt/llama-apu/xclbins",
    "/opt/fastflowlm/share/flm/xclbins",
    "/home/fencer/.openclaw/workspace/projects/fastflowlm/src/xclbins",
    "./xclbins",
    "../xclbins",
];

/// A resolved XCLBIN profile candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XclbinProfile {
    pub name: String,
    pub profile_dir: PathBuf,
    pub primary_xclbin_path: PathBuf,
}

impl XclbinProfile {
    /// Read the binary bytes of the primary XCLBIN hardware graph.
    pub fn read_bytes(&self) -> io::Result<Vec<u8>> {
        fs::read(&self.primary_xclbin_path)
    }
}

/// Discovers all available XCLBIN profile directories dynamically from the system search paths.
///
/// Any new XCLBIN profiles or base model directories dropped into the search paths
/// will be discovered on-the-fly and automatically added to the selection list.
pub fn discover_xclbin_profiles(custom_dir: Option<&Path>) -> Vec<XclbinProfile> {
    let mut search_paths: Vec<PathBuf> = Vec::new();

    if let Some(custom) = custom_dir {
        search_paths.push(custom.to_path_buf());
    }

    if let Ok(env_path) = std::env::var("LLAMA_APU_XCLBINS_DIR") {
        search_paths.push(PathBuf::from(env_path));
    }

    if let Ok(env_path) = std::env::var("APU_XCLBIN_DIR") {
        search_paths.push(PathBuf::from(env_path));
    }

    if let Ok(home) = std::env::var("HOME") {
        search_paths.push(PathBuf::from(home).join(".local/share/llama-apu/xclbins"));
    }

    for &default_path in DEFAULT_XCLBIN_SEARCH_PATHS {
        search_paths.push(PathBuf::from(default_path));
    }

    let mut profiles = Vec::new();
    let mut seen_names = std::collections::HashSet::new();

    for base_dir in search_paths {
        if !base_dir.is_dir() {
            continue;
        }

        scan_directory_for_profiles(&base_dir, 0, &mut profiles, &mut seen_names);
    }

    // Sort alphabetically for deterministic ordering
    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    profiles
}

/// Helper to recursively scan up to depth 3 for .xclbin files and profile folders.
fn scan_directory_for_profiles(
    dir: &Path,
    current_depth: usize,
    profiles: &mut Vec<XclbinProfile>,
    seen_names: &mut std::collections::HashSet<String>,
) {
    if current_depth > 3 {
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            let dir_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            // Check if this directory directly contains an XCLBIN kernel
            let candidates = ["layer.xclbin", "mm.xclbin", "attn.xclbin"];
            let mut found_xclbin = None;

            for candidate in candidates {
                let cand_path = path.join(candidate);
                if cand_path.is_file() {
                    found_xclbin = Some(cand_path);
                    break;
                }
            }

            // Fallback: check for any .xclbin directly in this directory
            if found_xclbin.is_none() {
                if let Ok(sub_entries) = fs::read_dir(&path) {
                    for sub_entry in sub_entries.flatten() {
                        let sub_path = sub_entry.path();
                        if sub_path.is_file()
                            && sub_path.extension().and_then(|ext| ext.to_str()) == Some("xclbin")
                        {
                            found_xclbin = Some(sub_path);
                            break;
                        }
                    }
                }
            }

            if let Some(primary_xclbin_path) = found_xclbin {
                if !seen_names.contains(&dir_name) {
                    seen_names.insert(dir_name.clone());
                    profiles.push(XclbinProfile {
                        name: dir_name,
                        profile_dir: path.clone(),
                        primary_xclbin_path,
                    });
                }
            } else {
                // Recurse into subdirectories
                scan_directory_for_profiles(&path, current_depth + 1, profiles, seen_names);
            }
        } else if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("xclbin") {
            // Direct standalone .xclbin file
            let file_stem = path
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            if !seen_names.contains(&file_stem) {
                seen_names.insert(file_stem.clone());
                let parent = path.parent().unwrap_or(dir).to_path_buf();
                profiles.push(XclbinProfile {
                    name: file_stem,
                    profile_dir: parent,
                    primary_xclbin_path: path,
                });
            }
        }
    }
}

/// Normalize an architecture or model name for fuzzy matching (lowercase, alphanumeric only).
fn normalize_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Automatically match a model identifier against discovered XCLBIN profiles.
pub fn auto_match_profile<'a>(
    model_identifier: &str,
    profiles: &'a [XclbinProfile],
) -> Option<&'a XclbinProfile> {
    if profiles.is_empty() {
        return None;
    }

    let norm_model = normalize_key(model_identifier);

    // 1. Exact normalized match
    for profile in profiles {
        let norm_prof = normalize_key(&profile.name);
        if norm_prof == norm_model {
            return Some(profile);
        }
    }

    // 2. Substring containment match (profile in model or model in profile)
    let mut candidate = None;
    let mut candidate_len = 0;

    for profile in profiles {
        let norm_prof = normalize_key(&profile.name);
        if norm_model.contains(&norm_prof) || norm_prof.contains(&norm_model) {
            // Prefer the longer, more specific match
            if norm_prof.len() > candidate_len {
                candidate = Some(profile);
                candidate_len = norm_prof.len();
            }
        }
    }

    if candidate.is_some() {
        return candidate;
    }

    // 3. Keyword matching across known base model families
    let model_lower = model_identifier.to_lowercase();
    let base_families = [
        "llama", "qwen", "gemma", "phi", "deepseek", "mistral",
        "falcon", "starcoder", "baichuan", "nanbeige", "lfm",
        "whisper", "gpt", "aquila", "command", "refact", "spark", "ornith"
    ];

    let mut best_score = 0;
    let mut best_profile = None;

    for profile in profiles {
        let prof_lower = profile.name.to_lowercase();
        let mut score = 0;

        for &family in &base_families {
            if model_lower.contains(family) && prof_lower.contains(family) {
                score += 10;
            }
        }

        // Check parameter sizes
        for size in &[
            "300m", "500m", "600m", "800m", "0.5b", "0.6b", "0.8b", "1b", "1.2b", "1.5b",
            "1.7b", "2b", "2.5b", "2.6b", "3b", "4b", "8b", "9b", "14b", "20b", "35b", "70b"
        ] {
            if model_lower.contains(size) && prof_lower.contains(size) {
                score += 20;
            }
        }

        // Check version tags
        for ver in &["1.0", "1.5", "2.0", "2.5", "3.0", "3.1", "3.2", "3.5", "3.6", "4.0", "4.1"] {
            if model_lower.contains(ver) && prof_lower.contains(ver) {
                score += 15;
            }
        }

        if score > best_score {
            best_score = score;
            best_profile = Some(profile);
        }
    }

    if best_score >= 10 {
        return best_profile;
    }

    None
}

/// Interactively prompt the user to select an XCLBIN profile from a dynamically generated list.
///
/// Any new profiles dropped into the folder are automatically rendered in the selection menu.
/// If input stream is closed or non-interactive, falls back to the first available profile.
pub fn interactive_select_profile<'a, R: BufRead, W: Write>(
    profiles: &'a [XclbinProfile],
    mut input: R,
    mut output: W,
) -> io::Result<&'a XclbinProfile> {
    if profiles.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No XCLBIN profiles found in search paths",
        ));
    }

    writeln!(output, "\n========================================================")?;
    writeln!(output, "  AMD Ryzen AI NPU: Dynamic XCLBIN Profile Selection")?;
    writeln!(output, "========================================================")?;
    writeln!(
        output,
        "Discovered {} compatible hardware graph profiles in search paths.",
        profiles.len()
    )?;
    writeln!(
        output,
        "Please choose the target hardware graph profile for your model:\n"
    )?;

    for (i, prof) in profiles.iter().enumerate() {
        let xclbin_filename = prof
            .primary_xclbin_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("layer.xclbin");
        writeln!(output, "  [{:>2}] {} ({})", i + 1, prof.name, xclbin_filename)?;
    }
    writeln!(output, "--------------------------------------------------------")?;
    write!(output, "Enter selection [1-{}]: ", profiles.len())?;
    output.flush()?;

    let mut line = String::new();
    if input.read_line(&mut line)? > 0 {
        let trimmed = line.trim();
        if let Ok(choice) = trimmed.parse::<usize>() {
            if choice >= 1 && choice <= profiles.len() {
                return Ok(&profiles[choice - 1]);
            }
        }
    }

    // Default fallback to first profile if invalid or empty
    writeln!(
        output,
        "No valid selection entered. Defaulting to profile [1]: {}",
        profiles[0].name
    )?;
    Ok(&profiles[0])
}

/// High-level resolver: finds or prompts for the compatible XCLBIN profile.
pub fn resolve_xclbin_profile(
    model_identifier: &str,
    override_xclbin_path: Option<&Path>,
    custom_search_dir: Option<&Path>,
) -> io::Result<XclbinProfile> {
    // 1. Explicit override provided
    if let Some(override_path) = override_xclbin_path {
        let p = override_path;
        if p.is_file() {
            let name = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom")
                .to_string();
            let parent = p.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
            return Ok(XclbinProfile {
                name,
                profile_dir: parent,
                primary_xclbin_path: p.to_path_buf(),
            });
        }
    }

    // 2. Discover local profile bank
    let profiles = discover_xclbin_profiles(custom_search_dir);
    if profiles.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No XCLBIN profiles found in search directories",
        ));
    }

    // 3. Try automatic match
    if let Some(matched) = auto_match_profile(model_identifier, &profiles) {
        return Ok(matched.clone());
    }

    // 4. In non-interactive / daemon / server mode, auto-select best candidate without blocking
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 0 {
        let model_lower = model_identifier.to_lowercase();
        // Try to find any profile with overlapping family tokens
        let fallback = profiles
            .iter()
            .find(|p| {
                let prof_lower = p.name.to_lowercase();
                for term in &["deepseek", "qwen", "gemma", "llama", "phi", "mistral", "lfm", "whisper", "gpt"] {
                    if model_lower.contains(term) && prof_lower.contains(term) {
                        return true;
                    }
                }
                false
            })
            .unwrap_or(&profiles[0]);

        eprintln!(
            "[APU BACKEND] Non-interactive environment: auto-selected hardware profile [{}] for model '{}'",
            fallback.name, model_identifier
        );
        return Ok(fallback.clone());
    }

    // 5. Interactive fallback via stdin/stdout
    let stdin = io::stdin();
    let stdout = io::stdout();
    let selected = interactive_select_profile(&profiles, stdin.lock(), stdout)?;
    Ok(selected.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_and_auto_match_profiles() {
        let profiles = discover_xclbin_profiles(None);
        assert!(!profiles.is_empty(), "Should discover local XCLBIN profiles");

        // Test matching Llama-3.1-8B
        let llama_match = auto_match_profile("Meta-Llama-3.1-8B-Instruct.gguf", &profiles);
        assert!(llama_match.is_some());
        let prof = llama_match.unwrap();
        assert!(prof.name.contains("Llama-3.1-8B"));
        assert!(prof.primary_xclbin_path.is_file());

        // Test matching Qwen3-8B
        let qwen_match = auto_match_profile("qwen3-8b-instruct.q4nx", &profiles);
        assert!(qwen_match.is_some());
        assert!(qwen_match.unwrap().name.contains("Qwen3-8B"));
    }

    #[test]
    fn test_interactive_selection_fallback() {
        let dummy_profiles = vec![
            XclbinProfile {
                name: "Profile-A".into(),
                profile_dir: PathBuf::from("/tmp"),
                primary_xclbin_path: PathBuf::from("/tmp/a.xclbin"),
            },
            XclbinProfile {
                name: "Profile-B".into(),
                profile_dir: PathBuf::from("/tmp"),
                primary_xclbin_path: PathBuf::from("/tmp/b.xclbin"),
            },
        ];

        // Simulate user typing "2\n"
        let input = b"2\n";
        let mut output = Vec::new();
        let selected = interactive_select_profile(&dummy_profiles, &input[..], &mut output).unwrap();
        assert_eq!(selected.name, "Profile-B");

        // Simulate invalid input -> default to [1]
        let input_invalid = b"invalid\n";
        let mut output2 = Vec::new();
        let selected_default =
            interactive_select_profile(&dummy_profiles, &input_invalid[..], &mut output2).unwrap();
        assert_eq!(selected_default.name, "Profile-A");
    }

    #[test]
    fn test_dynamic_xclbin_discovery_on_folder_addition() {
        let temp_dir = std::env::temp_dir().join(format!(
            "xclbin_dyn_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::create_dir_all(&temp_dir);

        // Initially no profiles in the new directory
        let initial_profiles = discover_xclbin_profiles(Some(&temp_dir));
        let initial_count = initial_profiles
            .iter()
            .filter(|p| p.profile_dir.starts_with(&temp_dir))
            .count();
        assert_eq!(initial_count, 0);

        // Dynamically add a new profile folder with layer.xclbin
        let new_model_dir = temp_dir.join("Phi-4-Mini");
        fs::create_dir_all(&new_model_dir).unwrap();
        fs::write(new_model_dir.join("layer.xclbin"), b"mock xclbin phi-4").unwrap();

        // Dynamically discover again - should immediately detect Phi-4-Mini
        let updated_profiles = discover_xclbin_profiles(Some(&temp_dir));
        let phi_profile = updated_profiles
            .iter()
            .find(|p| p.name == "Phi-4-Mini");
        assert!(phi_profile.is_some(), "Dynamic folder addition should be detected");
        assert_eq!(
            phi_profile.unwrap().primary_xclbin_path,
            new_model_dir.join("layer.xclbin")
        );

        // Dynamically add a standalone flat xclbin file
        let flat_xclbin = temp_dir.join("Gemma-2-9B.xclbin");
        fs::write(&flat_xclbin, b"mock xclbin gemma-2").unwrap();

        // Discover again - should detect both Phi-4-Mini and Gemma-2-9B
        let updated_profiles2 = discover_xclbin_profiles(Some(&temp_dir));
        let gemma_profile = updated_profiles2
            .iter()
            .find(|p| p.name == "Gemma-2-9B");
        assert!(gemma_profile.is_some(), "Flat xclbin addition should be detected");

        // Test dynamic auto matching on newly added base model
        let matched = auto_match_profile("Phi-4-mini-instruct.q4nx", &updated_profiles2);
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().name, "Phi-4-Mini");

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}

