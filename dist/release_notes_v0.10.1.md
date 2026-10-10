# llama-apu 0.10.1: Upstream Sync (`b11541`) & Model Architecture Expansion

`llama-apu` 0.10.1 incorporates the latest upstream changes from `ggml-org/llama.cpp` master (`2bbca8f20`, tag `b11541`, 107 commits) while preserving all AMD APU co-designed optimizations (KFD dynamic 75% memory governor, Sarathi-Serve stall-free scheduling, PRIME DMA-BUF, DRM syncobj timeline fences, ROCm 10.1). Built and validated on **AMD Ryzen AI APU (Strix Point / Radeon 890M gfx1150)**.

---

### Highlights in Release 0.10.1

- **Upstream Sync (`b11541` / `@2bbca8f20`)**:
  - Merged 107 commits from upstream master.
  - Native **K2 Horizon** dense and MoVA (Mixture of Value-Attention) support (`#29535`).
  - **PLaMo-3** tokenizer pre-segmentation and FIM token handling (`#30045`).
  - Upstream web UI models manager overhaul (`#29583`, `#30228`).
  - Server defaults aligned (port `9931`, context checkpoints preserved).
- **APU Architectural Protection**:
  - All APU co-designed acceleration modules remain active and untouched:
    - Sparse MoE chunk loader with KFD dynamic 75% memory scaling.
    - Sarathi-Serve stall-free scheduling (active by default).
    - Dynamic Q4_0 KV cache and Tile DMA heuristics.
    - Zero-copy PRIME DMA-BUF unified memory handoff (`PASS: host memcpy: 0`).
    - DRM syncobj timeline fence synchronization (`0.91 µs` avg latency).
- **Build & Quality Fixes**:
  - Resolved ROCm 10.1 Clang HIP header search path under `/opt/rocm/core-10.1`.
  - Deduplicated `n_value_expert` / `n_value_expert_used` definitions in `llama-hparams.h`.
  - Reapplied defensive null checks in `tools/imatrix/imatrix.cpp`.

---

### Verification & Inference Testing

- **APU Unit Test Suite**: 100% pass across `test-apu-moe`, `test-apu-sync`, `test-apu-lowquant`, `test-apu-sarathi`, `test-apu-spec`, `test-apu-convert`, and `test-apu-xclbin`.
- **End-to-End Small Model Inference**:
  - **Llama-3.2-1B-Instruct-Q4_K_M**: Prompt 695.7 t/s | Generation 34.0 t/s | Zero-copy handoff: PASS (host memcpy: 0).
  - **Qwen3.5-0.8B-Q4_K_M**: Prompt 201.1 t/s | Generation 48.0 t/s | Zero-copy handoff: PASS (host memcpy: 0).

---

### Installation & Quick Start

```bash
# Extract release tarball
tar -xzf llama-apu-0.10.1-apu-linux-x86_64.tar.gz
cd llama-apu-0.10.1-apu-linux-x86_64

# System diagnostic
./bin/llama-apu-cli apu-doctor

# Run a model (stall-free scheduling is on by default)
./bin/llama-cli -m model.gguf -p "Hello!"

# Optional installer (copies binaries, registers udev rules + systemd unit)
sudo ./install.sh
```

### Verified SHA-256 Checksums
See `llama-apu-0.10.1-apu-linux-x86_64.tar.gz.sha256` attached below.
