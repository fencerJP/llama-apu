# Inactive / Archived: Scratch Rust llama.cpp Re-implementations

**Status**: INACTIVE & RETIRED (Archived 2026-09-14)

### Background & Rationale:
The files in this directory (`llama_cli.rs`, `llama.rs`, `llama_server.rs`) were an experimental scratch attempt to re-implement standalone `llama-cli` and `llama-server` executables and a minimal transformer engine in pure Rust.

### Why They Are Retired:
1. **Unnecessary Duplication**: Upstream C++ `llama.cpp` already maintains over 50 model family architectures, exact RoPE frequencies, Jinja2 chat templating, GBNF grammar samplers, and battle-tested CLI/server front-ends.
2. **Output Quality**: Re-implementing transformer arithmetic from scratch produced subtle numerical and token layout discrepancies (garbled tokens) compared to upstream's canonical output.
3. **Correct Architecture**: The project architecture has been refocused strictly on being the **hardware acceleration backend (`apu-backend`) in Rust**. Rust provides the unified `dma-buf` zero-copy memory bridge, RDNA 3.5 iGPU prefill, and XDNA 2 NPU decode via the C ABI (`include/apu_backend.h` / `libzero_copy_model_runner.so`), while upstream C++ `llama.cpp` remains the frontend and graph orchestrator.
