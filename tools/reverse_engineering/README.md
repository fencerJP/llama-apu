# AMD XDNA XCLBIN Reverse-Engineering & Disassembly Suite

This tool suite provides comprehensive binary inspection, section carving, spatial geometry extraction, and PDI microcode disassembly for compiled AMD Ryzen AI NPU `.xclbin` files.

## Features

1. **Section Carving & Extraction (`xclbinutil`)**:
   - Dumps and parses `MEM_TOPOLOGY` (physical memory banks, DRAM/SRAM sizing, base addresses).
   - Dumps and parses `IP_LAYOUT` (DPU kernel IDs, PS kernels).
   - Dumps and parses `CONNECTIVITY` (routes kernel buffer arguments `bo0`..`bo4` to DRAM/SRAM banks).
   - Dumps and parses `EMBEDDED_METADATA` (kernel function C-ABI signature, argument offsets, model hyperparameters).
   - Dumps and parses `AIE_PARTITION` (column width, start columns, operations per cycle, fingerprint hashes).

2. **Deep PDI & CDO Stream Disassembly**:
   - Parses the Versal BootROM sync header (`0x11223344 0x55667788 0x99aabbcc`).
   - Identifies partition headers (`PPDI`, `aie_image`).
   - Decodes the binary CDO (Configuration Data Object) command stream:
     - `WRITE` / `MASK_WRITE` (spatial tile register configurations, tile locks, semaphores).
     - `AIE_STREAM_CONFIG` (inter-tile stream switch crossbar routes).
     - `DMA_XFER` (tile DMA channel transfer descriptor initialization).
     - `MASK_POLL` (hardware synchronization barrier polling).
   - Decodes target AIE tile coordinates (columns, rows, register offsets).
   - Recovers internal symbols and strings.

3. **Structured Markdown Dossier Generation**:
   - Outputs `<xclbin_name>.md` with zero external dependencies beyond standard Python 3.
   - Includes Mermaid diagrams of the kernel-to-memory crossbar.
   - Assesses cross-APU portability (Phoenix/Hawk Point NPU1 vs. Strix/Gorgon/Krackan/Halo NPU2).
   - Provides turnkey deployment steps for `apu-backend` and `.q4nx` container embedding.

---

## Usage

```bash
# Basic run (generates <name>.md alongside input file):
python3 tools/reverse_engineering/xclbin_disassembler.py /path/to/model.xclbin

# Verbose step-by-step diagnostic logging:
python3 tools/reverse_engineering/xclbin_disassembler.py /path/to/model.xclbin --verbose

# Custom output directory:
python3 tools/reverse_engineering/xclbin_disassembler.py /path/to/model.xclbin -o ./reports/ --verbose
```

## Example Outputs

- [`layer.md`](layer.md): Analysis of production Qwen3-8B NPU2 hardware graph.
- [`Spark-X2.5-1.7B-NPU2.md`](Spark-X2.5-1.7B-NPU2.md): Analysis of custom-synthesized Spark-X2.5-1.7B hardware graph.
