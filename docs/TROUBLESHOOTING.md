# Troubleshooting & Driver Setup: AMD Ryzen AI APUs

This guide addresses common hardware, driver, and permission issues when deploying `llama-apu` on AMD Ryzen AI APUs.

---

## 1. Permission Denied Errors (`EACCES` on `/dev/accel/accel0` or `/dev/dri/renderD128`)

### Symptom:
```
Error: Failed to open AMD XDNA NPU node /dev/accel/accel0: Permission denied (os error 13)
```

### Solution:
1. Ensure your user account belongs to the `render` and `video` groups:
   ```bash
   sudo usermod -aG render,video $USER
   ```
2. Install the udev rules to guarantee proper device permissions:
   ```bash
   sudo cp scripts/99-amdxdna-apu.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
3. Log out and log back in, then verify with `apu-doctor`:
   ```bash
   apu-doctor
   ```

---

## 2. Missing AMD XDNA 2 Kernel Driver Module (`amdxdna.ko`)

### Symptom:
```
/dev/accel/accel0 node not found.
```

### Solution:
1. Verify kernel version is $\ge 6.10$:
   ```bash
   uname -r
   ```
2. Check if the driver module is loaded:
   ```bash
   lsmod | grep amdxdna
   ```
3. If not loaded, load the module:
   ```bash
   sudo modprobe amdxdna
   ```
4. On systems without upstream XDNA 2 in-tree drivers, ensure the AMD XRT driver package is installed.

---

## 3. Strix Halo 256-Bit UMA Hugepage Allocation Warning

### Symptom:
On AMD Ryzen AI Max+ 395 (40 CUs), large KV caches may benefit from 2MB hugepage memory tuning.

### Solution:
Enable transparent hugepages or allocate explicit hugepages in `/etc/sysctl.conf`:
```bash
sudo sysctl -w vm.nr_hugepages=1024
```
Launch `llama-cli` with `--hugepages` or pass `-k` speculative decoding flags.

---

## 4. Sub-4-Bit Quantization Rejection

### Symptom:
```
Error: Non-recommended sub-4-bit quantization format (IQ2_XXS) is unsupported.
```

### Solution:
AMD XDNA 2 AIE2P tiles execute fixed 4-bit INT4 / BlockFP16 matrix operations. Use `Q4_0`, `Q4_K_M`, `IQ4_NL`, `Q5_K_M`, `Q8_0`, or `BF16`. Use `apu-model info <file.gguf>` to inspect model format before inference.
