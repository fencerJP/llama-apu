#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Download TokenRhythm/NeoHorse-1-4B safetensors model to NAS.
Target directory: /mnt/Media/Downloads/model_testing/NeoHorse-1-4B
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

target_dir = "/mnt/Media/Downloads/model_testing/NeoHorse-1-4B"
os.makedirs(target_dir, exist_ok=True)

print(f"[*] Downloading TokenRhythm/NeoHorse-1-4B to NAS: {target_dir}")
t0 = time.time()

try:
    path = snapshot_download(
        repo_id="TokenRhythm/NeoHorse-1-4B",
        local_dir=target_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        max_workers=4,
    )
    dt = time.time() - t0
    print(f"[+] Download complete in {dt:.1f} s to: {path}")
    
    # List files and sizes
    for root, _, files in os.walk(target_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            print(f"    {f:<35} : {sz:,} bytes ({sz/(1024*1024):.2f} MB)")

except Exception as e:
    print(f"[!] Download failed: {e}")
    sys.exit(1)
