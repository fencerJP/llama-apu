# Acknowledgements and Research Foundations

The development of **llama-apu** stands on the shoulders of open-source engineering teams, academic research groups, and pioneering projects across heterogeneous computing, low-bit model quantization, and compiler design.

---

## 1. Core Projects & Open-Source Inspirations

### `atomic-gern/guanaco`
- **Inspiration:** Pioneering work in exploring unified APU routing policies and zero-copy memory transfers between host CPU, integrated GPU, and early NPU accelerators. Their experiments demonstrated the viability of pipelining generation phases across heterogeneous compute blocks.

### `fastflowLM` & `FLM_Q4NX_Converter`
- **Inspiration:** Architected the companion `.q4nx` container specification, two-phase bounded metadata reading, and streaming out-of-core tensor quantization. Their container designs inspired the memory-isolated sidecar format that enables zero-copy loading on constrained edge APU hardware.

### `llama.cpp` & `ggml` Teams
- **Foundational Engine:** Georgi Gerganov and the entire `llama.cpp` and `ggml` developer community. Their relentless dedication to dependency-free, high-performance C/C++ inference made edge LLM execution a global reality.

### System, Hardware & Library Teams
- **AMD XRT & AMDXDNA Kernel Driver Teams:** For developing the upstream Linux kernel accelerator driver (`amdxdna`) and runtime user-space APIs (`libxrt_coreutil.so.2`) enabling direct user-space interaction with Ryzen AI XDNA 2 hardware.
- **AMD ROCm / HIP Engineering Teams:** For developing `ggml-hip` and enabling high-performance matrix multiplication on RDNA 3.5 (gfx1150 / gfx1151) integrated GPUs.
- **Linux Kernel DRM Subsystem Contributors:** For establishing PRIME dma-buf buffer sharing and DRM syncobj timeline synchronization ioctls that make sub-microsecond accelerator coordination possible.
- **Hugging Face Team:** For the `transformers` and `datasets` ecosystems that powered our out-of-core token streaming and multi-domain calibration datasets.

---

## 2. Research Publications & Algorithmic Foundations

The mathematical algorithms, transformation matrices, and hardware lowerings implemented in `llama-apu` are derived from the following foundational scientific papers:

### 1-Bit & Ternary LLM Architectures
- **BitNet: Scaling 1-bit Transformers for Large Language Models (2023)**
  * *Authors:* Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huishuai Zhang, Tong Wu, Peiyu Wang, Jaebum Yoo, Yutao Sun, Yi Zhu, Furu Wei (Microsoft Research)
  * *Contribution:* Introduced ternary linear layers $(-1, 0, +1)$ with activation quantization.
- **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58) (2024)**
  * *Authors:* Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft Research)
  * *Contribution:* Formalized ternary weight representations $T \in \{-1, 0, +1\}$ with per-tensor absolute mean scaling.
- **bitnet.cpp: Efficient Edge Inference for Ternary LLMs (2025)**
  * *Authors:* Jinheng Wang, Hansong Zhou, Ting Song, Shijie Cao, Yan Xia, Ting Cao, Jianyu Wei, Shuming Ma, Hongyu Wang, Furu Wei (Microsoft Research)
  * *Contribution:* Fast ternary lookup table (TL) kernels and $I2\_S$ 2-bit packing formats.
- **T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge (2025)**
  * *Authors:* Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, Mao Yang (Microsoft Research / EuroSys)
  * *Contribution:* Bit-serial table lookups replacing floating-point multiply-accumulate operations on CPU SIMD units.

### Orthogonal Rotations & Outlier Elimination
- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (2024)**
  * *Authors:* Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler (ETH Zurich / Microsoft Research)
  * *Contribution:* Randomized Fast Walsh-Hadamard Transform (FWHT) multiplying weight and activation tensors to eliminate cross-channel outliers.
- **SpinQuant: LLM Quantization with Learned Rotations (2024)**
  * *Authors:* Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra (Meta AI)
  * *Contribution:* Cayley-transform optimization of orthogonal rotation matrices for minimal quantization degradation.
- **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (2024)**
  * *Authors:* Albert Tseng, Jerry Chee, Qinghao Hu, Christopher De Sa (Cornell University)
  * *Contribution:* Proved theoretical incoherence bounds using randomized Hadamard transforms.
- **OffQ: Taming Structured Outliers in LLM Quantization by Offsetting (2026)**
  * *Authors:* Haoqi Wang, Lorenz K. Mueller, Jiawei Zhuang, Mathieu Salzmann, Lukas Cavigelli (EPFL / Sony)
  * *Contribution:* Top-1 PCA outlier channel rotation absorbed into hardware offset parameters.

### Post-Training Quantization (PTQ) & Scale Distillation
- **BiLLM: Pushing the Limit of Post-Training Quantization for LLMs (2024)**
  * *Authors:* Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, Xiaojuan Qi (HKUST / ETH Zurich)
  * *Contribution:* Structural selection of salient weights and residual binarization for sub-2-bit inference.
- **CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs (2026)**
  * *Authors:* Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Anbang Yao (Intel Labs China / ICML)
  * *Contribution:* Learnable Modulation (LM) and Softened Ternarization (ST) requiring only 512 calibration samples for 1B–235B models.
- **ScaleQ-1.58 / AYOT: Attend to Your Own Thoughts (2026)**
  * *Authors:* Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Anbang Yao (Intel Labs China)
  * *Contribution:* CoT calibration sequence generation preventing reasoning degradation during post-training ternarization.
- **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs (2023)**
  * *Authors:* Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han (MIT / NVIDIA)
  * *Contribution:* Mathematical migration of activation outlier difficulty into weight tensors via per-channel scaling.
- **SliceGPT: Compress Large Language Models by Deleting Rows and Columns (2024)**
  * *Authors:* Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman (Microsoft Research)
  * *Contribution:* Computational invariance transformations enabling zero-overhead structural weight reduction.

### Hardware Acceleration & Lookup Table Kernels
- **T-ACE: Fast, Accurate-aware and Cost-Efficient Accelerator for Ternary LLM (2026)**
  * *Authors:* Wonseok Jung, Junseok Kang, Sangwon Shin, Hongjun Um, Jangho Lim, Yongjun Park, Gunjae Koo, Sangwoo Park, Taeweon Suh (Korea University / ICS)
  * *Contribution:* Hardware co-packed 16-byte Tile DMA blocks (64 ternary weights + power-of-two shift exponent metadata) directly mapped into AMD XDNA 2 AIE2P vector processing elements.
- **FLUTE: Fast Matrix Multiplications for Lookup Table-Quantized LLMs (2024)**
  * *Authors:* Han Guo, William Brandon, Radostin Cholakov, Jonathan Ragan-Kelley, Eric P. Xing, Yoon Kim (MIT / CMU)
  * *Contribution:* Stream-K GPU lookup table kernel architecture for non-uniform quantization.
- **LUT-GEMM: Quantized Matrix Multiplication based on LUTs (2024)**
  * *Authors:* Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, Dongsoo Lee (Seoul National University / ICLR)
  * *Contribution:* Elimination of dequantization overhead in sub-4-bit GEMM via precomputed activation tables.
- **SpTMM: Multiplying Ternary Matrix Without Multiplication for Transformers (2025)**
  * *Authors:* Yushi Ogiwara, Hideyuki Kawashima (Keio University)
  * *Contribution:* Multiplication-free ternary execution combined with weight-map sparsity indexing.
- **FAT: An In-Memory Accelerator with Fast Addition for Ternary Weight Neural Networks (2023)**
  * *Authors:* Shien Zhu, Luan H.K. Duong, Hui Chen, Di Liu, Weichen Liu (NTU Singapore)
  * *Contribution:* Zero-weight skipping addition trees for ternary networks.
- **CUTIE: Completely Unrolled Ternary Inference Engine (2022)**
  * *Authors:* M. Scherer, G. Rutishauser, L. Cavigelli, L. Benini (ETH Zurich / Huawei)
  * *Contribution:* Structural sparsity exploitation in unrolled ternary accelerators.
- **HyQuant: Hybrid-Precision Quantization for LLM Attention (2026)**
  * *Authors:* Jiatong Ding et al. (Shanghai Jiao Tong University / EMNLP)
  * *Contribution:* Hybrid-precision attention keeping high-attention tokens in full precision while compressing the remaining KV cache.
- **PrismML 1-bit Bonsai 8B (2026)**
  * *Authors:* PrismML (Caltech IP)
  * *Contribution:* End-to-end 1-bit deployment on consumer silicon without floating-point fallbacks.
