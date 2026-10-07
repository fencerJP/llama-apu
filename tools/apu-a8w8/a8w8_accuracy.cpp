// llama-apu Phase 10.2b — a8w8 SCHEME accuracy probe.
//
// Validates the full a8w8 numeric scheme end to end:
//   weights   : per-row symmetric int8 quantization (per output channel m)
//   activations: per-row symmetric int8 quantization (per token n)
//   GEMM      : int8 x int8 -> int32 (hipBLASLt, gfx1150 I8I8_HPA)
//   epilogue  : C_f32[m,n] = C_int32[m,n] * scale_w[m] * scale_x[n]
// compared against an f32 reference product.
//
// Build:
//   clang++ -x hip --offload-arch=gfx1150 -O3 -std=c++17 -w \
//     -I/opt/rocm/core-10.1/include -L/opt/rocm/core-10.1/lib \
//     -lhipblaslt -lamdhip64 -o /tmp/a8w8_accuracy tools/apu-a8w8/a8w8_accuracy.cpp

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CHK(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "hipblaslt err %d @%d\n", (int) s_, __LINE__); exit(1); } } while (0)
#define HCHK(x) do { hipError_t s_ = (x); if (s_ != hipSuccess) { \
    fprintf(stderr, "hip err %d (%s) @%d\n", (int) s_, hipGetErrorString(s_), __LINE__); exit(1); } } while (0)

int main(int argc, char ** argv) {
    int64_t M = argc > 1 ? atoll(argv[1]) : 2048;   // output features (rows of C)
    int64_t N = argc > 2 ? atoll(argv[2]) : 64;     // tokens (cols of C)
    int64_t K = argc > 3 ? atoll(argv[3]) : 2048;   // in features

    printf("a8w8 scheme probe: M=%lld N=%lld K=%lld\n", (long long) M, (long long) N, (long long) K);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> W((size_t) M*K), X((size_t) N*K);
    for (auto & v : W) v = nd(rng);
    for (auto & v : X) v = nd(rng);

    // per-row symmetric int8 quantization
    std::vector<int8_t> qA((size_t) K*M);   // A stored k x m (col-major)
    std::vector<float>  sw(M);
    for (int64_t m = 0; m < M; ++m) {
        float mx = 0.0f;
        for (int64_t k = 0; k < K; ++k) mx = std::max(mx, std::fabs(W[m*K + k]));
        float s = mx > 0 ? mx / 127.0f : 1.0f;
        sw[m] = s;
        for (int64_t k = 0; k < K; ++k) {
            long q = lroundf(W[m*K + k] / s);
            q = std::max(-128L, std::min(127L, q));
            qA[k + m*K] = (int8_t) q;
        }
    }
    std::vector<int8_t> qB((size_t) K*N);   // B stored k x n (col-major)
    std::vector<float>  sx(N);
    for (int64_t n = 0; n < N; ++n) {
        float mx = 0.0f;
        for (int64_t k = 0; k < K; ++k) mx = std::max(mx, std::fabs(X[n*K + k]));
        float s = mx > 0 ? mx / 127.0f : 1.0f;
        sx[n] = s;
        for (int64_t k = 0; k < K; ++k) {
            long q = lroundf(X[n*K + k] / s);
            q = std::max(-128L, std::min(127L, q));
            qB[k + n*K] = (int8_t) q;
        }
    }

    // device buffers
    int8_t *dA=nullptr, *dB=nullptr; int32_t *dC=nullptr;
    HCHK(hipMalloc(&dA, qA.size())); HCHK(hipMalloc(&dB, qB.size()));
    HCHK(hipMalloc(&dC, (size_t)M*N*sizeof(int32_t)));
    HCHK(hipMemcpy(dA, qA.data(), qA.size(), hipMemcpyHostToDevice));
    HCHK(hipMemcpy(dB, qB.data(), qB.size(), hipMemcpyHostToDevice));
    HCHK(hipMemset(dC, 0, (size_t)M*N*sizeof(int32_t)));

    hipblasLtHandle_t handle; CHK(hipblasLtCreate(&handle));
    hipblasLtMatmulDesc_t desc;
    CHK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I, HIP_R_32I));
    hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
    CHK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    hipblasLtMatrixLayout_t ad, bd, cd;
    CHK(hipblasLtMatrixLayoutCreate(&ad, HIP_R_8I,   (uint64_t)K, (uint64_t)M, K));
    CHK(hipblasLtMatrixLayoutCreate(&bd, HIP_R_8I,   (uint64_t)K, (uint64_t)N, K));
    CHK(hipblasLtMatrixLayoutCreate(&cd, HIP_R_32I,  (uint64_t)M, (uint64_t)N, M));
    hipblasLtMatmulPreference_t pref; CHK(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t max_ws = 64ull*1024*1024;
    CHK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)));
    hipblasLtMatmulHeuristicResult_t heur; int returned = 0;
    hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(handle, desc, ad, bd, cd, cd, pref, 1, &heur, &returned);
    if (hs != HIPBLAS_STATUS_SUCCESS || returned < 1) { fprintf(stderr, "no a8w8 solution\n"); return 2; }
    void * ws = nullptr; if (heur.workspaceSize) HCHK(hipMalloc(&ws, heur.workspaceSize));
    hipStream_t stream; HCHK(hipStreamCreate(&stream));

    int32_t alpha = 1, beta = 0;
    CHK(hipblasLtMatmul(handle, desc, &alpha, dA, ad, dB, bd, &beta, dC, cd, dC, cd, &heur.algo, ws, heur.workspaceSize, stream));
    HCHK(hipStreamSynchronize(stream));
    std::vector<int32_t> hC((size_t)M*N);
    HCHK(hipMemcpy(hC.data(), dC, hC.size()*sizeof(int32_t), hipMemcpyDeviceToHost));

    // dequant + compare vs f32 reference
    double max_abs = 0.0, sum_sq = 0.0; double ref_sq = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            double got = (double) hC[m + n*M] * (double) sw[m] * (double) sx[n];
            double ref = 0.0;
            for (int64_t k = 0; k < K; ++k) ref += (double) W[m*K + k] * (double) X[n*K + k];
            double e = std::fabs(got - ref);
            max_abs = std::max(max_abs, e);
            sum_sq += e*e; ref_sq += ref*ref;
        }
    }
    double rms = std::sqrt(sum_sq / (double)(M*N));
    double rms_ref = std::sqrt(ref_sq / (double)(M*N));
    printf("dequantized a8w8 vs f32 reference: max_abs_err=%.4f  rms_err=%.4f  rms_ref=%.4f  rel_rms=%.4f%%\n",
           max_abs, rms, rms_ref, 100.0*rms/(rms_ref > 0 ? rms_ref : 1.0));

    if (ws) hipFree(ws);
    hipStreamDestroy(stream);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(ad); hipblasLtMatrixLayoutDestroy(bd); hipblasLtMatrixLayoutDestroy(cd);
    hipblasLtMatmulDescDestroy(desc); hipblasLtDestroy(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
