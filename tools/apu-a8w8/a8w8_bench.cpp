// llama-apu Phase 10.2b — a8w8 (int8 x int8 -> int32) hipBLASLt validation & throughput probe.
//
// Purpose: prove the gfx1150 I8I8_HPA kernels work via hipBLASLt and measure a8w8
// GEMM throughput on prefill-shaped problems, as the substrate for an int8 weight path.
//
// Semantics match llama.cpp mul_mat: C[m,n] = sum_k A[k,m] * B[k,n]
//   A: int8, stored k x m (column-major, lda=k)   -> opA = T
//   B: int8, stored k x n (column-major, ldb=k)   -> opB = N
//   C: int32, m x n (column-major, ldc=m)
//
// Build:
//   clang++ -x hip --offload-arch=gfx1150 -O3 -std=c++17 \
//     -I/opt/rocm/core-10.1/include -L/opt/rocm/core-10.1/lib \
//     -lhipblaslt -lamdhip64 -o /tmp/a8w8_bench tools/apu-a8w8/a8w8_bench.cpp

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>

#define CHK(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "hipblaslt error %d at %s:%d\n", (int) s_, __FILE__, __LINE__); exit(1); } } while (0)
#define HCHK(x) do { hipError_t s_ = (x); if (s_ != hipSuccess) { \
    fprintf(stderr, "hip error %d (%s) at %s:%d\n", (int) s_, hipGetErrorString(s_), __FILE__, __LINE__); exit(1); } } while (0)

static void cpu_ref(const int8_t * A, const int8_t * B, int32_t * C, int64_t M, int64_t N, int64_t K) {
    for (int64_t c = 0; c < N; ++c) {
        for (int64_t r = 0; r < M; ++r) {
            int32_t acc = 0;
            for (int64_t kk = 0; kk < K; ++kk) {
                acc += (int32_t) A[kk + r*K] * (int32_t) B[kk + c*K];
            }
            C[r + c*M] = acc;
        }
    }
}

int main(int argc, char ** argv) {
    int64_t M = argc > 1 ? atoll(argv[1]) : 64;
    int64_t N = argc > 2 ? atoll(argv[2]) : 8;
    int64_t K = argc > 3 ? atoll(argv[3]) : 128;
    int     iters = argc > 4 ? atoi(argv[4]) : 20;

    printf("a8w8 probe: M=%lld N=%lld K=%lld iters=%d\n", (long long) M, (long long) N, (long long) K, iters);

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t>  hA((size_t) K*M), hB((size_t) K*N);
    std::vector<int32_t> hC((size_t) M*N), hRef((size_t) M*N);
    for (auto & v : hA) v = (int8_t) dist(rng);
    for (auto & v : hB) v = (int8_t) dist(rng);

    int8_t  * dA = nullptr, * dB = nullptr;
    int32_t * dC = nullptr;
    HCHK(hipMalloc(&dA, hA.size()));
    HCHK(hipMalloc(&dB, hB.size()));
    HCHK(hipMalloc(&dC, hC.size()*sizeof(int32_t)));
    HCHK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    HCHK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
    HCHK(hipMemset(dC, 0, hC.size()*sizeof(int32_t)));

    hipblasLtHandle_t handle;
    CHK(hipblasLtCreate(&handle));

    hipblasLtMatmulDesc_t desc;
    CHK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I, HIP_R_32I));
    hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
    CHK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    hipblasLtMatrixLayout_t ad, bd, cd;
    CHK(hipblasLtMatrixLayoutCreate(&ad, HIP_R_8I, (uint64_t) K, (uint64_t) M, K));
    CHK(hipblasLtMatrixLayoutCreate(&bd, HIP_R_8I, (uint64_t) K, (uint64_t) N, K));
    CHK(hipblasLtMatrixLayoutCreate(&cd, HIP_R_32I, (uint64_t) M, (uint64_t) N, M));

    hipblasLtMatmulPreference_t pref;
    CHK(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t max_ws = 64ull * 1024 * 1024;
    CHK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)));

    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(handle, desc, ad, bd, cd, cd, pref, 1, &heur, &returned);
    if (hs != HIPBLAS_STATUS_SUCCESS || returned < 1) {
        fprintf(stderr, "NO A8W8 SOLUTION for M=%lld N=%lld K=%lld (status=%d, returned=%d)\n",
                (long long) M, (long long) N, (long long) K, (int) hs, returned);
        return 2;
    }
    printf("heuristic solution found; workspace=%zu bytes\n", heur.workspaceSize);

    void * ws = nullptr;
    if (heur.workspaceSize > 0) HCHK(hipMalloc(&ws, heur.workspaceSize));

    int32_t alpha = 1, beta = 0;
    hipStream_t stream;
    HCHK(hipStreamCreate(&stream));

    // --- correctness ---
    CHK(hipblasLtMatmul(handle, desc, &alpha, dA, ad, dB, bd, &beta, dC, cd, dC, cd,
                        &heur.algo, ws, heur.workspaceSize, stream));
    HCHK(hipStreamSynchronize(stream));
    HCHK(hipMemcpy(hC.data(), dC, hC.size()*sizeof(int32_t), hipMemcpyDeviceToHost));
    cpu_ref(hA.data(), hB.data(), hRef.data(), M, N, K);
    int64_t maxerr = 0;
    for (size_t i = 0; i < hC.size(); ++i) {
        int64_t e = (int64_t) hC[i] - (int64_t) hRef[i];
        if (e < 0) e = -e;
        if (e > maxerr) maxerr = e;
    }
    printf("correctness: max |GPU - CPU| = %lld  (%s)\n", (long long) maxerr, maxerr == 0 ? "EXACT int32" : "MISMATCH");

    // --- throughput ---
    // warmup
    for (int i = 0; i < 3; ++i)
        CHK(hipblasLtMatmul(handle, desc, &alpha, dA, ad, dB, bd, &beta, dC, cd, dC, cd, &heur.algo, ws, heur.workspaceSize, stream));
    HCHK(hipStreamSynchronize(stream));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        CHK(hipblasLtMatmul(handle, desc, &alpha, dA, ad, dB, bd, &beta, dC, cd, dC, cd, &heur.algo, ws, heur.workspaceSize, stream));
    HCHK(hipStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ops = 2.0 * (double) M * (double) N * (double) K * iters;   // MACs*2
    printf("time: %.3f ms/iter   throughput: %.1f GOPS (%.1f TOPS)\n",
           sec/iters*1e3, ops/sec/1e9, ops/sec/1e12);

    if (ws) hipFree(ws);
    hipStreamDestroy(stream);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(ad); hipblasLtMatrixLayoutDestroy(bd); hipblasLtMatrixLayoutDestroy(cd);
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
    return maxerr == 0 ? 0 : 3;
}
