#pragma once

// llama-apu Phase 10.1 — optional hipBLASLt routing for large prefill GEMMs.
//
// Enabled at BUILD time by GGML_HIP_USE_HIPBLASLT (ROCm/HIP only; no effect on
// CUDA builds). Enabled at RUN time unless LLAMA_APU_GEMM_BACKEND=rocblas|hipblas.
//
// Only the single-matrix (ne12==1 && ne13==1) case is routed, and only when the
// token dimension (n) reaches LLAMA_APU_HIPBLASLT_MIN_M (default 16), i.e. prefill.
// Any failure returns false so the caller falls back to rocBLAS/hipBLAS unchanged.

#if defined(GGML_USE_HIP) && defined(GGML_HIP_USE_HIPBLASLT)

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace ggml_hipblaslt {

// Disabled by default. Enable at runtime with LLAMA_APU_GEMM_BACKEND=hipblaslt (or "auto").
inline bool enabled() {
    static const bool e = []() {
        const char * v = getenv("LLAMA_APU_GEMM_BACKEND");
        if (v == nullptr) {
            return false;
        }
        return strcmp(v, "hipblaslt") == 0 || strcmp(v, "auto") == 0;
    }();
    return e;
}

inline int64_t min_tokens() {
    static const int64_t v = []() {
        const char * e = getenv("LLAMA_APU_HIPBLASLT_MIN_M");
        return e ? (int64_t) atoll(e) : (int64_t) 16;
    }();
    return v;
}

// Opt-in: force large quantized prefill GEMMs onto the dequant + hipBLASLt path
// instead of MMQ. Default OFF so the stock routing (MMQ) is unchanged.
inline bool prefill_override() {
    static const bool e = []() {
        const char * v = getenv("LLAMA_APU_HIPBLASLT_PREFILL");
        return v != nullptr && (strcmp(v, "1") == 0 || strcmp(v, "on") == 0 || strcmp(v, "true") == 0);
    }();
    return e;
}

inline hipblasLtHandle_t handle() {
    static hipblasLtHandle_t h = nullptr;
    static std::once_flag once;
    std::call_once(once, []() {
        if (hipblasLtCreate(&h) != HIPBLAS_STATUS_SUCCESS) {
            h = nullptr;
        }
    });
    return h;
}

struct Key {
    int64_t m, n, k, lda, ldb, ldc;
    int ta, tb, at, bt, ct;
    bool operator==(const Key & o) const {
        return m == o.m && n == o.n && k == o.k &&
               lda == o.lda && ldb == o.ldb && ldc == o.ldc &&
               ta == o.ta && tb == o.tb && at == o.at && bt == o.bt && ct == o.ct;
    }
};

struct KeyHash {
    size_t operator()(const Key & k) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&h](uint64_t x) { h ^= x; h *= 1099511628211ULL; };
        mix((uint64_t) k.m); mix((uint64_t) k.n); mix((uint64_t) k.k);
        mix((uint64_t) k.lda); mix((uint64_t) k.ldb); mix((uint64_t) k.ldc);
        mix((uint64_t) k.ta); mix((uint64_t) k.tb);
        mix((uint64_t) k.at); mix((uint64_t) k.bt); mix((uint64_t) k.ct);
        return h;
    }
};

struct AlgoEntry {
    hipblasLtMatmulAlgo_t algo;
    size_t ws = 0;
};

inline std::unordered_map<Key, AlgoEntry, KeyHash> & cache() {
    static std::unordered_map<Key, AlgoEntry, KeyHash> c;
    return c;
}
inline std::mutex & mtx() {
    static std::mutex m;
    return m;
}

// Persist a grow-only device workspace; returns nullptr on allocation failure.
inline void * workspace(size_t need, size_t & cap) {
    static void * buf = nullptr;
    static size_t bufsz = 0;
    if (need > bufsz) {
        if (buf != nullptr) {
            (void) hipFree(buf);
            buf = nullptr;
            bufsz = 0;
        }
        if (need == 0) {
            cap = 0;
            return nullptr;
        }
        if (hipMalloc(&buf, need) != hipSuccess) {
            buf = nullptr;
            bufsz = 0;
            cap = 0;
            return nullptr;
        }
        bufsz = need;
    }
    cap = bufsz;
    return buf;
}

// Attempt the GEMM via hipBLASLt. Returns true only on a completed matmul;
// otherwise the caller must run its normal rocBLAS/hipBLAS path.
inline bool try_matmul(const void * A, const void * B, void * D,
                       int64_t m, int64_t n, int64_t k,
                       int64_t lda, int64_t ldb, int64_t ldc,
                       hipDataType at, hipDataType bt, hipDataType ct,
                       hipStream_t stream) {
    if (!enabled()) {
        return false;
    }
    // Route prefill only: the token dimension is the distinguishing factor.
    if (n < min_tokens()) {
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return false;
    }
    hipblasLtHandle_t h = handle();
    if (h == nullptr) {
        return false;
    }

    if (getenv("LLAMA_APU_HIPBLASLT_VERBOSE") != nullptr) {
        static std::once_flag vb;
        std::call_once(vb, [&]() {
            fprintf(stderr, "[llama-apu] hipBLASLt route engaged: m=%lld n=%lld k=%lld\n",
                    (long long) m, (long long) n, (long long) k);
        });
    }

    hipblasLtMatmulDesc_t            desc = nullptr;
    hipblasLtMatrixLayout_t          ad   = nullptr;
    hipblasLtMatrixLayout_t          bd   = nullptr;
    hipblasLtMatrixLayout_t          cd   = nullptr;
    hipblasLtMatmulPreference_t      pref = nullptr;
    bool ok = false;

    do {
        if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) break;

        const hipblasOperation_t opA = HIPBLAS_OP_T;
        const hipblasOperation_t opB = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)) != HIPBLAS_STATUS_SUCCESS) break;

        // Stored (pre-transpose) layouts: A is k x m, B is k x n, C/D is m x n.
        if (hipblasLtMatrixLayoutCreate(&ad, at, (uint64_t) k, (uint64_t) m, lda) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&bd, bt, (uint64_t) k, (uint64_t) n, ldb) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&cd, ct, (uint64_t) m, (uint64_t) n, ldc) != HIPBLAS_STATUS_SUCCESS) break;

        if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) break;
        {
            uint64_t max_ws = (uint64_t) 32 * 1024 * 1024;
            if (hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)) != HIPBLAS_STATUS_SUCCESS) break;
        }

        const Key key = { m, n, k, lda, ldb, ldc, (int) opA, (int) opB, (int) at, (int) bt, (int) ct };

        AlgoEntry entry;
        bool have = false;
        {
            std::lock_guard<std::mutex> lk(mtx());
            auto it = cache().find(key);
            if (it != cache().end()) {
                entry = it->second;
                have  = true;
            }
        }

        if (!have) {
            hipblasLtMatmulHeuristicResult_t heur;
            int returned = 0;
            if (hipblasLtMatmulAlgoGetHeuristic(h, desc, ad, bd, cd, cd, pref, 1, &heur, &returned) != HIPBLAS_STATUS_SUCCESS) break;
            if (returned < 1 || heur.state != HIPBLAS_STATUS_SUCCESS) break;
            entry.algo = heur.algo;
            entry.ws   = heur.workspaceSize;
            std::lock_guard<std::mutex> lk(mtx());
            cache()[key] = entry;
        }

        size_t ws_cap = 0;
        void * ws = nullptr;
        if (entry.ws > 0) {
            ws = workspace(entry.ws, ws_cap);
            if (ws == nullptr) break;
        }

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        if (hipblasLtMatmul(h, desc, &alpha, A, ad, B, bd, &beta, D, cd, D, cd,
                            &entry.algo, ws, ws_cap, stream) != HIPBLAS_STATUS_SUCCESS) {
            break;
        }
        ok = true;
    } while (false);

    if (pref != nullptr) hipblasLtMatmulPreferenceDestroy(pref);
    if (cd   != nullptr) hipblasLtMatrixLayoutDestroy(cd);
    if (bd   != nullptr) hipblasLtMatrixLayoutDestroy(bd);
    if (ad   != nullptr) hipblasLtMatrixLayoutDestroy(ad);
    if (desc != nullptr) hipblasLtMatmulDescDestroy(desc);

    return ok;
}

} // namespace ggml_hipblaslt

#endif // GGML_USE_HIP && GGML_HIP_USE_HIPBLASLT
