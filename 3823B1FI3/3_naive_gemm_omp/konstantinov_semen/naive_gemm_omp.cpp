#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("avx2,fma")

#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float aik = A[i * n + k];
            float* Ci = C + i * n;
            const float* Bk = B + k * n;
            
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
    
    return c;
}