#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("avx2,fma")

#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n*n, 0.0f);

    const float* __restrict__ A = a.data();
    const float* __restrict__ B = b.data();
    float* __restrict__ C = c.data();


    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        float* cRow = C + i*n;
        float* aRow = A + i*n;
        for(int k = 0; k < n; k++){
            float aVal = *(aRow + k);
            float* bRow = B + k*n;
            #pragma omp simd
            for(int j = 0; j < n; j++){
                float bVal = *(bRow + j);
                *(cRow + j) += aVal*bVal;
            }
        }
    }
    return c;
}