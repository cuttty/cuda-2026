#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("avx2,fma")

#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

constexpr int BLOCK_SIZE = 64;

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();
    
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        int i_max = std::min(i0 + BLOCK_SIZE, n);
        
        for (int j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
            int j_max = std::min(j0 + BLOCK_SIZE, n);
            
            for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, n);
                
                for (int i = i0; i < i_max; ++i) {
                    float* Ci = C + i * n;
                    const float* Ai = A + i * n;
                    
                    for (int k = k0; k < k_max; ++k) {
                        float aik = Ai[k];
                        const float* Bk = B + k * n;
                        
                        #pragma omp simd
                        for (int j = j0; j < j_max; ++j) {
                            Ci[j] += aik * Bk[j];
                        }
                    }
                }
            }
        }
    }
    
    return c;
}