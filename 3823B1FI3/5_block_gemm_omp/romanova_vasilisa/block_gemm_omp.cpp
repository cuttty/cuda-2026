#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("avx2,fma")
#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n*n, 0.0f);
    int blocksize = (n < 64 ? n : 64);

    const float* __restrict__ A = a.data();
    const float* __restrict__ B = b.data();
    float* __restrict__ C = c.data();

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i+=blocksize){
        for(int j = 0; j < n; j+=blocksize){
            for(int k = 0; k < n; k+=blocksize){
                int in = std::min(n, i + blocksize);
                int jn = std::min(n, j + blocksize);
                int kn = std::min(n, k + blocksize);

                for(int ii = i; ii < in; ii++){
                    const float* aRow = A + ii*n;
                    float* cRow = C + ii*n;
                    for(int kk = k; kk < kn; kk++){
                        float aVal = *(aRow + kk);
                        const float* bRow = B + kk*n;

                        #pragma omp simd
                        for(int jj = j; jj < jn; jj++){
                            float bVal = *(bRow + jj);
                            *(cRow + jj) += aVal * bVal;
                        }
                    }
                }
            }
        }
    }

    return c;
}