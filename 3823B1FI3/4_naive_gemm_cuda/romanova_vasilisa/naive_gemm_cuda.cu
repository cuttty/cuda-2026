#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void NaiveMatrixMultiply(const __restrict__ float* a, const __restrict__ float* b, __restrict__ float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int col4 = col * 4;
    
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    for (int k = 0; k < n; ++k) {
        float a_val = a[row * n + k];
        
        float4 b_vec = reinterpret_cast<const float4*>(&b[k * n + col4])[0];
        
        sum0 = fmaf(a_val, b_vec.x, sum0);
        sum1 = fmaf(a_val, b_vec.y, sum1);
        sum2 = fmaf(a_val, b_vec.z, sum2);
        sum3 = fmaf(a_val, b_vec.w, sum3);
    }
    
    reinterpret_cast<float4*>(&c[row * n + col4])[0] = make_float4(sum0, sum1, sum2, sum3);
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    std::vector<float> c(n * n);
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    
    static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    static size_t allocated_bytes = 0;
    
    if (bytes > allocated_bytes) {
        if (d_a) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated_bytes = bytes;
    }
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);  // 256 потоков на блок
    dim3 numBlocks(n / 4 / 16, n / 16);
    
    NaiveMatrixMultiply<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    return c;
}