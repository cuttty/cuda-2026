#include "block_gemm_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define BLOCK_SIZE 32


__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float result = 0.0f;

    for (int tile_idx = 0; tile_idx < n; tile_idx += BLOCK_SIZE) {
        tile_a[threadIdx.y][threadIdx.x] = a[row * n + tile_idx + threadIdx.x];
        tile_b[threadIdx.y][threadIdx.x] = b[(tile_idx + threadIdx.y) * n + col];
        
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            result += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    c[row * n + col] = result;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);
    size_t matrix_size_bytes = n * n * sizeof(float);

    float* device_a;
    float* device_b;
    float* device_c;

    cudaMalloc(&device_a, matrix_size_bytes);
    cudaMalloc(&device_b, matrix_size_bytes);
    cudaMalloc(&device_c, matrix_size_bytes);

    cudaMemcpy(device_a, a.data(), matrix_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), matrix_size_bytes, cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    BlockGemmKernel<<<grid_dim, block_dim>>>(device_a, device_b, device_c, n);

    cudaMemcpy(c.data(), device_c, matrix_size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return c;
}