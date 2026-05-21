#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void block_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float acc = 0.0f;
    int num_tiles = n / TILE_DIM;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int a_col = tile * TILE_DIM + tx;
        tileA[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;

        int b_row = tile * TILE_DIM + ty;
        tileB[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static size_t prev_bytes = 0;
    static cudaStream_t stream = nullptr;

    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }

    if (prev_bytes != bytes) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        prev_bytes = bytes;
    }

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(n / TILE_DIM, n / TILE_DIM);

    block_gemm_kernel<<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, n);

    std::vector<float> c(n * n);
    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    return c;
}