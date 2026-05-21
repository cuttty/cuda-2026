// naive_gemm_cuda.cu
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_block = blockIdx.x * blockDim.x + threadIdx.x;
    int col_start = col_block * 4;

    if (row >= n || col_start >= n) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k = 0; k < n; ++k) {
        float a_val = A[row * n + k];

        const float4* b_ptr = reinterpret_cast<const float4*>(&B[k * n + col_start]);
        float4 b_vec = *b_ptr;

        sum.x += a_val * b_vec.x;
        sum.y += a_val * b_vec.y;
        sum.z += a_val * b_vec.z;
        sum.w += a_val * b_vec.w;
    }

    reinterpret_cast<float4*>(&C[row * n + col_start])[0] = sum;
}

static float* dev_A = nullptr;
static float* dev_B = nullptr;
static float* dev_C = nullptr;
static size_t alloc_bytes = 0;
static cudaStream_t stream = nullptr;

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);

    if (alloc_bytes < bytes) {
        if (dev_A) {
            cudaFree(dev_A);
            cudaFree(dev_B);
            cudaFree(dev_C);
        }
        cudaMalloc(&dev_A, bytes);
        cudaMalloc(&dev_B, bytes);
        cudaMalloc(&dev_C, bytes);
        alloc_bytes = bytes;
    }

    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }

    cudaMemcpyAsync(dev_A, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_B, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    const int THREADS_X = 16;
    const int THREADS_Y = 16;
    dim3 threads(THREADS_X, THREADS_Y);

    int grid_x = ((n / 4) + THREADS_X - 1) / THREADS_X;
    int grid_y = (n + THREADS_Y - 1) / THREADS_Y;
    dim3 grid(grid_x, grid_y);

    gemm_naive_kernel<<<grid, threads, 0, stream>>>(dev_A, dev_B, dev_C, n);

    std::vector<float> result(n * n);
    cudaMemcpyAsync(result.data(), dev_C, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    return result;
}