#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static size_t cap = 0;
static cudaStream_t stream_ = nullptr;
static cublasHandle_t handle_ = nullptr;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    if (stream_ == nullptr) {
        cudaStreamCreate(&stream_);
        cublasCreate(&handle_);
        cublasSetStream(handle_, stream_);
    }

    if (cap < bytes) {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cap = bytes;
    }

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream_);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha, d_b, n,
                d_a, n, &beta, d_c, n);

    std::vector<float> res(n * n);
    cudaMemcpyAsync(res.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return res;
}