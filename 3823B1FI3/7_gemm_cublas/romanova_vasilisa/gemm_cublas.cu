#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(n*n);
    size_t bytes = n * n * sizeof(float);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float* device_a;
    float* device_b;
    float* device_c;
    float* device_cT;

    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);
    cudaMalloc(&device_cT, bytes);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, device_a, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, device_b, n);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n,
                &alpha, device_a, n, device_b, n, &beta, device_c, n);

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
                &alpha, device_c, n, &beta, nullptr, n, device_cT, n);

    cublasGetMatrix(n, n, sizeof(float), device_cT, n, c.data(), n);

    cudaFree(device_b);
    cudaFree(device_c);
    cudaFree(device_a);
    cudaFree(device_cT);

    cublasDestroy(handle);

    return c;

}