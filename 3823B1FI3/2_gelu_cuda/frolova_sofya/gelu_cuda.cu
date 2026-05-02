#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];
    const float alpha = 0.7978845608f;  // sqrt(2/pi)
    float tmp = alpha * (x + 0.044715f * x * x * x);
    output[idx] = 0.5f * x * (1.0f + tanhf(tmp));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t N = input.size();
    std::vector<float> output(N);
    if (N == 0) return output;

    // Делаем память pinned для быстрых копирований
    cudaHostRegister((void*)input.data(), N * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(output.data(), N * sizeof(float), cudaHostRegisterDefault);

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Копирование на GPU
    cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Запуск ядра
    const int block = 512;
    const int grid = (N + block - 1) / block;
    gelu_kernel<<<grid, block>>>(d_input, d_output, N);

    // Копирование обратно
    cudaMemcpy(output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Очистка
    cudaFree(d_input);
    cudaFree(d_output);
    cudaHostUnregister((void*)input.data());
    cudaHostUnregister(output.data());

    return output;
}