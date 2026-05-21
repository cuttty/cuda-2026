#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

const float SQRT_2_PI = 0.7978845608f;   // sqrt(2/pi)
const float K = 0.044715f;               // x^3

// GELU(x) = x / (1 + exp(-2m)),
// m = sqrt(2/pi) * (x + 0.044715 * x^3)
__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float x3 = x * x * x;
        float m = SQRT_2_PI * (x + K * x3);
        out[idx] = x / (1.0f + expf(-2.0f * m));
    }
}

// Основная функция (CPU)
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static int capacity = 0;

    if (capacity < n) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        capacity = n;
    }

    cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(d_in, d_out, n);

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}