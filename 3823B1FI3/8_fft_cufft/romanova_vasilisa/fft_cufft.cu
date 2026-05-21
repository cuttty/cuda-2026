#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void kernel(float* data, int total_size, float normalization_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        data[idx] *= normalization_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int total_size = input.size();
    int transform_size = total_size / (2 * batch);
    int data_bytes = total_size * sizeof(float);

    cufftComplex* device_data;
    cudaMalloc(&device_data, data_bytes);
    cudaMemcpy(device_data, input.data(), data_bytes, cudaMemcpyHostToDevice);

    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, transform_size, CUFFT_C2C, batch);

    cufftExecC2C(fft_plan, device_data, device_data, CUFFT_FORWARD);
    cufftExecC2C(fft_plan, device_data, device_data, CUFFT_INVERSE);

    int threads_per_block = 256;
    int blocks_per_grid = (total_size + threads_per_block - 1) / threads_per_block;
    float normalization_factor = 1.0f / static_cast<float>(transform_size);
    kernel<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<float*>(device_data), total_size, normalization_factor);

    std::vector<float> result(total_size);
    cudaMemcpy(result.data(), device_data, data_bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(fft_plan);
    cudaFree(device_data);

    return result;
}