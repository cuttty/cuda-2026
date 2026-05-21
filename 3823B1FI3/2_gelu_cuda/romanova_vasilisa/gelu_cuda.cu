#include "gelu_cuda.h"

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#define ALPHA -1.59576912f
#define BETTA -0.07135482f


__global__ void gelu_kernel(const float4* __restrict__ in, float4* __restrict__ out, int size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float4 input = in[i];
        float4 result;
        
        float z_x = input.x*(ALPHA+BETTA*input.x*input.x);
        result.x = input.x/(1.0+expf(z_x));

        float z_y = input.y*(ALPHA+BETTA*input.y*input.y);
        result.y = input.y/(1.0+expf(z_y));

        float z_z = input.z*(ALPHA+BETTA*input.z*input.z);
        result.z = input.z/(1.0+expf(z_z));

        float z_w = input.w*(ALPHA+BETTA*input.w*input.w);
        result.w = input.w/(1.0+expf(z_w));

        out[i] = result;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t size = input.size();
    const size_t bytes = size * sizeof(float);
    std::vector<float> output(size);

    float *gpu_input;
    float *gpu_output;

    cudaMalloc(&gpu_input, bytes);
    cudaMalloc(&gpu_output, bytes);

    cudaMemcpy(gpu_input, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    int vec_size = size/4;
    gelu_kernel<<<num_blocks, block_size>>>((const float4*)gpu_input, (float4*)gpu_output, vec_size);

    cudaMemcpy(output.data(), gpu_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_input);
    cudaFree(gpu_output);

    return output;
}
