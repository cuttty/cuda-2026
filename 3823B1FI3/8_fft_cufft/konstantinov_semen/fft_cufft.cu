#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

static cufftComplex* devData = nullptr;
static size_t devBytes = 0;
static cufftHandle fwdPlan = 0, invPlan = 0;
static int prevN = 0, prevBatch = 0;
static cudaStream_t stream = nullptr;

__global__ void normKernel(cufftComplex* data, int totalComplex, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalComplex) {
        data[idx].x *= factor;
        data[idx].y *= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalFloat = static_cast<int>(input.size());
    int totalComplex = totalFloat / 2;
    int n = totalComplex / batch;

    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }

    size_t neededBytes = totalFloat * sizeof(float);
    if (devBytes != neededBytes) {
        if (devData) cudaFree(devData);
        cudaMalloc(&devData, neededBytes);
        devBytes = neededBytes;
    }

    if (prevN != n || prevBatch != batch) {
        if (fwdPlan) cufftDestroy(fwdPlan);
        if (invPlan) cufftDestroy(invPlan);
        cufftPlan1d(&fwdPlan, n, CUFFT_C2C, batch);
        cufftPlan1d(&invPlan, n, CUFFT_C2C, batch);
        cufftSetStream(fwdPlan, stream);
        cufftSetStream(invPlan, stream);
        prevN = n;
        prevBatch = batch;
    }

    cudaMemcpyAsync(devData, input.data(), neededBytes, cudaMemcpyHostToDevice, stream);

    cufftExecC2C(fwdPlan, devData, devData, CUFFT_FORWARD);
    cufftExecC2C(invPlan, devData, devData, CUFFT_INVERSE);

    int blockSize = 256;
    int gridSize = (totalComplex + blockSize - 1) / blockSize;
    float scale = 1.0f / static_cast<float>(n);
    normKernel<<<gridSize, blockSize, 0, stream>>>(devData, totalComplex, scale);

    std::vector<float> output(totalFloat);
    cudaMemcpyAsync(output.data(), devData, neededBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return output;
}