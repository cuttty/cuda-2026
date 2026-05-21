#define CL_TARGET_OPENCL_VERSION 300
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <cstring>
#include <iostream>

static const char* kernelSrc = R"(
__kernel void gelu_compute(__global const float* src, __global float* dst, int total) {
    int idx = get_global_id(0);
    if (idx >= total) return;
    float x = src[idx];
    float x3 = x * x * x;
    float inner = x + 0.044715f * x3;
    float z = 1.5957691216f * inner;   // 2 * sqrt(2/pi)
    dst[idx] = x / (1.0f + native_exp(-z));
}
)";

static cl_platform_id plat = nullptr;
static cl_device_id dev = nullptr;
static cl_context ctx = nullptr;
static cl_command_queue queue = nullptr;
static cl_program prog = nullptr;
static cl_kernel kern = nullptr;
static cl_mem devIn = nullptr;
static cl_mem devOut = nullptr;
static size_t prevSize = 0;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t n = input.size();
    if (n == 0) return std::vector<float>();

    size_t bytes = n * sizeof(float);

    if (ctx == nullptr) {
        cl_uint numPlats;
        clGetPlatformIDs(0, nullptr, &numPlats);
        if (numPlats == 0) return std::vector<float>();
        std::vector<cl_platform_id> platforms(numPlats);
        clGetPlatformIDs(numPlats, platforms.data(), nullptr);
        if (platform < 0 || platform >= (int)numPlats) platform = 0;
        plat = platforms[platform];

        cl_uint numDevs;
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevs);
        if (numDevs == 0) return std::vector<float>();
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);

        ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, nullptr);
        if (!ctx) return std::vector<float>();

        cl_queue_properties props[] = {0};
        queue = clCreateCommandQueueWithProperties(ctx, dev, props, nullptr);
        if (!queue) return std::vector<float>();

        prog = clCreateProgramWithSource(ctx, 1, &kernelSrc, nullptr, nullptr);
        if (!prog) return std::vector<float>();

        const char* opts = "-cl-fast-relaxed-math -cl-mad-enable";
        clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
        kern = clCreateKernel(prog, "gelu_compute", nullptr);
        if (!kern) return std::vector<float>();
    }

    if (prevSize != bytes) {
        if (devIn) clReleaseMemObject(devIn);
        if (devOut) clReleaseMemObject(devOut);
        devIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        devOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
        prevSize = bytes;
    }

    cl_event writeEv;
    clEnqueueWriteBuffer(queue, devIn, CL_FALSE, 0, bytes, input.data(), 0, nullptr, &writeEv);

    int n_int = static_cast<int>(n);
    clSetKernelArg(kern, 0, sizeof(cl_mem), &devIn);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &devOut);
    clSetKernelArg(kern, 2, sizeof(int), &n_int);

    size_t local = 256;
    size_t global = ((n + local - 1) / local) * local;
    cl_event kernEv;
    clEnqueueNDRangeKernel(queue, kern, 1, nullptr, &global, &local, 1, &writeEv, &kernEv);

    std::vector<float> result(n);
    cl_event readEv;
    clEnqueueReadBuffer(queue, devOut, CL_FALSE, 0, bytes, result.data(), 1, &kernEv, &readEv);

    clWaitForEvents(1, &readEv);

    clReleaseEvent(writeEv);
    clReleaseEvent(kernEv);
    clReleaseEvent(readEv);

    return result;
}