#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
const char* kGeluKernelSrc = R"(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   const int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  float x = input[idx];
  const float alpha = -1.59576912f;
  const float beta = -0.07135482f;
  float inner = x * (a + beta * x * x);
  output[idx] = x / (1.0f + exp(inner));
}
)";

cl::Context ctx;
cl::CommandQueue queue;
cl::Kernel kernel;
bool initialized = false;
std::mutex init_mutex;

void InitOpenCL(int id) {
  std::lock_guard<std::mutex> lock(init_mutex);
  if (initialized) return;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found");
  if (id < 0 || id >= (int)platforms.size()) id = 0;

  auto platform = platforms[id];
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.empty()) throw std::runtime_error("No GPU devices found");

  ctx = cl::Context(devices);
  queue = cl::CommandQueue(ctx, devices[0]);

  cl::Program program(ctx, kGeluKernelSrc);
  try {
    program.build(devices);
  } catch (...) {
    std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
    std::cerr << "Build log:\n" << log << std::endl;
    throw;
  }

  kernel = cl::Kernel(program, "gelu");
  initialized = true;
}
} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
  if (input.empty()) return {};

  InitOpenCL(platform);
  size_t n = input.size();

  cl::Buffer buf_in(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    n * sizeof(float), (void*)input.data());
  cl::Buffer buf_out(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float));

  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_out);
  kernel.setArg(2, static_cast<int>(n));

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
  queue.finish();

  std::vector<float> result(n);
  queue.enqueueReadBuffer(buf_out, CL_TRUE, 0, n * sizeof(float), result.data());
  return result;
}