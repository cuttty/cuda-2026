#include "gelu_ocl.h"
#include <CL/cl.h>

const char* kernelSource = R"(
__kernel void GeluKernel(__global const float* input, __global float* output, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        const float double_sqrt_2_over_pi = 1.59576912f;
        const float coeff = 0.044715f;
        float arg = double_sqrt_2_over_pi * (x + coeff * x * x * x);
        output[i] = x / (1.0f + exp(-arg)); // формула, аппроксимирующая GELU
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    int n = static_cast<int>(input.size());

    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static cl_kernel kernel = nullptr;
    static cl_program program = nullptr;
    static cl_mem d_in = nullptr;
    static cl_mem d_out = nullptr;
    static int allocated_size = 0;

    if (!context) {
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms); 
        std::vector<cl_platform_id> selected_platforms(num_platforms);
        clGetPlatformIDs(num_platforms, selected_platforms.data(), nullptr); 
        
        cl_platform_id platform_id = selected_platforms[platform]; 

        cl_device_id device;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr); 

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr); 
        queue = clCreateCommandQueue(context, device, 0, nullptr); 

       
        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "GeluKernel", nullptr);
    }

  
    if (allocated_size < n) {
        if (d_in)
            clReleaseMemObject(d_in);
        if (d_out)
            clReleaseMemObject(d_out);
        d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
        d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, nullptr);
        allocated_size = n;
    }

    clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, n * sizeof(float), input.data(), 0, nullptr, nullptr); 
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t global_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr); 

    std::vector<float> output(n); 

    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr); 

    return output;
}