#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx2,fma")

#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);
    float pi = std::acos(-1.0);
    float a = -1.59576912f;
    float b = -0.07135482f;
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++){
        float x = input[i];
        float z = x*(a+b*x*x);
        output[i] = x/(1.0+std::exp(z));
    }

    return output;
}