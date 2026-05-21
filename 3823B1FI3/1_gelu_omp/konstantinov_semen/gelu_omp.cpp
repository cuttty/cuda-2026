#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("avx2,fma")

#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <vector>

/*
GELU(x) = 0.5 * x * (1 + tanh(m))
m = sqrt(2/pi) * (x + 0.044715 * x^3)

tanh(m) = (exp(2m) - 1) / (exp(2m) + 1)
1 + tanh(m) = 2 * exp(2m) / (exp(2m) + 1)
GELU(x) = x * exp(2m) / (exp(2m) + 1)
*/

const float SQ2_PI = 0.7978845608f;  // sqrt(2/pi)
const float C = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    
    #pragma omp parallel for schedule(guided) num_threads(omp_get_max_threads())
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = x + C * x_cubed;
        float m = SQ2_PI * inner;
        float exp_2m = std::exp(2.0f * m);
        output[i] = x * exp_2m / (exp_2m + 1.0f);
    }
    
    return output;
}