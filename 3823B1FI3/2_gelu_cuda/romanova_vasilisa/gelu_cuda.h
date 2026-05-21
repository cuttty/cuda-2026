#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#define ALPHA -1.59576912f
#define BETTA -0.07135482f

#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H

