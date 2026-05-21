#include "gelu_omp.h"
#include <vector>
#include <chrono>
#include <iostream>

int main() {
    // ...
    std::vector<float> input;
    // Warming-up
    GeluOMP(input);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    
    std::cout << time;

    return 0;
}