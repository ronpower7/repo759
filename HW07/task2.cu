#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include "reduce.cuh"

// Function to initialize an array with random values
void initialize_array(float *array, unsigned int N) {
    for (unsigned int i = 0; i < N; ++i) {
        array[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random values in range [-1, 1]
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <array length N> <threads per block>" << std::endl;
        return 1;
    }

    unsigned int N = std::stoi(argv[1]);
    unsigned int threads_per_block = std::stoi(argv[2]);

    // Allocate host memory
    float *h_input = new float[N];
    initialize_array(h_input, N);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int num_blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc(&d_output, num_blocks * sizeof(float));

    // Call the reduce function
    reduce(&d_input, &d_output, N, threads_per_block);

    // Copy the result back to host
    float result;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum: " << result << std::endl;

    // Clean up
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
