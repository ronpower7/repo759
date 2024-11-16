#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include "matmul.cuh"

void genMatrix(float* matrix, size_t n) {
   std::random_device rd;  // a seed source for the random number engine
   std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
   std::uniform_real_distribution<> distribution_a(-1., 1.);
    for (size_t i = 0; i < n * n; ++i) {
        matrix[i] = distribution_a(gen);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n threads_per_block\n";
        return 1;
    }

    size_t n = std::stoi(argv[1]);
    unsigned int threads_per_block = std::stoi(argv[2]);

    // Allocate host memory
    float *h_A = new float[n * n];
    float *h_B = new float[n * n];
    float *h_C = new float[n * n];

    // Fill matrices A and B with random values
    genMatrix(h_A, n);
    genMatrix(h_B, n);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, n * n * sizeof(float));
    cudaMalloc((void**) &d_B, n * n * sizeof(float));
    cudaMalloc((void**) &d_C, n * n * sizeof(float));

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Measure the time taken by the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the matrix multiplication function
    matmul(d_A, d_B, d_C, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result matrix C back to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element of matrix C
    std::cout << h_C[n * n - 1] << "\n";

    // Calculate and print the elapsed time
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << elapsedTime << " ms\n";

    // Cleanup created event and matrices
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
