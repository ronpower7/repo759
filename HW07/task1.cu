#include <iostream>
#include <cuda_runtime.h>
#include "matmul.cuh"

// Function to initialize matrices with random values
void initialize_matrix(int *matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = rand() % 100;
    }
}

void initialize_matrix(float *matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void initialize_matrix(double *matrix, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix dimension n> <block dimension block_dim>" << std::endl;
        return EXIT_FAILURE;
    }

    unsigned int n = std::stoi(argv[1]);
    unsigned int block_dim = std::stoi(argv[2]);

    // Allocate host memory
    int *A_int = new int[n * n];
    int *B_int = new int[n * n];
    int *C_int = new int[n * n];

    float *A_float = new float[n * n];
    float *B_float = new float[n * n];
    float *C_float = new float[n * n];

    double *A_double = new double[n * n];
    double *B_double = new double[n * n];
    double *C_double = new double[n * n];

    // Initialize matrices
    initialize_matrix(A_int, n);
    initialize_matrix(B_int, n);
    initialize_matrix(A_float, n);
    initialize_matrix(B_float, n);
    initialize_matrix(A_double, n);
    initialize_matrix(B_double, n);

    // Call matmul_1
    matmul_1(A_int, B_int, C_int, n, block_dim);
    std::cout << "matmul_1: First element: " << C_int[0] << ", Last element: " << C_int[n * n - 1] << std::endl;

    // Call matmul_2
    matmul_2(A_float, B_float, C_float, n, block_dim);
    std::cout << "matmul_2: First element: " << C_float[0] << ", Last element: " << C_float[n * n - 1] << std::endl;

    // Call matmul_3
    matmul_3(A_double, B_double, C_double, n, block_dim);
    std::cout << "matmul_3: First element: " << C_double[0] << ", Last element: " << C_double[n * n - 1] << std::endl;

    // Clean up
    delete[] A_int;
    delete[] B_int;
    delete[] C_int;
    delete[] A_float;
    delete[] B_float;
    delete[] C_float;
    delete[] A_double;
    delete[] B_double;
    delete[] C_double;

    return 0;
}
