#include <iostream>
#include <vector>
#include <chrono>
#include "matmul.h" 

void genMatrix(float* matrix1, unsigned int n) {
    for (std::size_t i = 0; i < n * n; ++i) {
        matrix1[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main(int argc , char* argv[]) {
    unsigned int n ;
    unsigned int t ;

    if (argc != 3) {
        std::cerr << "Usage: " << argv << " <matrix dimension n> <number of threads t>" << std::endl;
        return 1;
    }

    n = std::stoi(argv[1]);
    t = std::stoi(argv[2]);
    omp_set_num_threads(t);

  
    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C1 = new float[n * n];

    genMatrix(A,n);
    genMatrix(B,n);

    auto start = std::chrono::high_resolution_clock::now();
    mmul(A, B, C1, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << C1[0] << "\n" << C1[n * n - 1] << "\n" << duration.count() << "ms"  << "\n";

    delete[] A;
    delete[] B;
    delete[] C1;

    return 0;
}
