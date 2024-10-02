#include <iostream>
#include <vector>
#include <chrono>
#include "matmul.h" 

void genMatrix1(double* matrix, int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void genMatrix2(std::vector<double>& matrix, int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main(int argc , char* argv[]) {
    int n = 2048; 
    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C1 = new double[n * n];
    double* C2 = new double[n * n];
    double* C3 = new double[n * n];
    std::vector<double> A_vec(n * n);
    std::vector<double> B_vec(n * n);
    std::vector<double> C4(n * n);

    genMatrix1(A, n);
    genMatrix1(B, n);
    genMatrix2(A_vec, n);
    genMatrix2(B_vec, n);

    auto start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C1, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << n << "\n" << duration.count() << "\n" << C1[n * n - 1] << "\n";

    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C2, n);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << duration.count() << "\n" << C2[n * n - 1] << "\n";

    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C3, n);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << duration.count() << "\n" << C3[n * n - 1] << "\n";

    start = std::chrono::high_resolution_clock::now();
    mmul4(A_vec, B_vec, C4, n);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << duration.count() << "\n" << C4[n * n - 1] << "\n";

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}