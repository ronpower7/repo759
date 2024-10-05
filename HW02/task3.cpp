#include <iostream>
#include <vector>
#include <chrono>
#include "matmul.h" 

void genMatrix(double* matrix1, std::vector<double>& matrix2 , int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix1[i] = static_cast<double>(rand()) / RAND_MAX;
        matrix2[i] = matrix1[i] ;
    }
}

int main(int argc , char* argv[]) {
    int n = 1010; 
    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C1 = new double[n * n];
    double* C2 = new double[n * n];
    double* C3 = new double[n * n];
    std::vector<double> A_vec(n * n);
    std::vector<double> B_vec(n * n);
    std::vector<double> C4(n * n);

    genMatrix(A,A_vec, n);
    genMatrix(B,B_vec, n);

    auto start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C1, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << n << "\n" << duration.count() << "ns \n" << C1[n * n - 1] << "\n";

    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C2, n);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << duration.count() << "ns \n" << C2[n * n - 1] << "\n";

    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C3, n);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << duration.count() << "ns  \n" << C3[n * n - 1] << "\n";

    start = std::chrono::high_resolution_clock::now();
    mmul4(A_vec, B_vec, C4, n);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << duration.count() << "ns \n" << C4[n * n - 1] << "\n";

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;
    A_vec.clear();
    B_vec.clear();
    C4.clear();

    return 0;
}
