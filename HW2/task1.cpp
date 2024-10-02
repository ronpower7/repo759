#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <chrono>
#include "scan.h"

void generateRandomArray(std::vector<float>& array , int n) {
    for (int i = 0; i < n; ++i) {
        array[i] = static_cast<float>(rand()) /static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number of elements>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    std::vector<float> input(n);

    std::srand(static_cast<unsigned int>(std::time(0)));
    generateRandomArray(input, n);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> scannedArray = special_scan (input, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken by special scan function: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "First element of special  scanned array: " << scannedArray[0] << std::endl;
    std::cout << "Last element of special  scanned array: " << scannedArray.back() << std::endl;


    return 0;
}
