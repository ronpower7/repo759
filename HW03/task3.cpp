#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <omp.h>
#include "msort.h"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv << " <array length n> <number of threads t> <threshold ts>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);
    int ts = std::atoi(argv[3]);

    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    // Initialize the array with random floating-point numbers in the range [-1000, 1000]
    std::vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = static_cast<int>(dist(gen));
    }

    // Set the number of threads
    omp_set_num_threads(t);

    // Perform parallel merge sort
    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            msort(arr, 0, n - 1, ts);
        }
    }
    double end_time = omp_get_wtime();

    // Print the first and last elements of the sorted array
    std::cout << arr[0] << std::endl;
    std::cout << arr[n - 1] << std::endl;

    // Print the time taken to run the msort function in milliseconds
    std::cout << "Time taken: " << (end_time - start_time) * 1000 << " milliseconds" << std::endl;

    return 0;
}
