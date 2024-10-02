#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Not proper Usage: " << argv << " <image size> <mask size>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);
    if ( m % 2 == 0) {
        std::cerr << "mask size must be an odd number." << std::endl;
        return 1;
    }

    std::vector<std::vector<float>> image(n, std::vector<float>(n));
    std::vector<std::vector<float>> mask(m, std::vector<float>(m));

   // std::srand(static_cast<unsigned>(std::time(0))); 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            image[i][j] = static_cast<float>(std::rand()) / RAND_MAX * 20.0f - 10.0f;
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            mask[i][j] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> convolvedImage = convolve(image, mask);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken by convolve function: " << duration.count() << " ms" << std::endl;
    std::cout << "First element of convolved array: " << convolvedImage[0][0] << std::endl;
    std::cout << "Last element of convolved array: " << convolvedImage[n-1][n-1] << std::endl;

    return 0;
}
