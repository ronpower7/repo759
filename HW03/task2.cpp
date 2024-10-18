#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "convolution.h"
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Not proper Usage: " << argv << " <image size> <mask size>" << std::endl;
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    unsigned int t = std::atoi(argv[2]);

    omp_set_num_threads(t);

    std::vector<float> image(n * n);
    std::size_t m = 3 ;
    std::vector<float> mask(m * m);

   // std::srand(static_cast<unsigned>(std::time(0))); 
    for (unsigned int i = 0; i < n * n ; ++i) {
            image[i] = static_cast<float>(std::rand()) / RAND_MAX * 20.0f - 10.0f;
    }

    for (unsigned int i = 0; i < m * m ; ++i) {
            mask[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> convolvedImage(n * n) ;
    convolve(&image[0],&convolvedImage[0],n,&mask[0],m);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Time taken by convolve function: " << duration.count() << " ms" << std::endl;
    std::cout << "First element of convolved array: " << convolvedImage[0] << std::endl;
    std::cout << "Last element of convolved array: " << convolvedImage[(n*n)-1] << std::endl;

    image.clear();
    mask.clear();
    convolvedImage.clear();


    return 0;
}
