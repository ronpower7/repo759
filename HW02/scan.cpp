#include "scan.h"
#include <vector>

void scan (const float *arr, float *output, std::size_t n) {
    

    output[0] = arr[0];

    for (unsigned int i = 1 ; i <= n ;++i) {
        output[i] = output[i-1] + arr[i];
    }
 }   
