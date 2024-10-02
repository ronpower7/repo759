#include "scan.h"
#include <vector>

std::vector<float> special_scan (std::vector<float>& input, int n) {
    
    std::vector<float> output(n);

    output[0] = input[0];

    for (int i = 1 ; i <= n ;++i) {
        output[i] = output[i-1] + input[i];
    }
    return output;
 }   
