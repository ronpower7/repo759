#include "convolution.h"
#include <algorithm>

std::vector<std::vector<float>> convolve(const std::vector<std::vector<float>>& image, const std::vector<std::vector<float>>& mask) {
    int n = image.size();
    int m = mask.size();
    int offset = (m - 1) / 2;
    std::vector<std::vector<float>> result(n, std::vector<float>(n, 0));

    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            float sum = 0.0f;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    int xi = x + i - offset;
                    int yj = y + j - offset;
                    float value = 0.0f;

                    if (xi >= 0 && xi < n && yj >= 0 && yj < n) {
                        value = image[xi][yj];
                    } else if ((xi >= 0 && xi < n) || (yj >= 0 && yj < n)) {
                        value = 1.0f;
                    }

                    sum += mask[i][j] * value;
                }
            }
            result[x][y] = sum;
        }
    }

    return result;
}
