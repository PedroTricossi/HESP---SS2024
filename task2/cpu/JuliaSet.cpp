#pragma warning(disable: 4146) // Disable warning C4146
#define cimg_display 0
#include "CImg.h"
#include <cmath>
#include <complex>
#include <iostream>

using namespace cimg_library;

constexpr int WIDTH = 800;
constexpr int HEIGHT = 800;
constexpr int MAX_ITER = 128;
constexpr double THRESHOLD = 10.0;
const std::complex<double> Julia_C(-0.8, 0.2);

int juliaSet(const std::complex<double>& z0) {
    std::complex<double> z = z0;
    int iterations = 0;
    while (std::abs(z) <= THRESHOLD && iterations < MAX_ITER) {
        z = z * z + Julia_C;
        iterations++;
    }
    return iterations;
}

unsigned char getColor(int iterations) {
    if (iterations == MAX_ITER)
        return 0;
    else
        return (unsigned char)(255 * iterations / MAX_ITER);
}

int main() {
    cimg_library::CImg<unsigned char> image(WIDTH, HEIGHT, 1, 3, 255);

    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            double real = (double)x / WIDTH * 4 - 2;
            double imag = (double)y / HEIGHT * 4 - 2;
            std::complex<double> z0(real, imag);
            int iterations = juliaSet(z0);
            unsigned char color = getColor(iterations);
            image(x, y, 0) = color; // Red channel
            image(x, y, 1) = 0;     // Green channel
            image(x, y, 2) = 0;     // Blue channel
        }
    }

    // Save the image to a file in BMP format
    const char* filename = "JuliaSet.bmp";
    image.save_bmp(filename);
    std::cout << "Image saved as " << filename << std::endl;

    return 0;
}

