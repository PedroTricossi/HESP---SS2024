#pragma warning(disable: 4146) // Disable warning C4146
#include "julia.cuh"
#include <cuda_runtime.h>
#define cimg_display 0
#include "CImg.h"
#include<X11/Xlib.h>
#include <iostream>

using namespace cimg_library;



int main() {
    unsigned char* d_imageData = computeJuliaSetHost();

    if (d_imageData == nullptr) {
        std::cerr << "Error: Failed to compute Julia set." << std::endl;
        return 1;
    }

    CImg<unsigned char> image(d_imageData, WIDTH, HEIGHT, 1, 3, false);

    // Save the image
    const char* filename = "JuliaSet.bmp";
    image.save_bmp("juliaSet.bmp");

   std::cout << "Image saved as " << filename << std::endl;

    // Free allocated memory
    cudaFree(d_imageData);

    return 0;
}
