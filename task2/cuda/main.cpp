#include "julia.cuh"
#pragma warning(disable: 4146) // Disable warning C4146
#define cimg_display 0
#include "CImg.h"
#include<X11/Xlib.h>
#include <iostream>

using namespace cimg_library;


int main() {
    CImg<unsigned char> image(WIDTH, HEIGHT, 1, 3, 255);

    unsigned char* d_imageData;

    computeJuliaSetHost(d_imageData);

    image.save_bmp("JuliaSet.bmp");
    
    // std::cout << "Image saved as " << filename << std::endl;
   

    return 0;
}
