#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#include <CImg.h>
#include <cuda/std/complex>

using namespace cimg_library;
using complex_t = cuda::std::complex<double>;

const int WIDTH = 800;
const int HEIGHT = 800;
const int MAX_ITER = 128;
const double THRESHOLD = 10.0;


// CUDA kernel to compute Julia set
__global__ void computeJuliaSet(unsigned char* imageData) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    complex_t Julia_C(-0.8, 0.156); // Julia set constant

    if (x < WIDTH && y < HEIGHT) {
        double real = (double)x / WIDTH * 4 - 2;
        double imag = (double)y / HEIGHT * 4 - 2;
        cuda::std::complex<double> z0(real, imag);
        cuda::std::complex<double> z = z0;
         int iterations = 0;
        while (abs(z) <= THRESHOLD && iterations < MAX_ITER) {
            z = z * z + Julia_C;
            iterations++;
        }
        unsigned char color = (iterations == MAX_ITER) ? 0 : (unsigned char)(255 * iterations / MAX_ITER);
        int index = (y * WIDTH + x) * 3;
        imageData[index] = color;       // Red channel
        imageData[index + 1] = 0;       // Green channel
        imageData[index + 2] = 0;       // Blue channel
    }
}

int main() {
    CImg<unsigned char> image(WIDTH, HEIGHT, 1, 3, 255);

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    unsigned char* d_imageData;
    int size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);

    cudaError_t syncError;
    syncError = cudaMallocManaged(&d_imageData, size);
    if (syncError != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(syncError));
        
    }

    int numberOfBlocks = 32 * prop.multiProcessorCount; // Declare and initialize numberOfBlocks
    int numberOfThreads = 256; // Declare and initialize numberOfThreads

    computeJuliaSet<<<numberOfBlocks, numberOfThreads>>>(d_imageData);

    cudaError_t err = cudaGetLastError(); // Declare and initialize err
    if (err != cudaSuccess) {
     printf("Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaFree(d_imageData);

    const char* filename = "JuliaSet.bmp";
    image.save_bmp(filename);
    std::cout << "Image saved as " << filename << std::endl;
   

    return 0;
}
