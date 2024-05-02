#include "julia.cuh"
#include <iostream>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using complex_t = cuda::std::complex<double>;

__global__ void computeJuliaSet(unsigned char* image)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    complex_t Julia_C(-0.8, 0.2); // Julia set constant
    for (int i = x; i < WIDTH; ++i) {
        for (int j = y; j < HEIGHT; ++j) {
            double real = (double)i / WIDTH * 4 - 2;
            double imag = (double)j / HEIGHT * 4 - 2;
            cuda::std::complex<double> z0(real, imag);
            cuda::std::complex<double> z = z0;
            int iterations = 0;
            while (abs(z) <= THRESHOLD && iterations < MAX_ITER)
             {
                z = z * z + Julia_C;
                iterations++;
             }
        unsigned char color = (iterations == MAX_ITER) ? 0 : (unsigned char)(255 * iterations / MAX_ITER);

        image[(j * WIDTH + i) * 3] = color; // Red channel
        image[(j * WIDTH + i) * 3 + 1] = color; // Green channel
        image[(j * WIDTH + i) * 3 + 2] = color; // Blue channel


        }
    }
}


// Host function to compute Julia set
unsigned char* computeJuliaSetHost() {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    int size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);

    unsigned char* image;
    cudaError_t allocError = cudaMallocManaged(&image, size);

     if (allocError != cudaSuccess) {
        printf("Error allocating memory: %s\n", cudaGetErrorString(allocError));
        return nullptr;
    }

    int numberOfBlocks = 32 * prop.multiProcessorCount; // Declare and initialize numberOfBlocks
    int numberOfThreads = 256; // Declare and initialize numberOfThreads

    computeJuliaSet<<<numberOfBlocks, numberOfThreads>>>(image);

    cudaError_t launchError = cudaGetLastError();

    if (launchError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(launchError));
        cudaFree(image);
        return nullptr;
    }

    cudaDeviceSynchronize();

    return image;
}
