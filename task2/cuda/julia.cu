#include "julia.cuh"
#include <iostream>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


using complex_t = cuda::std::complex<double>;

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

// Host function to compute Julia set
void computeJuliaSetHost(unsigned char* imageData) {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    int size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);

    cudaError_t syncError;
    syncError = cudaMallocManaged(&imageData, size);
    if (syncError != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(syncError));
        
    }

    int numberOfBlocks = 32 * prop.multiProcessorCount; // Declare and initialize numberOfBlocks
    int numberOfThreads = 256; // Declare and initialize numberOfThreads

    computeJuliaSet<<<numberOfBlocks, numberOfThreads>>>(imageData);

    std::cout  << " color: " << imageData <<std::endl;

    cudaError_t err = cudaGetLastError(); // Declare and initialize err

    if (err != cudaSuccess) {
     printf("Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}