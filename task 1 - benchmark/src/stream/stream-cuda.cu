#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#include "../util.h"
#include "stream-util.h"

__global__ 
void stream(size_t nx, const double *__restrict__ src, double *__restrict__ dest) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < nx; i += stride)
        dest[i] = src[i] + 1;
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt, threadsPerBlock, numberOfBlocks, size;
    int deviceId, numberOfSMs;
    double* src;
    double* dest;

    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size = nx * sizeof(double);
    cudaError_t r = cudaMallocManaged(&src, size);
    cudaError_t r2 = cudaMallocManaged(&dest, size);

    if (r != cudaSuccess) {
            fprintf(stderr, "CUDA Error on %s\n", cudaGetErrorString(r));
            exit(0);
    }

    cudaMemPrefetchAsync(src, size, deviceId);
    cudaMemPrefetchAsync(dest, size, deviceId);

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    // init
    initStream(src, nx);

    // warm-up
    for (int i = 0; i < nItWarmUp; ++i) {
        stream<<<numberOfBlocks, threadsPerBlock>>>(nx, src, dest);
        // cudaDeviceSynchronize();
        std::swap(src, dest);
    }
    cudaDeviceSynchronize();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i) {
        stream<<<numberOfBlocks, threadsPerBlock>>>(nx, src, dest);
        // cudaDeviceSynchronize();
        std::swap(src, dest);
        // cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // check solution
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    cudaFree(src);
    cudaFree(dest);

    return 0;
}
