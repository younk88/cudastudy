#include <stdio.h>
#include "util.h"
#include <math.h>

int _active_device_num;
int _max_threads_per_block = 64;
int _warp_size = 32;

void printDeviceProp(cudaDeviceProp dprop) {
    printf("===gpu device info================================\n");
    printf("name = %s\n", dprop.name);
    printf("totalGlobalMem = %ld\n", dprop.totalGlobalMem);
    printf("totalConstMem = %ld\n", dprop.totalConstMem);
    printf("canMapHostMemory = %ld\n", dprop.canMapHostMemory);
    printf("managedMemory = %ld\n", dprop.managedMemory);
    printf("warpSize = %d\n", dprop.warpSize);

    printf("sharedMemPerMultiprocessor = %d\n", dprop.sharedMemPerMultiprocessor);
    printf("regsPerMultiprocessor = %d\n", dprop.regsPerMultiprocessor);
    printf("multiProcessorCount = %d\n", dprop.multiProcessorCount);
    printf("maxThreadsPerMultiProcessor = %d\n", dprop.maxThreadsPerMultiProcessor);
    printf("maxThreadsPerBlock = %d\n", dprop.maxThreadsPerBlock);
    for (int i = 0; i < 3; i++) {
        printf("maxThreadsDim[%d] = %d\n", i, dprop.maxThreadsDim[i]);
        printf("maxGridSize[%d] = %d\n", i, dprop.maxGridSize[i]);
    }
    printf("======================================\n");
}

void initDevice(int idx) {
    int num;
    cudaGetDeviceCount(&num);
    printf("has %d gpu device\n", num);
    if (idx >= num) {
        return;
    }
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, idx);
    printDeviceProp(dprop);
    cudaSetDevice(idx);
    _active_device_num = idx;
    _max_threads_per_block = dprop.maxThreadsPerBlock;
    _warp_size = dprop.warpSize;
}

void assume_1d_thread_grid(dim3 *grid, dim3 *block, int size) {
    int threadSize = size <= _max_threads_per_block ? size : _max_threads_per_block;
    if ((threadSize % _warp_size) != 0) {
        int exSize = ceil(size / (float)_warp_size) * _warp_size;
        if (exSize > _max_threads_per_block) {
            threadSize = exSize - _warp_size;
        } else {
            threadSize = exSize;
        }
    }
    block->x = threadSize;
    block->y = 1;
    block->z = 1;
    assume_1d_grid_by_block(grid, *block, size);
}

void assume_1d_grid_by_block(dim3 *grid, dim3 block, int size) {
    grid->x = ceil(size / (float)block.x);
    grid->y = 1;
    grid->z = 1;
}