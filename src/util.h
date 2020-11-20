#ifndef UTIL_H
#define UTIL_H

#pragma once
#include <cuda_runtime.h>

void printDeviceProp(cudaDeviceProp dprop);
void initDevice(int idx);
void assume_1d_grid_by_block(dim3 *grid, dim3 block, int size);
void assume_1d_thread_grid(dim3 *grid, dim3 *block, int size);

#define CHECK(err) printf("err:%d\n", err);

#define CHECK_RESULT(err) \
    { \
        cudaError_t e = err; \
        if (e != cudaSuccess) { \
            printf("%s failed: %d %s\n", #err, e, cudaGetErrorString(e)); \
        } \
    }

#define CHECK_ERROR(expr) \
    expr; \
    { \
        cudaError_t e = cudaGetLastError(); \
        if (e != cudaSuccess) { \
            printf("%s failed: %d %s\n", #expr, e, cudaGetErrorString(e)); \
        } \
    }

#endif // UTIL_H