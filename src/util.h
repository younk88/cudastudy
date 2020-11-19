#ifndef UTIL_H
#define UTIL_H

#pragma once
#include <cuda_runtime.h>

void printDeviceProp(cudaDeviceProp dprop);
void initDevice(int idx);
void assume_1d_grid_by_block(dim3 *grid, dim3 block, int size);
void assume_1d_thread_grid(dim3 *grid, dim3 *block, int size);

#define CHECK(err) printf("err:%d\n", err);

#endif // UTIL_H