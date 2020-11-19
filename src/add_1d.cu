#include <cuda_runtime.h>
#include "add_1d.h"
#include "util.h"
#include <stdio.h>

__global__ void add_1d(float * A, float * B, float * C, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

void c_add_1d(float * A, float * B, float * C, int size) {
    dim3 block(32, 1);
    dim3 grid(32, 1);
    assume_1d_thread_grid(&grid, &block, size);
    printf("use grid(%d,%d) block(%d, %d)\n", grid.x, grid.y, block.x, block.y);
    add_1d<<<grid, block>>>(A, B, C, size);
}