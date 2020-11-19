#include "generate_img.h"
#include "util.h"
#include <stdio.h>

__global__ void generate_bgr_1d_by_thread_schedule(unsigned char *data, int width, int height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < width * height) {
        data[3 * idx] = threadIdx.x % 256;
        data[3 * idx + 1] = blockDim.x % 256;
        data[3 * idx + 2] = blockIdx.x % 256;
    }
}

void c_generate_bgr_1d_by_thread_schedule(unsigned char *data, int width, int height) {
    dim3 block(32, 1);
    dim3 grid(32, 1);
    assume_1d_thread_grid(&grid, &block, width * height);
    // assume_1d_grid_by_block(&grid, block, width * height);
    printf("use grid(%d,%d) block(%d, %d)\n", grid.x, grid.y, block.x, block.y);
    generate_bgr_1d_by_thread_schedule<<<grid, block>>>(data, width, height);
}