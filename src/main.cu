#include <cuda_runtime.h>
#include <stdio.h>
#include "util.h"
#include "add_1d.h"
#include "img_generator.h"
#include <opencv2/imgcodecs.hpp>

__global__ void sumMatrix(float * A, float * B, float * C, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * ny;
    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

void initData(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)(i % 64);
    }
}

void printData(float *arr, int size, const char* tag) {
    printf("\n%s", tag);
    int maxs = size > 512 ? 512 : size;
    for (int i = 0; i < maxs; i++) {
        if ((i % 64) == 0) {
            printf("\n");
        }
        printf("%.0f,", arr[i]);
    }
    printf("\n");
}

void printData(unsigned char *arr, int size, const char* tag) {
    printf("\n%s", tag);
    int maxs = size > 512 ? 512 : size;
    for (int i = 0; i < maxs; i++) {
        if ((i % 64) == 0) {
            printf("\n");
        }
        printf("%d,", arr[i]);
    }
    printf("\n");
}

void call_calc() {
    int nx = 1<<12;
    int ny = 1<<12;
    int nBytes = nx * ny * sizeof(float);
    int size = nx * ny;

    float *A_host = (float *)malloc(nBytes);
    float *B_host = (float *)malloc(nBytes);
    // float *C_host = (float *)malloc(nBytes);
    float *C_from_device = (float *)malloc(nBytes);
    initData(A_host, size);
    initData(B_host, size);
    // printData(A_host, size, "===A=======");
    // printData(B_host, size, "===B=======");

    float *A_device = NULL;
    float *B_device = NULL;
    float *C_device = NULL;
    cudaMalloc((void **)&A_device, nBytes);
    cudaMalloc((void **)&B_device, nBytes);
    cudaMalloc((void **)&C_device, nBytes);

    cudaMemcpy(A_device, A_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, nBytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

    // sumMatrix<<<grid, block>>>(A_device, B_device, C_device, nx, ny);
    c_add_1d(A_device, B_device, C_device, size);

    cudaDeviceSynchronize();
    cudaMemcpy(C_from_device, C_device, nBytes, cudaMemcpyDeviceToHost);
    printData(C_from_device, size, "===C=======");

    printf("end\n");

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    free(A_host);
    free(B_host);
}

int main(int argc, char ** argv) {
    printf("starting...\n");
    initDevice(0);

    // call_calc();

    //
    cv::Mat img(256, 256, CV_8UC3);
    generate_bgr_1d(&img);
    printData(img.data, 255, "===img data=====");
    cv::imwrite("/tmp/gen_1d.jpg", img);

    cudaDeviceReset();
    return 0;
}