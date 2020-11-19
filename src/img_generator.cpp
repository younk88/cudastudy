#include "img_generator.h"
#include <cuda_runtime.h>
#include "generate_img.h"
#include "util.h"

void generate_bgr_1d(cv::Mat *mat) {
    unsigned char *data = NULL;
    int size = mat->cols * mat->rows * 3 * sizeof(unsigned char);
    printf("img data size = %d\n", size);
    CHECK(cudaMalloc((void **)&data, size));
    c_generate_bgr_1d_by_thread_schedule(data, mat->cols, mat->rows);
    cudaDeviceSynchronize();
    unsigned char *hdata = (unsigned char *)malloc(size);
    cudaMemcpy(mat->data, data, size, cudaMemcpyDeviceToHost);
    // mat->data = hdata;
    cudaFree(data);
}