#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "millipyde_image.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "use_numpy.h"
#include "gpuarray.h"

__global__ void g_color_to_greyscale(unsigned char * rgbImg, double * greyImg, 
        int width, int height, int channels)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int greyOffset = y * width + x;

        // 3 is for the 3 channels in rgb
        int rgbOffset = greyOffset * channels;
        unsigned char r = rgbImg[rgbOffset];
        unsigned char g = rgbImg[rgbOffset + 1];
        unsigned char b = rgbImg[rgbOffset + 2];
        greyImg[greyOffset] = 0.21 * r + 0.71 * g + 0.07 * b;
    }
}

extern "C"{

void mpimg_color_to_greyscale(PyGPUArrayObject *array){
    int channels = array->dims[2];
    int height = array->dims[0];
    int width = array->dims[1];

    unsigned char *d_rgb;
    double *d_grey;

    if (array->device_data != NULL) {
        d_rgb = (unsigned char*)(array->device_data);
    }
    else {
        printf("OOOPS!\n");
        return;
    }

    HIP_CHECK(hipMalloc(&d_grey, (array->nbytes / channels) * sizeof(double)));

    array->nbytes = (array->nbytes / channels) * sizeof(double);
    array->ndims = 2;
    array->type = 12;
    array->dims[2] = (npy_intp)(width * sizeof(double));
    array->dims[3] = (npy_intp)(sizeof(double));

    hipLaunchKernelGGL(g_color_to_greyscale, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(32, 32, 1),
            0,
            0,
            d_rgb,
            d_grey,
            width,
            height,
            channels);

    array->device_data = d_grey;

    HIP_CHECK(hipFree(d_rgb));
}

} // extern "C"