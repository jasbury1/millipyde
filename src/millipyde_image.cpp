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

/*
 * We use this technique to optimize transposition using shared memory:
 * https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter3/HIP-Coding-3.pdf
 */
template <typename T>
__global__ void g_transpose(T *in_arr, T *out_arr, int width, int height)
{
    // Block dimensions must be square for shared memory intermediate block
    __shared__ T shared_block[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM];
	
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

	if (x < width && y < height) {
		int in_idx = y * width + x;
		shared_block[hipThreadIdx_y][hipThreadIdx_x] = in_arr[in_idx];
	}

	__syncthreads();

    x = hipThreadIdx_x + hipBlockIdx_y * TRANSPOSE_BLOCK_DIM;
    y = hipThreadIdx_y + hipBlockIdx_x * TRANSPOSE_BLOCK_DIM;


	if (x < height && y < width) {
		int out_idx = y * height + x;
		out_arr[out_idx] = shared_block[hipThreadIdx_x][hipThreadIdx_y];
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
        //TODO
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


void mpimg_transpose(PyGPUArrayObject *array)
{
    int height = array->dims[0];
    int width = array->dims[1];

    double *d_img;
    double *d_transpose;

    if (array->device_data != NULL) {
        d_img = (double *)(array->device_data);
    }
    else {
        //TODO
        return;
    }

    HIP_CHECK(hipMalloc(&d_transpose, array->nbytes));
    
    int temp = array->dims[0];
    array->dims[0] = array->dims[1];
    array->dims[1] = temp;
    array->dims[array->ndims] = array->dims[0] * sizeof(double);
    array->dims[array->ndims + 1] = sizeof(double);


    hipLaunchKernelGGL(g_transpose, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM, 1),
            //TODO: Double check this
            TRANSPOSE_BLOCK_DIM * TRANSPOSE_BLOCK_DIM * array->dims[array->ndims + 1],
            0,
            d_img,
            d_transpose,
            width,
            height);


    array->device_data = d_transpose;

    HIP_CHECK(hipFree(d_img));
}


} // extern "C"