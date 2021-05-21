#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "millipyde_image.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "use_numpy.h"

__global__ void color_to_greyscale_kernel(unsigned char * rgbImg, double * greyImg, 
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

PyObject * mpimg_color_to_greyscale(PyObject *array){
    printf("To Greyscale...\n");

    int ndims = PyArray_NDIM(array);
    printf("Number of dimensions: %d\n", ndims);

    npy_intp *dims = PyArray_DIMS(array);
    npy_intp *strides = PyArray_STRIDES(array);

    int channels = dims[2];

    for(int i = 0; i < ndims; ++i) {
        printf("Dimension: %d, Stride: %d\n", dims[i], strides[i]);
    }

    printf("Array Type: %d\n", PyArray_TYPE(array));
    printf("Number of bytes: %ld\n", PyArray_NBYTES(array));

    npy_intp height = dims[0];
    npy_intp width = dims[1];

    unsigned char *rgbImg_d;
    double *greyImg_d;
    size_t Nbytes = PyArray_NBYTES(array);
    HIP_CHECK(hipMalloc(&rgbImg_d, Nbytes));
    HIP_CHECK(hipMalloc(&greyImg_d, (Nbytes / channels) * sizeof(double)));

    void *greyImg_h = PyArray_malloc((Nbytes / channels) * sizeof(double));

    HIP_CHECK(hipMemcpy(rgbImg_d, PyArray_DATA(array), Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(color_to_greyscale_kernel, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(32, 32, 1),
            0,
            0,
            rgbImg_d,
            greyImg_d,
            width,
            height,
            channels);

    HIP_CHECK(hipMemcpy(greyImg_h, greyImg_d, (Nbytes / channels) * sizeof(double), hipMemcpyDeviceToHost));

    const npy_intp grey_dims[4] = {height, 
                                   width, 
                                   (npy_intp)(width * sizeof(double)), 
                                   (npy_intp)(sizeof(double))};
    PyObject *result = PyArray_SimpleNewFromData(2, grey_dims, NPY_FLOAT64, greyImg_h);
    PyArray_ENABLEFLAGS((PyArrayObject*)result, NPY_ARRAY_OWNDATA);

    Py_INCREF(result);

    HIP_CHECK(hipFree(rgbImg_d));
    HIP_CHECK(hipFree(greyImg_d));

    return result;
}

} // extern "C"