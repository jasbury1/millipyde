#include <stdio.h>
#include <iostream>

#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "gpuarray.h"
#include "gpuarray_funcs.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "use_numpy.h"


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

/*
PyObject * gpuarray_transpose(PyObject *array)
{
    int ndims = PyArray_NDIM(array);
    npy_intp *dims = PyArray_DIMS(array);
    npy_intp *strides = PyArray_STRIDES(array);

    npy_intp height = dims[0];
    npy_intp width = dims[1];
    int channels = dims[2];
    size_t Nbytes = PyArray_NBYTES(array);

    double *d_img;
    double *d_transpose;

    for(int i = 0; i < ndims; ++i) {
        printf("Dim: %d, Stride: %d\n", dims[i],   strides[i]);
    }
    
    HIP_CHECK(hipMalloc(&d_img, Nbytes));
    HIP_CHECK(hipMalloc(&d_transpose, Nbytes));

    HIP_CHECK(hipMemcpy(d_img, PyArray_DATA(array), Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(g_transpose, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM, 1),
            //TODO: Double check this
            TRANSPOSE_BLOCK_DIM * TRANSPOSE_BLOCK_DIM * strides[1],
            0,
            d_img,
            d_transpose,
            width,
            height);

    HIP_CHECK(hipMemcpy(PyArray_DATA(array), d_transpose, Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_img));
    HIP_CHECK(hipFree(d_transpose));

    npy_intp *new_dims = (npy_intp *)malloc(sizeof(npy_intp) * 2 * ndims);
    memcpy(new_dims, dims, sizeof(npy_intp) * 2 * ndims);
    new_dims[1] = dims[0];
    new_dims[0] = dims[1];
    new_dims[ndims] = dims[1] * sizeof(double);
    new_dims[ndims + 1] = sizeof(double);

    PyArray_Dims dim_data;
    dim_data.ptr = new_dims;
    dim_data.len = ndims;

    for(int i = 0; i < ndims; ++i) {
        printf("Dim: %d, Stride: %d\n", new_dims[i],   new_dims[i + ndims]);
    }

    return PyArray_Newshape((PyArrayObject *)array, &dim_data, NPY_CORDER);
    //return PyArray_Flatten((PyArrayObject *)array, NPY_CORDER);
}

void gpuarray_mem_transfer_to_host(PyObject *gpuarray)
{
    
}
*/

void gpuarray_transfer_from_host(PyGPUArrayObject *array, void *data, size_t nbytes) {
    // Free any existing data
    if(array->device_data != NULL) {
        HIP_CHECK(hipFree(array->device_data));
    }
    HIP_CHECK(hipMalloc(&(array->device_data), nbytes));
    HIP_CHECK(hipMemcpy(array->device_data, data, nbytes, hipMemcpyHostToDevice));
    array->nbytes = nbytes;
}

void *gpuarray_transfer_to_host(PyGPUArrayObject *array) {
    void *data = PyMem_Malloc(array->nbytes);
    HIP_CHECK(hipMemcpy(data, array->device_data, array->nbytes, hipMemcpyDeviceToHost));
    return data;
}

void gpuarray_dealloc_device_data(PyGPUArrayObject *array) {
    if(array->device_data != NULL) {
        HIP_CHECK(hipFree(array->device_data));
    }
    array->device_data = NULL; 
}

} // extern "C"