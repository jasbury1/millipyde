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


extern "C"{

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