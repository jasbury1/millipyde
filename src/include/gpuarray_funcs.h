#ifndef MILLIPYDE_GPUARRAY_FUNCS_H
#define MILLIPYDE_GPUARRAY_FUNCS_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "gpuarray.h"

#define TRANSPOSE_BLOCK_DIM 32

#ifdef __cplusplus
extern "C" {
#endif

//PyObject * gpuarray_transpose(PyObject *array);
void gpuarray_transfer_from_host(PyGPUArrayObject *array, void *data, size_t nbytes);
void *gpuarray_transfer_to_host(PyGPUArrayObject *array);
void gpuarray_dealloc_device_data(PyGPUArrayObject *array);


#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_GPUARRAY_FUNCS_H