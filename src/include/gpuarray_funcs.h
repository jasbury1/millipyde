#ifndef MILLIPYDE_GPUARRAY_FUNCS_H
#define MILLIPYDE_GPUARRAY_FUNCS_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define TRANSPOSE_BLOCK_DIM 32

#ifdef __cplusplus
extern "C" {
#endif

PyObject * gpuarray_transpose(PyObject *array);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_GPUARRAY_FUNCS_H