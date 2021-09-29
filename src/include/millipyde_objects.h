#ifndef MILLIPYDE_OBJECTS_H
#define MILLIPYDE_OBJECTS_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "gpuarray.h"
#include "gpuimage.h"
#include "millipyde.h"

#define CHECK_GPU_OBJECT(o) (PyObject_TypeCheck(o, &PyGPUArray_Type) || PyObject_TypeCheck(o, &PyGPUImage_Type))

#ifdef __cplusplus
extern "C"
{
#endif

    //PyObject * gpuarray_transpose(PyObject *array);
    void mpobj_copy_from_host(PyGPUArrayObject *array, void *data, size_t nbytes);
    void *mpobj_copy_to_host(PyGPUArrayObject *array);

    void mpobj_change_device(PyGPUArrayObject *array, int device_id);

    void mpobj_dealloc_device_data(PyGPUArrayObject *array);


#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_OBJECTS_H