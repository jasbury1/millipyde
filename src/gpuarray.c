#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "gpuarray.h"
#include "GPUKernels.h"
#include "use_numpy.h"
#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_objects.h"


PyObject *
PyGPUArray_add_one(PyGPUArrayObject *self, void *closure)
{
    /*
    printf("Adding one...\n");
    PyArrayObject *array = self->base_array;
    int dim = PyArray_NDIM(array);
    printf("Dimensions: %d\n", dim);
    int type = PyArray_TYPE(array);
    printf("Type: %d\n", type);
    printf("Type check returned: %d\n", PyArray_ISSIGNED(array));
    printf("Type check returned: %d\n", PyArray_ISINTEGER(array));
    printf("Type check returned: %d\n", PyArray_ISSTRING(array));
    
    int elements = (int)(PyArray_DIM(array, 0));
    printf("elements: %d\n", elements);
    void *data = PyArray_DATA(array);
    printf("Data: %p\n", data);

    int result = add_one(data, elements);
    printf("Result was: %d\n", result);
    */
    return Py_None;
}

void
PyGPUArray_dealloc(PyGPUArrayObject *self)
{
    mpobj_dealloc_device_data(self->capsule);
    PyMem_Free(self->array_data);

    if (self->capsule != NULL)
    {
        if (self->capsule->ndims != NULL)
        {
            free(self->capsule->dims);
        }
        free(self->capsule);
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *
PyGPUArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUArrayObject *self;
    self = (PyGPUArrayObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        self->array_data = NULL;
        self->capsule = NULL;
    }
    return (PyObject *) self;
}

int
PyGPUArray_init(PyGPUArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *any = NULL;
    PyArrayObject *array = NULL;
    GPUCapsule *capsule;

    if (!PyArg_ParseTuple(args, "O", &any))
    {
        return -1;
    }
    if (!any)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
        return -1;
    }
    // Typecast to PyArrayObject if it is already of that type
    if (PyArray_Check(any))
    {
        array = (PyArrayObject *)any;
    }
    // Attempt to create an array from the type passed to initializer
    else
    {
        array = (PyArrayObject *)PyArray_FROM_OTF(any, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
        if (array == NULL)
        {
            PyErr_SetString(PyExc_ValueError,
                            mperr_str(GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
            return -1;
        }
    }
    if (!PyArray_ISNUMBER(array))
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_NUMERIC_ARRAY));
        return -1;
    }
    self->capsule = malloc(sizeof(GPUCapsule));
    capsule = self->capsule;

    /* Set default values to be overwritten later */
    capsule->device_data = NULL;
    capsule->type = -1;
    capsule->nbytes = (size_t)0;
    capsule->mem_loc = HOST_LOC;
    capsule->stream = NULL;
    capsule->pinned = MP_FALSE;

    // Get information from numpy array
    size_t array_nbytes = (size_t)PyArray_NBYTES(array);
    void *array_data = PyArray_DATA(array);
    npy_intp *array_dims = PyArray_DIMS(array);
    npy_intp *array_strides = PyArray_STRIDES(array);

    // Transfer memory from numpy array to our device pointer
    mpobj_copy_from_host(capsule, array_data, array_nbytes);

    // The current memory location is now set. Safe to now set the stream
    capsule->stream = mpdev_get_stream(capsule->mem_loc, 0);

    capsule->ndims = PyArray_NDIM(array);
    capsule->type = PyArray_TYPE(array);

    // Allocate space and store dimensions and strides
    capsule->dims = malloc(capsule->ndims * 2 * sizeof(int));
    for (int i = 0; i < capsule->ndims; ++i)
    {
        capsule->dims[i] = array_dims[i];
    }
    for (int i = 0; i < capsule->ndims; ++i)
    {
        capsule->dims[i + capsule->ndims] = array_strides[i];
    }

    return 0;
}

PyObject *
PyGPUArray_to_array(PyGPUArrayObject *self, void *closure)
{
    GPUCapsule *capsule = self->capsule;
    void *data = mpobj_copy_to_host(capsule);

    // Fine to allocate outside python heap since lifetime is only this function
    npy_intp *array_dims = malloc(capsule->ndims * sizeof(npy_intp));
    for (int i = 0; i < capsule->ndims; ++i)
    {
        array_dims[i] = capsule->dims[i];
    }

    PyObject *array = PyArray_SimpleNewFromData(capsule->ndims, array_dims, capsule->type, data);
    if (array == NULL)
    {
        //TODO: set an error
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)array, NPY_ARRAY_OWNDATA);
    Py_INCREF(array);

    free(array_dims);

    return array;
}

PyObject *
PyGPUArray_array_ufunc(PyGPUArrayObject *self, PyObject *arg1, void *closure)
{
    // Should mostly 
    printf("Called __array_ufunc__()\n");
    return PyExc_NotImplementedError;
}

PyObject *
PyGPUArray_array_function(PyGPUArrayObject *self, void *closure)
{
    printf("Called __array_function__()\n");
    return NULL;
}
