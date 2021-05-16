#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "ndgpuarray.h"
#include "GPUKernels.h"
#include "millipyde_image.h"
#include "use_numpy.h"

PyObject *
PyGPUArray_add_one(PyGPUArrayObject *self, void *closure)
{
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
    return Py_None;
}

PyObject *
PyGPUArray_color_to_greyscale(PyGPUArrayObject *self, void *closure)
{
    PyObject *array = self->base_array;
    mpimg_color_to_greyscale(array);
    return Py_None;
 }


void
PyGPUArray_dealloc(PyGPUArrayObject *self)
{
    Py_XDECREF(self->base_array);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *
PyGPUArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUArrayObject *self;
    self = (PyGPUArrayObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        // Initialize any data members here
    }
    return (PyObject *) self;
}

int
PyGPUArray_init(PyGPUArrayObject *self, PyObject *args, PyObject *kwds)
{
    //if(PyArray_Type.tp_init((PyObject *)self, args, kwds) < 0) {
    //    return -1;
    //}
    PyObject *array = NULL;

    if (!PyArg_ParseTuple(args, "O", &array)) {
        return -1;
    }

    if(array) {
        Py_INCREF(array);
        self->base_array = array;
    }

    self->mem_loc = 0;
    return 0;
}

PyObject *
PyGPUArray_to_array(PyGPUArrayObject *self, void *closure)
{
    Py_INCREF(self->base_array);
    return (PyObject*)self->base_array;
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


