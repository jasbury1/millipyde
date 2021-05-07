#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "ndgpuarray.h"

PyObject *
PyGPUArray_add_one(PyGPUArrayObject *self, void *closure)
{
    printf("Adding one...\n");
    return self;
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
    PyObject *any;
    if (!PyArg_ParseTuple(args, "O", &any)) {
        return -1;
    }
    self->base_array = (PyArrayObject *)any;
    Py_INCREF(any);
    self->mem_loc = 0;
    return 0;
}

PyObject *
PyGPUArray_to_array(PyGPUArrayObject *self, void *closure)
{
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


