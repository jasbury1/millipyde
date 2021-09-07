#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "use_numpy.h"
#include "gpuoperation.h"


void
PyGPUOperation_dealloc(PyGPUOperationObject *self)
{
    Py_XDECREF(self->callable);
    Py_XDECREF(self->arg_tuple);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


PyObject *
PyGPUOperation_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUOperationObject *self;
    self = (PyGPUOperationObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        self->callable = NULL;
        self->arg_tuple = NULL;
    }
    return (PyObject *) self;
    
}


int
PyGPUOperation_init(PyGPUOperationObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *callable = NULL;
    PyObject *call_args;

    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_ValueError, 
                "Error constructing gpuarray from argument.");
        return -1;
    }
    if (PyTuple_Size(args) == 1) {
        call_args = PyTuple_New(0);
    }
    else {
        call_args  = PyTuple_GetSlice(args, 1, PyTuple_Size(args));
    }    
    callable = PyTuple_GetItem(args, 0);
    Py_INCREF(callable);
    Py_INCREF(call_args);

    self->callable = callable;
    self->arg_tuple = call_args;

    PyObject_Call(callable, call_args, NULL);

    return 0;
}