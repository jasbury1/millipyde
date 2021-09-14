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
        self->requires_instance = MP_FALSE;
    }
    return (PyObject *) self;
    
}


int
PyGPUOperation_init(PyGPUOperationObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *callable = NULL;
    PyObject *call_args;

    Py_ssize_t num_call_args = PyTuple_Size(args);

    if (num_call_args < 1) {
        PyErr_SetString(PyExc_ValueError, 
                "Error constructing gpuarray from argument.");
        return -1;
    }

    callable = PyTuple_GetItem(args, 0);

    // Instance methods are denoted by a string rather than a callable
    if (PyUnicode_Check(callable)) {
        self->requires_instance = MP_TRUE;
    }

    // If we only have the function name, we have no args so create empty tuple
    if (num_call_args == 1) {
        call_args = PyTuple_New(0);
    }
    // Create tuple from all args except the function name
    else {
        call_args  = PyTuple_GetSlice(args, 1, num_call_args);
    }

    Py_INCREF(callable);
    Py_INCREF(call_args);

    self->callable = callable;
    self->arg_tuple = call_args;

    return 0;
}


PyObject *
PyGPUOperation_run(PyGPUOperationObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *result;
    result = PyObject_Call(self->callable, self->arg_tuple, NULL);
    return result;
}


PyObject *
PyGPUOperation_run_on(PyGPUOperationObject *self, PyObject *args, void *closure)
{
    // TODO: Make sure we are an instance method
    PyObject *result;
    PyObject *instance;

    const char *method_name = PyUnicode_AsUTF8(self->callable);


    if (!PyArg_ParseTuple(args, "O", &instance)) {
        // TODO
    }

    Py_INCREF(instance);
    result = PyObject_Call(PyObject_GetAttrString(instance, method_name), self->arg_tuple, NULL);
    Py_DECREF(instance);

    return result;
}
