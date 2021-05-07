#ifndef ND_GPU_ARRAY_H
#define ND_GPU_ARRAY_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <stdio.h>

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"

/*******************************************************************************
* STRUCTS
*******************************************************************************/

typedef struct {
    PyObject_HEAD
    PyArrayObject *base_array;
    /* Type-specific fields */
    int mem_loc;
} PyGPUArrayObject;

/*******************************************************************************
* FUNCTION HEADERS
*******************************************************************************/

void
PyGPUArray_dealloc(PyGPUArrayObject *self);

PyObject *
PyGPUArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

int
PyGPUArray_init(PyGPUArrayObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUArray_to_array(PyGPUArrayObject *self, void *closure);

PyObject *
PyGPUArray_array_ufunc(PyGPUArrayObject *self, PyObject *arg1, void *closure);

PyObject *
PyGPUArray_array_function(PyGPUArrayObject *self, void *closure);

PyObject *
PyGPUArray_add_one(PyGPUArrayObject *self, void *closure);

static PyMemberDef PyGPUArray_members[] = {
    {NULL}
};

/*******************************************************************************
* TYPE DATA
*******************************************************************************/

static PyMethodDef PyGPUArray_methods[] = {
    {"__array__", (PyCFunction) PyGPUArray_to_array, METH_NOARGS,
     "Return an array representation of a GPUArray"
    },
    // {"__array_ufunc__", (PyCFunction) PyGPUArray_array_ufunc, METH_VARARGS,
    //  "TODO"
    // },
    {"__array_function__", (PyCFunction) PyGPUArray_array_function, METH_VARARGS,
     "TODO"
    },
    {"add_one", (PyCFunction) PyGPUArray_add_one, METH_NOARGS,
     "TODO"
    },
    {NULL}
};

static PyTypeObject PyGPUArray_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "millipyde.GPUArray",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(PyGPUArrayObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyGPUArray_new,
    .tp_init = (initproc) PyGPUArray_init,
    .tp_dealloc = (destructor) PyGPUArray_dealloc,
    .tp_members = PyGPUArray_members,
    .tp_methods = PyGPUArray_methods,
};



#endif // ND_GPU_ARRAY_H