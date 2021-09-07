#ifndef MP_GPU_OPERATION_H
#define MP_GPU_OPERATION_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"

/*******************************************************************************
* STRUCTS
*******************************************************************************/

typedef struct {
    PyObject_HEAD
    PyObject *callable;
    PyObject *arg_tuple;
} PyGPUOperationObject;

/*******************************************************************************
* FUNCTION HEADERS
*******************************************************************************/


void
PyGPUOperation_dealloc(PyGPUOperationObject *self);

PyObject *
PyGPUOperation_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

int
PyGPUOperation_init(PyGPUOperationObject *self, PyObject *args, PyObject *kwds);



/*******************************************************************************
* TYPE DATA
*******************************************************************************/


static PyMemberDef PyGPUOperation_members[] = {
    {NULL}
};

static PyMethodDef PyGPUOperation_methods[] = {
    {NULL}
};


static PyTypeObject PyGPUOperation_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "millipyde.Operation",
    .tp_basicsize = sizeof(PyGPUOperationObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyGPUOperation_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Custom objects",
    .tp_methods = PyGPUOperation_methods,
    .tp_members = PyGPUOperation_members,
    .tp_init = (initproc) PyGPUOperation_init,
    .tp_new = PyGPUOperation_new,
};



#endif // MP_GPU_OPERATION_H