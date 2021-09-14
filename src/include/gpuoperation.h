#ifndef MP_GPU_OPERATION_H
#define MP_GPU_OPERATION_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"
#include "millipyde.h"

/*******************************************************************************
* STRUCTS
*******************************************************************************/

typedef struct {
    PyObject_HEAD
    PyObject *callable;
    PyObject *arg_tuple;
    MPBool requires_instance;
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

PyObject *
PyGPUOperation_run(PyGPUOperationObject *self, PyObject *ignored);

PyObject *
PyGPUOperation_run_on(PyGPUOperationObject *self, PyObject *subject, void *closure);


/*******************************************************************************
* TYPE DATA
*******************************************************************************/


static PyMemberDef PyGPUOperation_members[] = {
    {NULL}
};

static PyMethodDef PyGPUOperation_methods[] = {
    {"run", (PyCFunction) PyGPUOperation_run, METH_NOARGS,
    "Run the function represented by the Operation instance"},
    {"run_on", (PyCFunction) PyGPUOperation_run_on, METH_VARARGS,
    "Run the instance method represented by the Operation instance on the object supplied"},
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