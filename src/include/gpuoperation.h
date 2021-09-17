#ifndef MP_GPU_OPERATION_H
#define MP_GPU_OPERATION_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"
#include "millipyde.h"

/*******************************************************************************
* DOCUMENTATION
*******************************************************************************/

#define __GPUOPERATION_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUOPERATION_RUN_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUOPERATION_RUN_ON_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

/*******************************************************************************
* STRUCTS
*******************************************************************************/

typedef struct {
    PyObject_HEAD
    PyObject *callable;
    PyObject *arg_tuple;
    MPBool requires_instance;
    double probability;
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
PyGPUOperation_run_on(PyGPUOperationObject *self, PyObject *instance);


/*******************************************************************************
* TYPE DATA
*******************************************************************************/


static PyMemberDef PyGPUOperation_members[] = {
    {NULL}
};

static PyMethodDef PyGPUOperation_methods[] = {
    {"run", (PyCFunction)PyGPUOperation_run, METH_NOARGS,
     __GPUOPERATION_RUN_DOC},
    {"run_on", (PyCFunction)PyGPUOperation_run_on, METH_O,
     __GPUOPERATION_RUN_ON_DOC},
    {NULL}};

static PyTypeObject PyGPUOperation_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "millipyde.Operation",
    .tp_basicsize = sizeof(PyGPUOperationObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyGPUOperation_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = __GPUOPERATION_DOC,
    .tp_methods = PyGPUOperation_methods,
    .tp_members = PyGPUOperation_members,
    .tp_init = (initproc)PyGPUOperation_init,
    .tp_new = PyGPUOperation_new,
};

#endif // MP_GPU_OPERATION_H