#ifndef MP_DEVICE_H
#define MP_DEVICE_H

#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"
#include "millipyde.h"

/*******************************************************************************
* DOCUMENTATION
*******************************************************************************/

#define __DEVICE_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __DEVICE_ENTER_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __DEVICE_EXIT_DOC PyDoc_STR( \
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
    int device_id;
} PyDeviceObject;


/*******************************************************************************
* FUNCTION HEADERS
*******************************************************************************/

void
PyDevice_dealloc(PyDeviceObject *self);

PyObject *
PyDevice_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

int
PyDevice_init(PyDeviceObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyDevice_enter(PyDeviceObject *self, void *closure);

PyObject *
PyDevice_exit(PyDeviceObject *self, PyObject *args, PyObject *kwds);

/*******************************************************************************
* TYPE DATA
*******************************************************************************/

/*
static PyMemberDef PyDevice_members[] = {
    {NULL}
};
*/

static PyMethodDef PyDevice_methods[] = {
    {"__enter__", (PyCFunction)PyDevice_enter, METH_NOARGS,
    __DEVICE_ENTER_DOC},
    {"__exit__", (PyCFunction)PyDevice_exit, METH_VARARGS,
    __DEVICE_EXIT_DOC},
    {NULL}};

static PyTypeObject PyDevice_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "millipyde.Device",
    .tp_basicsize = sizeof(PyDeviceObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) PyDevice_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = __DEVICE_DOC,
    .tp_methods = PyDevice_methods,
    .tp_init = (initproc) PyDevice_init,
    .tp_new = PyDevice_new,
};



#endif // MP_DEVICE_H