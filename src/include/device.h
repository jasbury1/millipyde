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
    int prev_device_id;
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
    "millipyde.Device",                                 /*tp_name*/
    sizeof(PyDeviceObject),                             /*tp_basicsize*/
    0,                                                  /*tp_itemsize*/
    /* methods */
    (destructor) PyDevice_dealloc,                      /*tp_dealloc*/
    0,                                                  /*tp_print*/
    0,                                                  /*tp_getattr*/
    0,                                                  /*tp_setattr*/
    0,                                                  /*tp_compare*/
    0,                                                  /*tp_repr*/
    0,                                                  /*tp_as_number*/
    0,                                                  /*tp_as_sequence*/
    0,                                                  /*tp_as_mapping*/
    0,                                                  /*tp_hash*/
    0,                                                  /*tp_call*/
    0,                                                  /*tp_str*/
    0,                                                  /*tp_getattro*/
    0,                                                  /*tp_setattro*/
    0,                                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                                 /*tp_flags*/
    __DEVICE_DOC,                                       /*tp_doc*/
    0,                                                  /*tp_traverse*/
    0,                                                  /*tp_clear*/
    0,                                                  /*tp_richcompare*/
    0,                                                  /*tp_weaklistoffset*/
    0,                                                  /*tp_iter*/
    0,                                                  /*tp_iternext*/
    PyDevice_methods,                                   /*tp_methods*/
    0,                                                  /*tp_members*/
    0,                                                  /*tp_getset*/
    0,                                                  /*tp_base*/
    0,                                                  /*tp_dict*/
    0,                                                  /*tp_descr_get*/
    0,                                                  /*tp_descr_set*/
    0,                                                  /*tp_dictoffset*/
    (initproc) PyDevice_init,                           /*tp_init*/
    0,                                                  /*tp_alloc*/
    PyDevice_new,                                       /*tp_new*/
    0,                                                  /*tp_free*/
    0,                                                  /*tp_is_gc*/
};



#endif // MP_DEVICE_H