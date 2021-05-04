#ifndef ND_GPU_ARRAY_H
#define ND_GPU_ARRAY_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"

typedef struct {
    PyObject_HEAD
    PyObject *base_array;
    /* Type-specific fields */
    int mem_loc;
} PyGPUArrayObject;

static void
PyGPUArray_dealloc(PyGPUArrayObject *self)
{
    Py_XDECREF(self->base_array);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
PyGPUArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUArrayObject *self;
    self = (PyGPUArrayObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        // Initialize any data members here
    }
    return (PyObject *) self;
}

static int
PyGPUArray_init(PyGPUArrayObject *self, PyObject *args, PyObject *kwds)
{
    //if(PyArray_Type.tp_init((PyObject *)self, args, kwds) < 0) {
    //    return -1;
    //}
    PyObject *any;
    if (!PyArg_ParseTuple(args, "O", &any)) {
        return -1;
    }
    self->base_array = any;
    Py_INCREF(any);
    self->mem_loc = 0;
    return 0;
}

static PyObject *
PyGPUArray_to_array(PyGPUArrayObject *self, void *closure)
{
    return self->base_array;
}

static PyMemberDef PyGPUArray_members[] = {
    {NULL}
};

static PyMethodDef PyGPUArray_methods[] = {
    {"__array__", (PyCFunction) PyGPUArray_to_array, METH_NOARGS,
     "Return an array representation of a GPUArray"
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