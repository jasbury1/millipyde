#ifndef MP_GPU_IMAGE_H
#define MP_GPU_IMAGE_H

#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"

/*******************************************************************************
* STRUCTS
*******************************************************************************/

typedef struct {
    PyGPUArrayObject array;
    int width;
    int height;
} PyGPUImageObject;

/*******************************************************************************
* FUNCTION HEADERS
*******************************************************************************/

int
PyGPUImage_init(PyGPUImageObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUImage_color_to_greyscale(PyGPUImageObject *self, void *closure);

PyObject *
PyGPUImage_transpose(PyGPUImageObject *self, void *closure);

/*******************************************************************************
* TYPE DATA
*******************************************************************************/

static PyMemberDef PyGPUImage_members[] = {
    {NULL}
};

static PyMethodDef PyGPUImage_methods[] = {
    {"rgb2gray", (PyCFunction) PyGPUImage_color_to_greyscale, METH_NOARGS,
     "TODO"
    },
    {"rgb2grey", (PyCFunction) PyGPUImage_color_to_greyscale, METH_NOARGS,
     "TODO"
    },
    {"rgba2gray", (PyCFunction) PyGPUImage_color_to_greyscale, METH_NOARGS,
     "TODO"
    },
    {"rgba2grey", (PyCFunction) PyGPUImage_color_to_greyscale, METH_NOARGS,
     "TODO"
    },
    {"transpose", (PyCFunction) PyGPUImage_transpose, METH_NOARGS,
     "TODO"
    },
    {NULL}
};

static PyTypeObject PyGPUImage_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "millipyde.gpuimage",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(PyGPUImageObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_init = (initproc) PyGPUImage_init,
    .tp_members = PyGPUImage_members,
    .tp_methods = PyGPUImage_methods
};



#endif // ND_GPU_ARRAY_H