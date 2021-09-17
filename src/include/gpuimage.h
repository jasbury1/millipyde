#ifndef MP_GPU_IMAGE_H
#define MP_GPU_IMAGE_H

#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"
#include "gpuarray.h"


/*******************************************************************************
* DOCUMENTATION
*******************************************************************************/

#define __GPUIMAGE_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUIMAGE_RGB_TO_GREY_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUIMAGE_TRANSPOSE_DOC PyDoc_STR( \
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
     __GPUIMAGE_RGB_TO_GREY_DOC
    },
    {"rgb2grey", (PyCFunction) PyGPUImage_color_to_greyscale, METH_NOARGS,
     __GPUIMAGE_RGB_TO_GREY_DOC
    },
    {"rgba2gray", (PyCFunction) PyGPUImage_color_to_greyscale, METH_NOARGS,
     __GPUIMAGE_RGB_TO_GREY_DOC
    },
    {"rgba2grey", (PyCFunction) PyGPUImage_color_to_greyscale, METH_NOARGS,
     __GPUIMAGE_RGB_TO_GREY_DOC
    },
    {"transpose", (PyCFunction) PyGPUImage_transpose, METH_NOARGS,
     __GPUIMAGE_TRANSPOSE_DOC
    },
    {NULL}
};

static PyTypeObject PyGPUImage_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "millipyde.gpuimage",
    .tp_basicsize = sizeof(PyGPUImageObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = __GPUIMAGE_DOC,
    .tp_methods = PyGPUImage_methods,
    .tp_members = PyGPUImage_members,
    .tp_init = (initproc) PyGPUImage_init,
};



#endif // MP_GPU_IMAGE_H