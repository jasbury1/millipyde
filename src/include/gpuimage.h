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

#define __GPUIMAGE_GAUSSIAN_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUIMAGE_FLIPLR_DOC PyDoc_STR( \
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

PyObject *
PyGPUImage_gaussian(PyGPUImageObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUImage_fliplr(PyGPUImageObject *self, void *closure);

PyObject *
PyGPUImage_rotate(PyGPUImageObject *self, PyObject *args, PyObject *kwds);

void *
gpuimage_rotate_args(PyObject *args);

void *
gpuimage_gaussian_args(PyObject *args);

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
    {"gaussian", (PyCFunction) PyGPUImage_gaussian, METH_VARARGS,
     __GPUIMAGE_GAUSSIAN_DOC
    },
    {"fliplr", (PyCFunction) PyGPUImage_fliplr, METH_NOARGS,
     __GPUIMAGE_FLIPLR_DOC
    },
    {"rotate", (PyCFunction) PyGPUImage_rotate, METH_VARARGS,
     __GPUIMAGE_FLIPLR_DOC
    },
    {NULL}
};

static PyTypeObject PyGPUImage_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "millipyde.gpuimage",                               /*tp_name*/
    sizeof(PyGPUImageObject),                           /*tp_basicsize*/
    0,                                                  /*tp_itemsize*/
    /* methods */
    0,                                                  /*tp_dealloc*/
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,           /*tp_flags*/
    __GPUIMAGE_DOC,                                     /*tp_doc*/
    0,                                                  /*tp_traverse*/
    0,                                                  /*tp_clear*/
    0,                                                  /*tp_richcompare*/
    0,                                                  /*tp_weaklistoffset*/
    0,                                                  /*tp_iter*/
    0,                                                  /*tp_iternext*/
    PyGPUImage_methods,                                 /*tp_methods*/
    PyGPUImage_members,                                 /*tp_members*/
    0,                                                  /*tp_getset*/
    0,                                                  /*tp_base*/
    0,                                                  /*tp_dict*/
    0,                                                  /*tp_descr_get*/
    0,                                                  /*tp_descr_set*/
    0,                                                  /*tp_dictoffset*/
    (initproc) PyGPUImage_init,                         /*tp_init*/
    0,                                                  /*tp_alloc*/
    0,                                                  /*tp_new*/
    0,                                                  /*tp_free*/
    0,                                                  /*tp_is_gc*/
};



#endif // MP_GPU_IMAGE_H