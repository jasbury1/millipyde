#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "gpuarray.h"
#include "gpuimage.h"
#include "GPUKernels.h"
#include "millipyde_image.h"
#include "gpuarray_funcs.h"
#include "use_numpy.h"

int
PyGPUImage_init(PyGPUImageObject *self, PyObject *args, PyObject *kwds) {
    PyObject *any = NULL;
    PyArrayObject *array = NULL;

    if (!PyArg_ParseTuple(args, "O", &any)) {
        return -1;
    }
    if(!any) {
        PyErr_SetString(PyExc_ValueError, 
                "Error constructing gpuarray from argument.");
        return -1;
    }
    // Typecast to PyArrayObject if it is already of that type
    if(PyArray_Check(any)) {
        array = (PyArrayObject *)any;
    }
    // Attempt to create an array from the type passed to initializer
    else {
        array = (PyArrayObject *)PyArray_FROM_OTF(any, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
        if (array == NULL) {
            PyErr_SetString(PyExc_ValueError, 
                    "Construcing gpuimages requires an ndarray or array compatible argument.");
            return -1;
        }
    }
    if (!PyArray_ISNUMBER(array)) {
        PyErr_SetString(PyExc_ValueError,
                "Construcing gpuimages requires a compatible image format.");
        return -1; 
    }

    int ndims = PyArray_NDIM(array);

    if(ndims != 2 && ndims != 3) {
        PyErr_SetString(PyExc_ValueError,
                "Construcing gpuimages either a 2 dimensional array image " 
                "format for single channel images (greyscale), or a 3 "
                "dimensional array image format for multi-channel images "
                "(rgb/rgba).");
        return -1; 
    }

    if (PyGPUArray_Type.tp_init((PyObject *) self, args, kwds) < 0) {
        return -1;
    }
    self->width = self->array.dims[1];
    self->height = self->array.dims[0];
    return 0;
}

PyObject *
PyGPUImage_color_to_greyscale(PyGPUImageObject *self, void *closure)
{
    // TODO: Type checking, etc
    mpimg_color_to_greyscale(self);
    return Py_None;
}

PyObject *
PyGPUImage_transpose(PyGPUImageObject *self, void *closure)
{
    // TODO: Type checking, etc
    mpimg_transpose(self);
    return Py_None;
}