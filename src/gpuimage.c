#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "gpuarray.h"
#include "gpuimage.h"
#include "GPUKernels.h"
#include "millipyde_image.h"
#include "millipyde_objects.h"
#include "use_numpy.h"
#include "millipyde.h"

int
PyGPUImage_init(PyGPUImageObject *self, PyObject *args, PyObject *kwds) {
    PyObject *any = NULL;
    PyArrayObject *array = NULL;
    MPObjData *obj_data;

    if (!PyArg_ParseTuple(args, "O", &any))
    {
        return -1;
    }
    if (!any)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
        return -1;
    }
    // Typecast to PyArrayObject if it is already of that type
    if (PyArray_Check(any))
    {
        array = (PyArrayObject *)any;
    }
    // Attempt to create an array from the type passed to initializer
    else
    {
        array = (PyArrayObject *)PyArray_FROM_OTF(any, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
        if (array == NULL)
        {
            PyErr_SetString(PyExc_ValueError,
                            mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
            return -1;
        }
    }
    if (!PyArray_ISNUMBER(array))
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_FORMAT));
        return -1;
    }

    int ndims = PyArray_NDIM(array);

    if (ndims != 2 && ndims != 3)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUIMAGE_ERROR_CONSTRUCTION_WITHOUT_IMAGE_DIMS));
        return -1;
    }

    if (PyGPUArray_Type.tp_init((PyObject *)self, args, kwds) < 0)
    {
        return -1;
    }

    obj_data = self->array.obj_data;
    self->width = obj_data->dims[1];
    self->height = obj_data->dims[0];
    return 0;
}

PyObject *
PyGPUImage_color_to_greyscale(PyGPUImageObject *self, void *closure)
{
    // TODO: Type checking, etc
    MPObjData *obj_data = self->array.obj_data;

    mpimg_color_to_greyscale(obj_data);
    return Py_None;
}

PyObject *
PyGPUImage_transpose(PyGPUImageObject *self, void *closure)
{
    MPObjData *obj_data = self->array.obj_data;

    // TODO: Type checking, etc
    mpimg_transpose(obj_data);
    return Py_None;
}