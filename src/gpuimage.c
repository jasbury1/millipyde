#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "gpuarray.h"
#include "gpuimage.h"
#include "millipyde_image.h"
#include "millipyde_devices.h"
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

/*******************************************************************************
 * Calls the greyscale kernel on the gpuimage. If the image is not pinned to a
 * specific device, and if the user specified a 'target device', then this
 * function is responsible for moving the gpuimage data over to the target if
 * necessary. This is done before calling the kernel.
 * 
 * @param self The image that will be greyscaled
 * @param closure An unused closure argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_color_to_greyscale(PyGPUImageObject *self, void *closure)
{
    // TODO: Type checking, etc
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // The greyscale function will set the current device based on obj_data
    mpimg_color_to_greyscale(obj_data, NULL);
    return Py_None;
}


/*******************************************************************************
 * Calls the transposition kernel on the gpuimage. If the image is not pinned to 
 * a specific device, and if the user specified a 'target device', then this
 * function is responsible for moving the gpuimage data over to the target if
 * necessary. This is done before calling the kernel.
 * 
 * @param self The image that will be transposed
 * @param closure An unused closure argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_transpose(PyGPUImageObject *self, void *closure)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // TODO: Type checking, etc
    mpimg_transpose(obj_data, NULL);
    return Py_None;
}


/*******************************************************************************
 * Calls the gaussian blur kernel on the gpuimage. If the image is not pinned to 
 * a specific device, and if the user specified a 'target device', then this
 * function is responsible for moving the gpuimage data over to the target if
 * necessary. This is done before calling the kernel.
 * 
 * @param self The image that will be transposed
 * @param closure An unused closure argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUImage_gaussian(PyGPUImageObject *self, void *closure)
{
    MPObjData *obj_data = self->array.obj_data;
    int target_device = mpdev_get_target_device();

    if (!obj_data->pinned)
    {
        // If a target device is specified and it isn't the current location
        if (target_device != DEVICE_LOC_NO_AFFINITY && target_device != obj_data->mem_loc)
        {
            // We need to move our memory to a different device
            mpobj_change_device(obj_data, target_device);
        }
    }

    // TODO: Type checking, etc
    mpimg_gaussian(obj_data, NULL);
    return Py_None;
}