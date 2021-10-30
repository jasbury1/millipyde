#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "gpuarray.h"
#include "use_numpy.h"
#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_objects.h"


void
PyGPUArray_dealloc(PyGPUArrayObject *self)
{
    mpobj_dealloc_device_data(self->obj_data);
    PyMem_Free(self->array_data);

    if (self->obj_data != NULL)
    {
        if (self->obj_data->dims != NULL)
        {
            free(self->obj_data->dims);
        }
        free(self->obj_data);
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *
PyGPUArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUArrayObject *self;
    self = (PyGPUArrayObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        self->array_data = NULL;
        self->obj_data = NULL;
    }
    return (PyObject *) self;
}

int
PyGPUArray_init(PyGPUArrayObject *self, PyObject *args, PyObject *kwds)
{
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
                        mperr_str(GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
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
                            mperr_str(GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_ARRAY_TYPE));
            return -1;
        }
    }
    if (!PyArray_ISNUMBER(array))
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUARRAY_ERROR_CONSTRUCTION_WITHOUT_NUMERIC_ARRAY));
        return -1;
    }
    self->obj_data = malloc(sizeof(MPObjData));
    obj_data = self->obj_data;

    /* Set default values to be overwritten later */
    obj_data->device_data = NULL;
    obj_data->type = -1;
    obj_data->nbytes = (size_t)0;
    obj_data->mem_loc = HOST_LOC;
    obj_data->stream = NULL;
    obj_data->pinned = MP_FALSE;

    // Get information from numpy array
    size_t array_nbytes = (size_t)PyArray_NBYTES(array);
    void *array_data = PyArray_DATA(array);
    npy_intp *array_dims = PyArray_DIMS(array);
    npy_intp *array_strides = PyArray_STRIDES(array);

    // Transfer memory from numpy array to our device pointer
    mpobj_copy_from_host(obj_data, array_data, array_nbytes);

    // The current memory location is now set. Safe to now set the stream
    obj_data->stream = mpdev_get_stream(obj_data->mem_loc, 0);

    obj_data->ndims = PyArray_NDIM(array);
    obj_data->type = PyArray_TYPE(array);

    // Allocate space and store dimensions and strides
    obj_data->dims = malloc(obj_data->ndims * 2 * sizeof(int));
    for (int i = 0; i < obj_data->ndims; ++i)
    {
        obj_data->dims[i] = array_dims[i];
    }
    for (int i = 0; i < obj_data->ndims; ++i)
    {
        obj_data->dims[i + obj_data->ndims] = array_strides[i];
    }

    return 0;
}

PyObject *
PyGPUArray_to_array(PyGPUArrayObject *self, void *closure)
{
    MPObjData *obj_data = self->obj_data;
    void *data = mpobj_copy_to_host(obj_data);

    // Fine to allocate outside python heap since lifetime is only this function
    npy_intp *array_dims = malloc(obj_data->ndims * sizeof(npy_intp));
    for (int i = 0; i < obj_data->ndims; ++i)
    {
        array_dims[i] = obj_data->dims[i];
    }

    PyObject *array = PyArray_SimpleNewFromData(obj_data->ndims, array_dims, obj_data->type, data);
    if (array == NULL)
    {
        //TODO: set an error
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject *)array, NPY_ARRAY_OWNDATA);
    Py_INCREF(array);

    free(array_dims);

    return array;
}

PyObject *
PyGPUArray_array_ufunc(PyGPUArrayObject *self, PyObject *arg1, void *closure)
{
    // Should mostly 
    printf("Called __array_ufunc__()\n");
    return PyExc_NotImplementedError;
}

PyObject *
PyGPUArray_array_function(PyGPUArrayObject *self, void *closure)
{
    printf("Called __array_function__()\n");
    return NULL;
}


/*******************************************************************************
 * Create a clone of the given array. All memory will be identical -- a deep
 * copy is performed.
 * 
 * Note that cloning won't necessarily place the new copy on the same device.
 * Specify target device before calling clone to ensure the copy is on a
 * specific device.
 * 
 * @param self The image that will be transposed
 * @param closure An unused closure argument
 * 
 * @return A copy of self
 * 
 ******************************************************************************/
PyObject *
PyGPUArray_clone(PyGPUArrayObject *self, void *closure)
{
    int device_id;
    device_id = mpdev_get_target_device();
    if (device_id == DEVICE_LOC_NO_AFFINITY)
    {
        device_id = mpdev_get_recommended_device();
    }

    return gpuarray_clone(self, device_id, 0);
}


PyObject *
gpuarray_clone(PyGPUArrayObject *self, int device_id, int stream_id)
{
    PyObject *mp_module = PyImport_ImportModule("millipyde");
    PyTypeObject *gpuarray_type =
        (PyTypeObject *)PyObject_GetAttrString(mp_module, "gpuarray");
    PyObject *gpuarray = _PyObject_New(gpuarray_type);
    
    ((PyGPUArrayObject *)gpuarray)->obj_data =
        mpobj_clone_data(self->obj_data, device_id, stream_id);

    Py_INCREF(gpuarray);
    return gpuarray;
}
