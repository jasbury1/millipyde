/*  Example of wrapping cos function from math.h with the Python-C-API. */

#include <Python.h>
#include <math.h>
#include <stdio.h>

#include "millipyde.h"
#include "millipyde_devices.h"
#include "device.h"
#include "gpuarray.h"
#include "gpuimage.h"
#include "gpuoperation.h"
#include "gpupipeline.h"
#include "gpugenerator.h"

#define INIT_NUMPY_ARRAY_CPP
#include "use_numpy.h"


PyDoc_STRVAR(device_count_doc,
             "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident");

PyDoc_STRVAR(current_device_doc,
             "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident");

PyDoc_STRVAR(best_device_doc,
             "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident");

PyDoc_STRVAR(image_from_path_doc,
             "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident");

PyDoc_STRVAR(images_from_path_doc,
             "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident");



static PyObject *
mpmod_get_device_count(PyObject *self, PyObject *args)
{
    PyObject *result;
    int device_count = mpdev_get_device_count();
    result = Py_BuildValue("i", device_count);
    return result;
}

static PyObject *
mpmod_get_current_device(PyObject *self, PyObject *args)
{
    PyObject *result;
    int device_id = mpdev_get_target_device();
    if (device_id == DEVICE_LOC_NO_AFFINITY)
    {
        device_id = mpdev_get_recommended_device();
    }
    result = Py_BuildValue("i", device_id);
    return result;
}

static PyObject *
mpmod_get_best_device(PyObject *self, PyObject *args)
{
    PyObject *result;
    int device_id = mpdev_get_recommended_device();
    result = Py_BuildValue("i", device_id);
    return result;
}

static PyObject *
mpmod_image_from_path(PyObject *self, PyObject *path)
{
    return gpuimage_single_from_path(path);
}

static PyObject *
mpmod_images_from_path(PyObject *self, PyObject *path)
{
    return gpuimage_all_from_path(path);
}


/*  define functions in module */
static PyMethodDef MillipydeMethods[] =
{
     {"device_count", mpmod_get_device_count, METH_NOARGS, device_count_doc},
     {"get_current_device", mpmod_get_current_device, METH_NOARGS, current_device_doc},
     {"best_device", mpmod_get_best_device, METH_NOARGS, best_device_doc},
     {"image_from_path", mpmod_image_from_path, METH_O, image_from_path_doc},
     {"images_from_path", mpmod_images_from_path, METH_O, images_from_path_doc},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
/* Python version 3*/
static struct PyModuleDef millipydeModule =
{
    PyModuleDef_HEAD_INIT,
    "Millipyde Module", 
    "Some documentation",
    -1,
    MillipydeMethods
};

PyMODINIT_FUNC
PyInit_millipyde(void)
{
    import_array();
    PyObject *m;

    MPStatus status;
    long device_count;

    /*
     * Setup the devices on the system 
     */

    status = mpdev_initialize();
    if (status != MILLIPYDE_SUCCESS) {
        PyErr_SetString(PyExc_ImportError, 
                    mperr_str(status));
        return NULL;
    }
    if (mpdev_get_device_count() > 1 && mpdev_peer_to_peer_supported() == MP_FALSE)
    {
        PyErr_WarnEx(PyExc_ImportWarning,
                     mperr_str(DEV_WARN_NO_PEER_ACCESS),
                     1);
    }

    /*
     * Register cleanup functions
     */

    Py_AtExit(mpdev_teardown);

    /*
     * Prepare all of the types
     */

    if (PyType_Ready(&PyGPUArray_Type) < 0)
    {
        PyErr_SetString(PyExc_ImportError,
                        mperr_str(MOD_ERROR_CREATE_GPUARRAY_TYPE));
        return NULL;
    }

    PyGPUImage_Type.tp_base = &PyGPUArray_Type;
    if (PyType_Ready(&PyGPUImage_Type) < 0)
    {
        PyErr_SetString(PyExc_ImportError,
                        mperr_str(MOD_ERROR_CREATE_GPUIMAGE_TYPE));
        return NULL;
    }

    if (PyType_Ready(&PyGPUOperation_Type) < 0)
    {
        PyErr_SetString(PyExc_ImportError,
                        mperr_str(MOD_ERROR_CREATE_OPERATION_TYPE));
        return NULL;
    }

    if (PyType_Ready(&PyGPUPipeline_Type) < 0)
    {
        PyErr_SetString(PyExc_ImportError,
                        mperr_str(MOD_ERROR_CREATE_PIPELINE_TYPE));
        return NULL;
    }

    if (PyType_Ready(&PyGPUGenerator_Type) < 0)
    {
        PyErr_SetString(PyExc_ImportError,
                        mperr_str(MOD_ERROR_CREATE_GENERATOR_TYPE));
        return NULL;
    }

    if (PyType_Ready(&PyDevice_Type) < 0)
    {
        PyErr_SetString(PyExc_ImportError,
                        mperr_str(MOD_ERROR_CREATE_DEVICE_TYPE));
        return NULL;
    }

    /* 
     * Ceate the module object 
     */

    m = PyModule_Create(&millipydeModule);
    if (m == NULL) {
        return NULL;
    }

    /*
     * Add constants
     */
    device_count = (long)(mpdev_get_device_count());
    PyModule_AddIntConstant(m, "DEVICE_COUNT", device_count);
    

    /*
     * Create all supported Millipyde objects 
     */

    Py_INCREF(&PyGPUArray_Type);
    if (PyModule_AddObject(m, "gpuarray", (PyObject *) &PyGPUArray_Type) < 0) {
        PyErr_Print();
        fprintf(stderr, mperr_str(MOD_ERROR_ADD_GPUARRAY));
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&PyGPUImage_Type);
    if (PyModule_AddObject(m, "gpuimage", (PyObject *) &PyGPUImage_Type) < 0) {
        PyErr_Print();
        fprintf(stderr, mperr_str(MOD_ERROR_ADD_GPUIMAGE));
        Py_DECREF(&PyGPUImage_Type);
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PyGPUOperation_Type);
    if (PyModule_AddObject(m, "Operation", (PyObject *) &PyGPUOperation_Type) < 0) {
        fprintf(stderr, mperr_str(MOD_ERROR_ADD_OPERATION));
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(&PyGPUImage_Type);
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        PyErr_Print();
        return NULL;
    }

    Py_INCREF(&PyGPUPipeline_Type);
    if (PyModule_AddObject(m, "Pipeline", (PyObject *) &PyGPUPipeline_Type) < 0) {
        fprintf(stderr, mperr_str(MOD_ERROR_ADD_PIPELINE));
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(&PyGPUImage_Type);
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(m);
        PyErr_Print();
        return NULL;
    }

    Py_INCREF(&PyDevice_Type);
    if (PyModule_AddObject(m, "Device", (PyObject *) &PyDevice_Type) < 0) {
        fprintf(stderr, mperr_str(MOD_ERROR_ADD_DEVICE));
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(&PyGPUImage_Type);
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(&PyGPUPipeline_Type);
        Py_DECREF(m);
        PyErr_Print();
        return NULL;
    }

    Py_INCREF(&PyGPUGenerator_Type);
    if (PyModule_AddObject(m, "Generator", (PyObject *) &PyGPUGenerator_Type) < 0) {
        fprintf(stderr, mperr_str(MOD_ERROR_ADD_GENERATOR));
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(&PyGPUImage_Type);
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(&PyGPUPipeline_Type);
        Py_DECREF(&PyDevice_Type);
        Py_DECREF(m);
        PyErr_Print();
        return NULL;
    }

    return m;
}