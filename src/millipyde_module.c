/*  Example of wrapping cos function from math.h with the Python-C-API. */

#include <Python.h>
#include <math.h>
#include <stdio.h>
#include "gpuarray.h"
#include "gpuimage.h"
#include "gpuoperation.h"
#include "GPUKernels.h"
#include "millipyde_devices.h"

#define INIT_NUMPY_ARRAY_CPP
#include "use_numpy.h"

void helper() {
    printf("Trying something!\n");
}

static PyObject * test_func(PyObject* self, PyObject* args)
{
    int test = run_bit_extract();
    printf("Test result: %d\n", test);
    return self;
}

/*  define functions in module */
static PyMethodDef MillipydeMethods[] =
{
     {"test_func", test_func, METH_VARARGS, "Does this work?"},
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
    
    /*
     * Prepare all of the types
     */

    if (PyType_Ready(&PyGPUArray_Type) < 0){
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'millipyde' while creating internal type 'gpuarray'\n");
        return NULL;
    }

    PyGPUImage_Type.tp_base = &PyGPUArray_Type;
    if (PyType_Ready(&PyGPUImage_Type) < 0){
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'millipyde' while creating internal type 'gpuimage'\n");
        return NULL;
    }

    if (PyType_Ready(&PyGPUOperation_Type) < 0){
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'millipyde' while creating internal type 'Operation'\n");
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
     * Setup the devices on the system 
     */
    
    if (-1 == mpdev_initialize()) {
        PyErr_SetString(PyExc_ImportError, 
                    "Millipyde could not succesfully find default GPU device(s) on this system.");
        return NULL;
    }
    if (mpdev_get_device_count() > 1 && mpdev_peer_to_peer_supported() == MP_FALSE)
    {
        PyErr_WarnEx(PyExc_ImportWarning,
                     "Multiple devices were detected, but peer2peer is not supported on this system.",
                     1);
    }

    // Register cleanup function for the device information
    Py_AtExit(mpdev_teardown);

    /*
     * Create all supported Millipyde objects 
     */

    Py_INCREF(&PyGPUArray_Type);
    if (PyModule_AddObject(m, "gpuarray", (PyObject *) &PyGPUArray_Type) < 0) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'millipyde' while loading internal type 'gpuarray'\n");
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        return NULL;
    }
    
    Py_INCREF(&PyGPUImage_Type);
    if (PyModule_AddObject(m, "gpuimage", (PyObject *) &PyGPUImage_Type) < 0) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'millipyde' while loading internal type 'gpuimage'\n");
        Py_DECREF(&PyGPUImage_Type);
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PyGPUOperation_Type);
    if (PyModule_AddObject(m, "Operation", (PyObject *) &PyGPUOperation_Type) < 0) {
        fprintf(stderr, "Error: could not import module 'millipyde' while loading internal type 'Operation'\n");
        Py_DECREF(&PyGPUOperation_Type);
        Py_DECREF(&PyGPUImage_Type);
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        PyErr_Print();
        return NULL;
    }

    return m;
}