/*  Example of wrapping cos function from math.h with the Python-C-API. */

#include <Python.h>
#include <math.h>
#include <stdio.h>
#include "gpuarray.h"
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
    if (PyType_Ready(&PyGPUArray_Type) < 0){
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'millipyde'\n");
        return NULL;
    }
    m = PyModule_Create(&millipydeModule);
    if(m == NULL) {
        return NULL;
    }

    if (-1 == mphip_get_default_device()) {
        PyErr_SetString(PyExc_ImportError, 
                    "Millipyde could not succesfully find default GPU device(s) on this system.");
        return NULL;
    }

    Py_INCREF(&PyGPUArray_Type);
    if (PyModule_AddObject(m, "gpuarray", (PyObject *) &PyGPUArray_Type) < 0) {
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'millipyde'\n");
        return NULL;
    }

    return m;
}