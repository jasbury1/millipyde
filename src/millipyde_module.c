/*  Example of wrapping cos function from math.h with the Python-C-API. */

#include <Python.h>
#include <math.h>
#include <stdio.h>
#include "ndgpuarray.h"
#include "GPUKernels.h"

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

static PyObject * test_func2(PyObject* self, PyObject* args)
{
    PyObject *array;
    if (!PyArg_ParseTuple(args, "O", &array)) {
        return -1;
    }
    PyArray_Squeeze(array);
    printf("Stride 0: %d\n", PyArray_STRIDE(array, 0));
    printf("Stride 1: %d\n", PyArray_STRIDE(array, 1));
    npy_intp *dims = PyArray_DIMS(array);
    int ndim = PyArray_NDIM(array);
    printf("Array Dimensions: %d\n", ndim);

    PyArray_Dims newshape;
    npy_intp temp[4] = {2, 2, 16, 8};
    newshape.ptr = temp;
    newshape.len = 2;

    array = PyArray_Resize(array, &newshape, 0, NPY_ANYORDER);


    /*

    unsigned char *p = (unsigned char *)&d;
    size_t i;
    for (i=0; i < sizeof d; ++i)
        printf("%02x\n", p[i]);

    PyArray_Dims new_dims;
    
    npy_int * new_vals = PyMem_Malloc(3 * sizeof(npy_int));
    memcpy(new_vals, vals, 3 * sizeof(npy_int));
    new_vals[0]--;


    any = PyArray_Resize(any, &new_dims, 0, NPY_ANYORDER);
    */
    return Py_None;
    
}

static PyObject * test_func3(PyObject* self, PyObject* args)
{
    PyGPUArrayObject *gpuarr;
    if (!PyArg_ParseTuple(args, "O", &gpuarr)) {
        return -1;
    }
    
    printf("GPUArray object : %p\n", gpuarr);

    PyObject *array = gpuarr->base_array;

    printf("Array object : %p\n", array);

    printf("Func address : %p\n", PyArray_Squeeze);

    printf("Squeezing the array...\n");
    PyArray_Squeeze(array);
    printf("Done squeezing the array\n");
    return Py_None;
    
}

/*  define functions in module */
static PyMethodDef MillipydeMethods[] =
{
     {"test_func", test_func, METH_VARARGS, "Does this work?"},
     {"test_func2", test_func2, METH_VARARGS, "Does this work?"},
     {"test_func3", test_func3, METH_VARARGS, "Does this work?"},
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
        return NULL;
    }
    m = PyModule_Create(&millipydeModule);
    if(m == NULL) {
        return NULL;
    }

    Py_INCREF(&PyGPUArray_Type);
    if (PyModule_AddObject(m, "gpuarray", (PyObject *) &PyGPUArray_Type) < 0) {
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}