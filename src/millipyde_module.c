/*  Example of wrapping cos function from math.h with the Python-C-API. */

#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include "ndgpuarray.h"

/*
typedef struct {
    PyObject_HEAD
} TestObject;

static PyTypeObject TestType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "test",
    .tp_doc = "test",
    .tp_basicsize = sizeof(TestObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};

static PyObject* test_func(PyObject* self, PyObject* args)
{
    PyObject *array;
    if (!PyArg_ParseTuple(args, "O", &array)) {
        return NULL;
    }
    
    TestObject *obj;
    obj = (TestObject *) TestType.tp_alloc(&TestType, 0);
    PyObject *pobj = (PyObject *)obj;

    *array = *pobj;
    return pobj;
}
*/

static PyObject * test_func(PyObject* self, PyObject* args)
{
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
    "test_func", "Some documentation",
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
    if (PyModule_AddObject(m, "GPUArray", (PyObject *) &PyGPUArray_Type) < 0) {
        Py_DECREF(&PyGPUArray_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}