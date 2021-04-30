#ifndef ND_GPU_ARRAY_H
#define ND_GPU_ARRAY_H

#include <Python.h>
#include <numpy/arrayobject.h>

typedef struct {
    PyArrayObject base;
    /* Type-specific fields go here. */
} PyGPUArrayObject;

static PyTypeObject PyGPUArray_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "custom.Custom",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(PyGPUArrayObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};



#endif // ND_GPU_ARRAY_H