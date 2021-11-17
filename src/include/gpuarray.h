#ifndef MP_GPU_ARRAY_H
#define MP_GPU_ARRAY_H

#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"
#include "millipyde.h"

/*******************************************************************************
* DOCUMENTATION
*******************************************************************************/

#define __GPUARRAY_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUARRAY_TO_ARRAY_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUARRAY_ARRAY_FUNCTION_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUARRAY_ARRAY_UFUNC_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUARRAY_CLONE_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")


/*******************************************************************************
* STRUCTS
*******************************************************************************/

typedef struct {
    PyObject_HEAD
    void *array_data;
    MPObjData *obj_data;
} PyGPUArrayObject;


#define MP_OBJ_DATA(o) (((PyGPUArrayObject *)o)->obj_data)

/*******************************************************************************
* FUNCTION HEADERS
*******************************************************************************/

void
PyGPUArray_dealloc(PyGPUArrayObject *self);

PyObject *
PyGPUArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

int
PyGPUArray_init(PyGPUArrayObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUArray_to_array(PyGPUArrayObject *self, void *closure);

PyObject *
PyGPUArray_array_ufunc(PyGPUArrayObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUArray_array_function(PyGPUArrayObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUArray_clone(PyGPUArrayObject *self, void *closure);

PyObject *
gpuarray_clone(PyGPUArrayObject *self, int device_id, int stream_id);

MPBool
gpuarray_check(PyObject *object);

MPBool
gpuarray_check_subtype(PyObject *object);

/*******************************************************************************
* TYPE DATA
*******************************************************************************/

static PyMemberDef PyGPUArray_members[] = {
    {NULL}
};

static PyMethodDef PyGPUArray_methods[] = {
    {"__array__", (PyCFunction)PyGPUArray_to_array, METH_NOARGS,
     __GPUARRAY_TO_ARRAY_DOC},
    // {"__array_ufunc__", (PyCFunction) PyGPUArray_array_ufunc, METH_VARARGS,
    //  "TODO"
    // },
    {"__array_ufunc__", (PyCFunction)PyGPUArray_array_ufunc, METH_VARARGS,
     __GPUARRAY_ARRAY_UFUNC_DOC},
    {"__array_function__", (PyCFunction)PyGPUArray_array_function, METH_VARARGS,
     __GPUARRAY_ARRAY_FUNCTION_DOC},
    {"clone", (PyCFunction)PyGPUArray_clone, METH_NOARGS,
     __GPUARRAY_CLONE_DOC},
    {NULL}};

static PyTypeObject PyGPUArray_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "millipyde.gpuarray",                               /*tp_name*/
    sizeof(PyGPUArrayObject),                           /*tp_basicsize*/
    0,                                                  /*tp_itemsize*/
    /* methods */
    (destructor) PyGPUArray_dealloc,                    /*tp_dealloc*/
    0,                                                  /*tp_print*/
    0,                                                  /*tp_getattr*/
    0,                                                  /*tp_setattr*/
    0,                                                  /*tp_compare*/
    0,                                                  /*tp_repr*/
    0,                                                  /*tp_as_number*/
    0,                                                  /*tp_as_sequence*/
    0,                                                  /*tp_as_mapping*/
    0,                                                  /*tp_hash*/
    0,                                                  /*tp_call*/
    0,                                                  /*tp_str*/
    0,                                                  /*tp_getattro*/
    0,                                                  /*tp_setattro*/
    0,                                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,           /*tp_flags*/
    __GPUARRAY_DOC,                                     /*tp_doc*/
    0,                                                  /*tp_traverse*/
    0,                                                  /*tp_clear*/
    0,                                                  /*tp_richcompare*/
    0,                                                  /*tp_weaklistoffset*/
    0,                                                  /*tp_iter*/
    0,                                                  /*tp_iternext*/
    PyGPUArray_methods,                                 /*tp_methods*/
    PyGPUArray_members,                                 /*tp_members*/
    0,                                                  /*tp_getset*/
    0,                                                  /*tp_base*/
    0,                                                  /*tp_dict*/
    0,                                                  /*tp_descr_get*/
    0,                                                  /*tp_descr_set*/
    0,                                                  /*tp_dictoffset*/
    (initproc) PyGPUArray_init,                         /*tp_init*/
    0,                                                  /*tp_alloc*/
    PyGPUArray_new,                                     /*tp_new*/
    0,                                                  /*tp_free*/
    0,                                                  /*tp_is_gc*/
};



#endif // MP_GPU_ARRAY_H