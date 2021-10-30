#ifndef MP_GPU_GENERATOR_H
#define MP_GPU_GENERATOR_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"
#include "millipyde.h"

/*******************************************************************************
* DOCUMENTATION
*******************************************************************************/

#define __GPUGENERATOR_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")


#define NO_OUTPUT_MAX (-1)

/*******************************************************************************
* STRUCTS
*******************************************************************************/

typedef struct PyGPUGeneratorObject {
    PyObject_HEAD
    int device_id;
    PyObject *inputs;
    PyObject *operations;
    Py_ssize_t max;
    Py_ssize_t i;
    MPBool return_to_host;
} PyGPUGeneratorObject;


/*******************************************************************************
* FUNCTION HEADERS
*******************************************************************************/


void
PyGPUGenerator_dealloc(PyGPUGeneratorObject *self);

PyObject *
PyGPUGenerator_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

int
PyGPUGenerator_init(PyGPUGeneratorObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUGenerator_iter(PyObject *self);

PyObject *
PyGPUGenerator_next(PyObject *self);

PyObject *
gpugenerator_produce_next(PyGPUGeneratorObject *generator, int i);


/*******************************************************************************
* TYPE DATA
*******************************************************************************/


static PyMemberDef PyGPUGenerator_members[] = {
    {NULL}
};

static PyMethodDef PyGPUGenerator_methods[] = {
    {NULL}};


static PyTypeObject PyGPUGenerator_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "millipyde.Generator",                              /*tp_name*/
    sizeof(PyGPUGeneratorObject),                       /*tp_basicsize*/
    0,                                                  /*tp_itemsize*/
    /* methods */
    (destructor)PyGPUGenerator_dealloc,                 /*tp_dealloc*/
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
    Py_TPFLAGS_DEFAULT,                                 /*tp_flags*/
    __GPUGENERATOR_DOC,                                 /*tp_doc*/
    0,                                                  /*tp_traverse*/
    0,                                                  /*tp_clear*/
    0,                                                  /*tp_richcompare*/
    0,                                                  /*tp_weaklistoffset*/
    PyGPUGenerator_iter,                                /*tp_iter*/
    PyGPUGenerator_next,                                /*tp_iternext*/
    PyGPUGenerator_methods,                             /*tp_methods*/
    PyGPUGenerator_members,                             /*tp_members*/
    0,                                                  /*tp_getset*/
    0,                                                  /*tp_base*/
    0,                                                  /*tp_dict*/
    0,                                                  /*tp_descr_get*/
    0,                                                  /*tp_descr_set*/
    0,                                                  /*tp_dictoffset*/
    (initproc) PyGPUGenerator_init,                     /*tp_init*/
    0,                                                  /*tp_alloc*/
    PyGPUGenerator_new,                                 /*tp_new*/
    0,                                                  /*tp_free*/
    0,                                                  /*tp_is_gc*/
};

#endif // MP_GPU_GENERATOR_H