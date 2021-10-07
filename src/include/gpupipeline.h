#ifndef MP_GPU_PIPELINE_H
#define MP_GPU_PIPELINE_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "structmember.h"
#include "millipyde.h"

/*******************************************************************************
* DOCUMENTATION
*******************************************************************************/

#define __GPUPIPELINE_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUPIPELINE_RUN_DOC PyDoc_STR( \
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \n \
sed do eiusmod tempor incididunt ut labore et dolore magna \n \
aliqua. Ut enim ad minim veniam, quis nostrud exercitation \n \
ullamco laboris nisi ut aliquip ex ea commodo consequat. \n \
Duis aute irure dolor in reprehenderit in voluptate velit \n \
esse cillum dolore eu fugiat nulla pariatur. \n \
Excepteur sint occaecat cupidatat non proident")

#define __GPUPIPELINE_CONNECT_TO_DOC PyDoc_STR( \
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
    PyObject *inputs;
    PyObject *operations;
    int device_id;
    PyObject *receiver;
} PyGPUPipelineObject;


/*******************************************************************************
* FUNCTION HEADERS
*******************************************************************************/


void
PyGPUPipeline_dealloc(PyGPUPipelineObject *self);

PyObject *
PyGPUPipeline_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

int
PyGPUPipeline_init(PyGPUPipelineObject *self, PyObject *args, PyObject *kwds);

PyObject *
PyGPUPipeline_run(PyGPUPipelineObject *self, PyObject *ignored);

PyObject *
PyGPUPipeline_connect_to(PyGPUPipelineObject *self, PyObject *receiver);

void
gpupipeline_run_sequence(PyObject *input, PyObject *operations, int device_id, int stream_id);

void *
gpupipeline_run_stages(void *arg);


/*******************************************************************************
* TYPE DATA
*******************************************************************************/


static PyMemberDef PyGPUPipeline_members[] = {
    {NULL}
};

static PyMethodDef PyGPUPipeline_methods[] = {
    {"run", (PyCFunction)PyGPUPipeline_run, METH_NOARGS,
     __GPUPIPELINE_RUN_DOC},
     {"connect_to", (PyCFunction)PyGPUPipeline_connect_to, METH_O,
     __GPUPIPELINE_CONNECT_TO_DOC},
    {NULL}};

static PyTypeObject PyGPUPipeline_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "millipyde.Pipeline",
    .tp_basicsize = sizeof(PyGPUPipelineObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)PyGPUPipeline_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = __GPUPIPELINE_DOC,
    .tp_methods = PyGPUPipeline_methods,
    .tp_members = PyGPUPipeline_members,
    .tp_init = (initproc)PyGPUPipeline_init,
    .tp_new = PyGPUPipeline_new,
};

#endif // MP_GPU_PIPELINE_H