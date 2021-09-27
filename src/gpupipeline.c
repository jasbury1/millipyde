#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "use_numpy.h"
#include "gpuarray.h"
#include "gpupipeline.h"
#include "gpuoperation.h"


void
PyGPUPipeline_dealloc(PyGPUPipelineObject *self)
{
    Py_XDECREF(self->inputs);
    Py_XDECREF(self->operations);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


PyObject *
PyGPUPipeline_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUPipelineObject *self;
    self = (PyGPUPipelineObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        self->inputs = NULL;
        self->operations = NULL;
    }
    return (PyObject *) self;
}


int
PyGPUPipeline_init(PyGPUPipelineObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *inputs = NULL;
    PyObject *operations = NULL;

    Py_ssize_t num_call_args = PyTuple_Size(args);

    if (!PyArg_ParseTuple(args, "OO", &inputs, &operations))
    {
        return -1;
    }

    if(!PyList_CheckExact(inputs))
    {
        PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUPIPELINE_ERROR_NONLIST_INPUTS));
        return -1;
    }

    if(!PyList_CheckExact(operations))
    {
        PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUPIPELINE_ERROR_NONLIST_OPERATIONS));
        return -1;
    }

    Py_INCREF(inputs);
    Py_INCREF(operations);

    self->inputs = inputs;
    self->operations = operations;

    return 0;
}


PyObject *
PyGPUPipeline_start(PyGPUPipelineObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *input;
    Py_ssize_t num_inputs = PyList_Size(self->inputs);
    Py_ssize_t iter;

    for(iter = 0; iter < num_inputs; ++iter)
    {
        input = PyList_GetItem(self->inputs, iter);
        gpupipeline_run_stages(input, self->operations);
    }
    return Py_None;
}


PyObject *
gpupipeline_run_stages(PyObject *input, PyObject *operations)
{
    PyGPUOperationObject *operation;
    PyObject *result;
    Py_ssize_t num_stages = PyList_Size(operations);
    Py_ssize_t iter;
    
    for(iter = 0; iter < num_stages; ++iter)
    {
        operation = (PyGPUOperationObject *)PyList_GetItem(operations, iter);
        if (operation->requires_instance)
        {
            result = PyGPUOperation_run_on(operation, input);
        }
        else
        {
            result = PyGPUOperation_run(operation, NULL);
        }
    }
    return result;
}

