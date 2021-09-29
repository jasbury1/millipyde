#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "use_numpy.h"
#include "gpuarray.h"
#include "gpuimage.h"
#include "gpupipeline.h"
#include "gpuoperation.h"
#include "millipyde_devices.h"
#include "millipyde_workers.h"
#include "millipyde_objects.h"

typedef struct execution_arguments {
    int stream_id;
    int device_id;
    PyObject *input;
    PyObject *operations;
} ExecutionArgs;


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
        self->device_id = 0;
    }
    return (PyObject *) self;
}


int
PyGPUPipeline_init(PyGPUPipelineObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *inputs = NULL;
    PyObject *operations = NULL;
    PyObject *device_arg;
    int device_id;

    int iter;

    Py_ssize_t num_call_args = PyTuple_Size(args);

    if(num_call_args < 2 || num_call_args > 3)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUPIPELINE_ERROR_CONSTRUCTION_INVALID_ARGS));
        return -1;
    }

    inputs = PyTuple_GetItem(args, 0);
    operations = PyTuple_GetItem(args, 1);

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

    if (kwds)
    {
        device_arg = PyDict_GetItemString(kwds, "device");

        if (device_arg && PyDict_Size(kwds) == 1)
        {
            if (PyLong_Check(device_arg))
            {
                device_id = PyLong_AsLong(device_arg);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUPIPELINE_ERROR_INVALID_DEVICE));
                return -1;
            }

            if (!mpdev_is_valid_device(device_id))
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUPIPELINE_ERROR_UNUSABLE_DEVICE));
                return -1;
            }
        }
        else
        {
            PyErr_SetString(PyExc_ValueError,
                            mperr_str(GPUPIPELINE_ERROR_CONSTRUCTION_NAMED_ARGS));
            return -1;
        }
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
        ExecutionArgs *args = (ExecutionArgs *)malloc(sizeof(ExecutionArgs));

        input = PyList_GetItem(self->inputs, iter);

        args->device_id = self->device_id;
        args->stream_id = iter % DEVICE_STREAM_COUNT;
        args->input = input;
        args->operations = self->operations;

        gpupipeline_run_stages((void *)args);
    }
    mpdev_synchronize(self->device_id);
    return Py_None;
}


void *
gpupipeline_run_stages(void *arg)
{
    ExecutionArgs *args = (ExecutionArgs *)arg;
    int stream = args->stream_id;
    int device_id = args->device_id;
    PyObject *operations = args->operations;
    PyObject *input = args->input;

    PyGPUOperationObject *operation;
    PyObject *result;
    Py_ssize_t num_stages = PyList_Size(operations);
    Py_ssize_t iter;

    ((PyGPUArrayObject *)input)->stream = mpdev_get_stream(device_id, stream);

    // TODO: set the device

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

    free(args);
    return (void *)result;
}

