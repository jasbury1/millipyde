#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

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
    PyObject *input;
    PyObject *operations;
    int device_id;
    int stream_id;
} ExecutionArgs;

#define NUM_THREADS 4

static pthread_t threads[NUM_THREADS];



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
        self->device_id = mpdev_get_current_device();
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
                self->device_id = device_id;
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
PyGPUPipeline_connect_to(PyGPUPipelineObject *self, PyObject *receiver)
{
    /* TODO: Type checking */
    /* TODO: Make sure receiver is assigned to different GPU than our own.
    If one of the the two is unassigned, auto assign. */
    self->receiver = receiver;
    return Py_None;
}


void *
gpupipeline_thread_run_sequence(void *arg)
{
    ExecutionArgs *args = (ExecutionArgs *)arg;
    PyObject *input = args->input;    
    PyObject *operations = args->operations;
    int device_id = args->device_id;
    int stream_id = args->stream_id;

    gpupipeline_run_sequence(input, operations, device_id, stream_id);
    free(args);

    return NULL;
}


void 
gpupipeline_run_sequence(PyObject *input, PyObject *operations, int device_id, int stream_id)
{

    PyGPUArrayObject *array;
    GPUCapsule *capsule;
    PyObject *result;
    PyGPUOperationObject *operation;
    Py_ssize_t num_stages = PyList_Size(operations);
    Py_ssize_t iter;

    void *stream_ptr = mpdev_get_stream(device_id, stream_id);

    array = (PyGPUArrayObject *)input;
    capsule = array->capsule;
    
    capsule->pinned = MP_TRUE;
    if (capsule->mem_loc != device_id)
    {
        printf("Changing the memory location peer2peer\n");
        mpobj_change_device(capsule, device_id);
    }

    capsule->stream = stream_ptr;

    mpdev_set_device(device_id);

    // TODO: Segfaults when these iterations happen too fast
    for(iter = 0; iter < num_stages; ++iter)
    {
        operation = (PyGPUOperationObject *)PyList_GetItem(operations, iter);
        
        if (operation->requires_instance)
        {
            
            result = PyGPUOperation_run_on(operation, (PyObject *)array);
            
            mpdev_stream_synchronize(device_id, stream_id);
            
        }
        else
        {
            result = PyGPUOperation_run(operation, NULL);
        }
    }

    capsule->pinned = MP_FALSE;
    capsule->stream = mpdev_get_stream(device_id, 0);
    
    return;
}



PyObject *
PyGPUPipeline_run(PyGPUPipelineObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *input;
    Py_ssize_t num_inputs = PyList_Size(self->inputs);
    Py_ssize_t iter;

    for(iter = 0; iter < num_inputs; ++iter)
    {
        input = PyList_GetItem(self->inputs, iter);

        /*
        ExecutionArgs *args = (ExecutionArgs *)malloc(sizeof(ExecutionArgs));
        args->device_id = self->device_id;
        args->input = input;
        args->operations = self->operations;
        args->stream_id = iter % DEVICE_STREAM_COUNT;

        if (pthread_create(&threads[iter], NULL, gpupipeline_thread_run_sequence, args))
        {
            printf("Error creating thread!!\n");
            return Py_None;
        }
        */

        gpupipeline_run_sequence(input, self->operations, self->device_id, iter % DEVICE_STREAM_COUNT);
    }
    
    /*
    for(iter = 0; iter < num_inputs; ++iter)
    {
        if(pthread_join(threads[iter], NULL))
        {
            printf("Error joining threads!!\n");
        }
    }
    */
    return Py_None;
}



