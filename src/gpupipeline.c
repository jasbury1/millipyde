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
    MPObjData *obj_data;
    MPRunnable *runnables;
    int num_stages;
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
        if (self->obj_data)
        {
            free(self->obj_data);
        }
        if (self->runnables)
        {
            free(self->runnables);
        }
    }
    return (PyObject *) self;
}


int
PyGPUPipeline_init(PyGPUPipelineObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *inputs = NULL;
    PyGPUArrayObject *input = NULL;
    PyObject *operations = NULL;
    PyGPUOperationObject *operation = NULL;
    PyObject *device_arg;
    int device_id;

    int iter;
    int input_size;
    int operation_size;

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

    input_size = PyList_Size(inputs);
    operation_size = PyList_Size(operations);
    self->obj_data = calloc(input_size, sizeof(MPObjData *));
    self->runnables = calloc(operation_size, sizeof(MPRunnable));

    for (iter = 0; iter < input_size; ++iter)
    {
        input = (PyGPUArrayObject *)PyList_GetItem(inputs, iter);
        self->obj_data[iter] = input->obj_data;
    }

    for (iter = 0; iter < operation_size; ++iter)
    {
        operation = (PyGPUOperationObject *)PyList_GetItem(operations, iter);
        if (operation->requires_instance)
        {
            self->runnables[iter].func = gpuoperation_func_from_name(operation->callable);
        }
    }

    self->inputs = inputs;
    self->operations = operations;

    return 0;
}

ExecutionArgs *
gpupipeline_create_args(MPObjData *obj_data, MPRunnable *runnables, int num_stages, int device_id, int stream_id)
{
    ExecutionArgs *args = malloc(sizeof(ExecutionArgs));
    args->obj_data = obj_data;
    args->runnables = runnables;
    args->num_stages = num_stages;
    args->device_id = device_id;
    args->stream_id = stream_id;

    return args;
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
    MPObjData *obj_data = args->obj_data;
    MPRunnable *runnables = args->runnables;
    int num_stages = args->num_stages;
    int device_id = args->device_id;
    int stream_id = args->stream_id;

    gpupipeline_run_sequence(obj_data, runnables, num_stages, device_id, stream_id);
    
    free(args);

    return NULL;
}



void 
gpupipeline_run_sequence(MPObjData *obj_data, MPRunnable *runnables, int num_stages, int device_id, int stream_id)
{

    PyObject *result;
    int iter;

    void *stream_ptr = mpdev_get_stream(device_id, stream_id);
    
    obj_data->pinned = MP_TRUE;
    if (obj_data->mem_loc != device_id)
    {
        printf("Changing the memory location peer2peer\n");
        mpobj_change_device(obj_data, device_id);
    }

    obj_data->stream = stream_ptr;

    mpdev_set_device(device_id);

    // TODO: Segfaults when these iterations happen too fast
    for(iter = 0; iter < num_stages; ++iter)
    {
        if (runnables[iter].func != NULL)
        {
            runnables[iter].func(obj_data);
            mpdev_stream_synchronize(device_id, stream_id);
        }
        else
        {
            //TODO: Here is where we have to sync with GIL
            printf("That's a problem...\n"); 
        }
    }

    obj_data->pinned = MP_FALSE;
    obj_data->stream = mpdev_get_stream(device_id, 0);

    return;
}



PyObject *
PyGPUPipeline_run(PyGPUPipelineObject *self, PyObject *Py_UNUSED(ignored))
{
    MPObjData *obj_data;
    Py_ssize_t num_inputs = PyList_Size(self->inputs);
    Py_ssize_t num_stages = PyList_Size(self->operations);
    Py_ssize_t iter;

    Py_BEGIN_ALLOW_THREADS
    for(iter = 0; iter < num_inputs; ++iter)
    {
        obj_data = MP_OBJ_DATA(PyList_GetItem(self->inputs, iter));

        ExecutionArgs *args = gpupipeline_create_args(obj_data, self->runnables, num_stages, self->device_id, iter % DEVICE_STREAM_COUNT);
        
        if (pthread_create(&threads[iter], NULL, gpupipeline_thread_run_sequence, args))
        {
            printf("Error creating thread!!\n");
            return Py_None;
        }

        //gpupipeline_run_sequence(obj_data, self->runnables, num_stages, self->device_id, iter % DEVICE_STREAM_COUNT);
    }
    
    
    for(iter = 0; iter < num_inputs; ++iter)
    {
        if(pthread_join(threads[iter], NULL))
        {
            printf("Error joining threads!!\n");
        }
    }
    Py_END_ALLOW_THREADS
    
    return Py_None;
}



