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

    // Receivers are set by an alternate function. See PyGPUPipeline_connect_to
    self->receiver = NULL;

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


PyObject *
PyGPUPipeline_connect_to(PyGPUPipelineObject *self, PyObject *receiver)
{
    /* TODO: Type checking */
    /* TODO: Make sure receiver is assigned to different GPU than our own.
    If one of the the two is unassigned, auto assign. */
    self->receiver = receiver;
    return Py_None;
}


PyObject *
PyGPUPipeline_run(PyGPUPipelineObject *self, PyObject *Py_UNUSED(ignored))
{
    MPObjData *obj_data;
    Py_ssize_t num_inputs = PyList_Size(self->inputs);
    Py_ssize_t num_stages = PyList_Size(self->operations);
    Py_ssize_t iter;
    int device_id = self->device_id;
    PyGPUPipelineObject *receiver = self->receiver;

    Py_BEGIN_ALLOW_THREADS
    for(iter = 0; iter < num_inputs; ++iter)
    {
        obj_data = MP_OBJ_DATA(PyList_GetItem(self->inputs, iter));

        ExecutionArgs *args = gpupipeline_create_args(obj_data, self->runnables, num_stages,
                                                      device_id, iter % DEVICE_STREAM_COUNT,
                                                      receiver);
        
        mpdev_submit_work(device_id, gpupipeline_thread_run_sequence, args);
    }

    // Wait for all threads to finish and for our GPU tasks to complete
    mpdev_synchronize(device_id);

    // If we are piping the data elsewhere, we must wait for both ends of the pipe to finish
    // By the time our threads have ended, they should have already sent their data before our own synchronize
    // call, so its safe to now start waiting on the next receiver to synchronize
    if (self->receiver)
    {
        // TODO: This should be a synchronize chain just in case our receiver also has a receiver
        mpdev_synchronize(self->receiver->device_id);
    }

    // All threads should be idle. Safe to rejoin with GIL
    Py_END_ALLOW_THREADS
    
    return Py_None;
}


void
gpupipeline_send_input(PyGPUPipelineObject *receiver, MPObjData *obj_data, int stream_id)
{
    Py_ssize_t num_stages = PyList_Size(receiver->operations);
    PyGPUPipelineObject *next_receiver = receiver->receiver;
    int device_id = receiver->device_id;

    ExecutionArgs *args = gpupipeline_create_args(obj_data, receiver->runnables,
                                                  num_stages, device_id, stream_id,
                                                  next_receiver);
    mpdev_submit_work(device_id, gpupipeline_thread_run_sequence, args);
}

void
gpupipeline_thread_run_sequence(void *arg)
{
    ExecutionArgs *args = (ExecutionArgs *)arg;
    MPObjData *obj_data = args->obj_data;
    MPRunnable *runnables = args->runnables;
    PyGPUPipelineObject *receiver = args->receiver;
    int num_stages = args->num_stages;
    int device_id = args->device_id;
    int stream_id = args->stream_id;

    gpupipeline_run_sequence(obj_data, runnables, num_stages, device_id, stream_id);

    if (receiver != NULL)
    {
        gpupipeline_send_input(receiver, obj_data, stream_id);
    }
    
    free(args);
}

void 
gpupipeline_run_sequence(MPObjData *obj_data, MPRunnable *runnables, int num_stages,
                              int device_id, int stream_id)
{

    PyObject *result;
    int iter;

    void *stream_ptr = mpdev_get_stream(device_id, stream_id);
    
    obj_data->pinned = MP_TRUE;
    if (obj_data->mem_loc != device_id)
    {
        mpobj_change_device(obj_data, device_id);
    }

    obj_data->stream = stream_ptr;

    mpdev_set_device(device_id);

    for(iter = 0; iter < num_stages; ++iter)
    {
        if (runnables[iter].func != NULL)
        {
            // Call the function contained at this stage in the sequence
            runnables[iter].func(obj_data);
            // Synchronize with the stream before attempting to execute next stage
            mpdev_stream_synchronize(device_id, stream_id);
        }
        else
        {
            //TODO: Here is where we have to sync with GIL to perform non-parallel task
        }
    }

    obj_data->pinned = MP_FALSE;
    obj_data->stream = mpdev_get_stream(device_id, 0);

    return;
}

ExecutionArgs *
gpupipeline_create_args(MPObjData *obj_data, MPRunnable *runnables, int num_stages,
                        int device_id, int stream_id, PyGPUPipelineObject *receiver)
{
    ExecutionArgs *args = malloc(sizeof(ExecutionArgs));
    args->obj_data = obj_data;
    args->runnables = runnables;
    args->num_stages = num_stages;
    args->device_id = device_id;
    args->stream_id = stream_id;
    args->receiver = receiver;

    return args;
}
