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
    if (self->obj_data)
    {
        free(self->obj_data);
    }
    if (self->runnables)
    {
        int operation_size = PyList_Size(self->operations);
        for (int i = 0; i < operation_size; ++i)
        {
            if(self->runnables[i].args != NULL)
            {
                free(self->runnables[i].args);
            }
        }
        free(self->runnables);
    }
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
        self->device_id = DEVICE_LOC_NO_AFFINITY;
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

    // No device was specified. Default to target device
    else
    {
        self->device_id = mpdev_get_target_device();    
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
            self->runnables[iter].args = gpuoperation_args_from_name(operation->callable, operation->arg_tuple);
            self->runnables[iter].probability = operation->probability;
        }
    }

    self->inputs = inputs;
    self->operations = operations;

    return 0;
}


/*******************************************************************************
 * Connects two pipelines together so that running the first one will pass its
 * results on to the second pipeline to be automatically run. Connecting two
 * pipelines will automatically try to schedule them on separate devices for
 * enhanced parallel efficiency. If a device was specified while creating either
 * pipeline, however, the device specified will not be overwritten on either one.
 * The same is true for if either pipeline was created with a target device
 * specified globally. @see mpdev_get_target_device @see mpdev_set_target_device
 * 
 * @param self The pipeline that will run first
 * @param other The pipeline that receivers input from self
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUPipeline_connect_to(PyGPUPipelineObject *self, PyObject *other)
{
    PyGPUPipelineObject *receiver = (PyGPUPipelineObject *)other;
    int recommended_device = mpdev_get_recommended_device();
    int alternative_device = mpdev_get_alternative_device(recommended_device);

    // If we only one device available, schedule both on the same device
    if (alternative_device == DEVICE_LOC_NO_AFFINITY)
    {
        self->device_id = recommended_device;
        receiver->device_id = recommended_device;
    }
    // If neither pipeline is assigned to a device, assign them both to separate ones
    else if ((self->device_id == DEVICE_LOC_NO_AFFINITY) &&
        (receiver->device_id == DEVICE_LOC_NO_AFFINITY))
    {
        self->device_id = recommended_device;
        receiver->device_id = alternative_device;
    }
    // If our receiver is assigned to a device and we aren't, assign ourselves to a different one
    else if (self->device_id == DEVICE_LOC_NO_AFFINITY)
    {
        self->device_id = mpdev_get_alternative_device(receiver->device_id);
    }
    // If we are assigned to a device and our receiver isn't, assign our receiver to a different one
    else if (receiver->device_id == DEVICE_LOC_NO_AFFINITY)
    {
        receiver->device_id = mpdev_get_alternative_device(self->device_id);
    }

    self->receiver = receiver;
    return Py_None;
}


/*******************************************************************************
 * Calls for the pipeline to run. Will also automatically run all pipelines
 * that are connected starting at this one if this pipeline is the beginning of
 * a chain. Run synchronizes all work so that it is safe to assume all GPUs are
 * idle and all work is complete after run returns.
 * 
 * @param self The pipeline that will be run
 * @param ignored An unused argument
 * 
 * @return Py_None
 * 
 ******************************************************************************/
PyObject *
PyGPUPipeline_run(PyGPUPipelineObject *self, PyObject *Py_UNUSED(ignored))
{
    MPObjData *obj_data;
    Py_ssize_t num_inputs = PyList_Size(self->inputs);
    Py_ssize_t num_stages = PyList_Size(self->operations);
    Py_ssize_t iter;
    int device_id;
    PyGPUPipelineObject *receiver = self->receiver;

    MPBool cycle_devices = MP_FALSE;

    if (self->device_id == DEVICE_LOC_NO_AFFINITY)
    {
        int target_device = mpdev_get_target_device();

        // See if we globally speficied a device to use
        if (target_device != DEVICE_LOC_NO_AFFINITY)
        {   
            device_id = target_device;
        }
        // No limits on what devices we can use. So use them all
        else
        {
            cycle_devices = MP_TRUE;
            device_id = mpdev_get_recommended_device();
        }
    }
    else
    {
        device_id = self->device_id;
    }

    Py_BEGIN_ALLOW_THREADS
    for(iter = 0; iter < num_inputs; ++iter)
    {
        obj_data = MP_OBJ_DATA(PyList_GetItem(self->inputs, iter));

        ExecutionArgs *args = gpupipeline_create_args(obj_data,
                                                      self->runnables,
                                                      num_stages,
                                                      device_id,
                                                      (iter % THREADS_PER_DEVICE) + 1,
                                                      receiver);

        mpdev_submit_work(device_id, gpupipeline_thread_run_sequence, args);
        if(cycle_devices && ((iter + 1) % THREADS_PER_DEVICE == 0))
        {
            device_id = mpdev_get_next_device(device_id);
        }
    }

    // Wait for all threads to finish and for our GPU tasks to complete everywhere
    if (cycle_devices)
    {
        mpdev_hard_synchronize_all();
    }
    // Wait only for our own device and any connected devices
    else
    {
        // Wait for all threads to finish and for our GPU tasks to complete
        mpdev_hard_synchronize(device_id);

        // If we are piping the data elsewhere, we must wait for both ends of the pipe to finish
        // By the time our threads have ended, they should have already sent their data before our own synchronize
        // call, so its safe to now start waiting on the next receiver to synchronize
        PyGPUPipelineObject *cur_receiver = self->receiver;
        // Iterate through the chain to sync with all connected pipelines
        while (cur_receiver != NULL)
        {
            mpdev_hard_synchronize(cur_receiver->device_id);
            cur_receiver = cur_receiver->receiver;
        }
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

    int iter;
    MPBool should_run;

    void *stream_ptr = mpdev_get_stream(device_id, stream_id);
    
    obj_data->pinned = MP_TRUE;
    if (obj_data->mem_loc != device_id)
    {
        // Use temporary stream for memory transfer
        obj_data->stream = mpdev_get_stream(obj_data->mem_loc, stream_id);
        mpobj_change_device(obj_data, device_id);
    }

    obj_data->stream = stream_ptr;

    mpdev_set_device(device_id);

    for(iter = 0; iter < num_stages; ++iter)
    {
        if (runnables[iter].func != NULL)
        {
            // Skip over the function if it has a probability and the odds don't work out
            double probability = runnables[iter].probability;
            void * args = runnables[iter].args;
            if (probability > 0)
            {
                gpuoperation_evaluate_probability(&should_run, probability);
                if (!should_run)
                {
                    continue;
                }
            }
            // Call the function contained at this stage in the sequence
            runnables[iter].func(obj_data, args);
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
