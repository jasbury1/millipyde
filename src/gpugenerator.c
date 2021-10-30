#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "use_numpy.h"
#include "gpuarray.h"
#include "gpuimage.h"
#include "gpugenerator.h"
#include "gpuoperation.h"
#include "millipyde_devices.h"
#include "millipyde_workers.h"
#include "millipyde_objects.h"


void
PyGPUGenerator_dealloc(PyGPUGeneratorObject *self)
{
    // These may be set to NULL by Py_CLEAR after all outputs produced
    Py_XDECREF(self->inputs);
    Py_XDECREF(self->operations);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *
PyGPUGenerator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUGeneratorObject *self;
    self = (PyGPUGeneratorObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        self->device_id = DEVICE_LOC_NO_AFFINITY;
        self->max = NO_OUTPUT_MAX;
        self->i = 0;
        self->inputs = NULL;
        self->operations = NULL;
        self->return_to_host = MP_FALSE;
    }
    return (PyObject *) self;
}


int
PyGPUGenerator_init(PyGPUGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *inputs;
    PyObject *operations;
    PyObject *outputs_arg;
    PyObject *device_arg;
    PyObject *return_arg;
    int device_id;
    Py_ssize_t num_call_args = PyTuple_Size(args);

    if(num_call_args < 2 || num_call_args > 5)
    {
        // TODO: Throw an error
        return -1;
    }

    // GetItem returns borrowed references
    inputs = PyTuple_GetItem(args, 0);
    operations = PyTuple_GetItem(args, 1);
    if (kwds)
    {
        // GetItemString returns borrowed references
        device_arg = PyDict_GetItemString(kwds, "device");
        outputs_arg = PyDict_GetItemString(kwds, "outputs");
        return_arg = PyDict_GetItemString(kwds, "return_to_host");

        // Set the device id only if one was specified
        if (device_arg)
        {
            if (PyLong_Check(device_arg))
            {
                device_id = PyLong_AsLong(device_arg);
                self->device_id = device_id;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUGENERATOR_ERROR_INVALID_DEVICE));
                return -1;
            }

            if (!mpdev_is_valid_device(device_id))
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUGENERATOR_ERROR_UNUSABLE_DEVICE));
                return -1;
            }
        }

        // If no device id was specified, don't make any assumptions about what device
        // outputs will be produced on. This will be determined in the generation func

        // Set the number of outputs only if one was specified
        if (outputs_arg)
        {
            if (PyLong_Check(outputs_arg))
            {
                self->max = PyLong_AsLong(outputs_arg);
                if (self->max < 0)
                {
                    PyErr_SetString(PyExc_ValueError,
                                    mperr_str(GPUGENERATOR_ERROR_INVALID_MAX));
                    return -1;
                }
            }
            else
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUGENERATOR_ERROR_INVALID_MAX));
                return -1;
            }
        }

        // If no output count was specified, don't set one. We will assume we produce
        // infinite outputs

        if (return_arg)
        {
            if (PyBool_Check(return_arg))
            {
                 self->return_to_host = PyObject_IsTrue(return_arg);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUGENERATOR_ERROR_INVALID_RETURN_TO));
                return -1;
            }
        }
    }

    // Our inputs can either be a list or a path
    if(!PyList_CheckExact(inputs) && !PyUnicode_Check(inputs))
    {
        PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUGENERATOR_ERROR_INVALID_INPUT));
        return -1;
    }

    if(!PyList_CheckExact(operations))
    {
        PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUGENERATOR_ERROR_NONLIST_OPERATIONS));
        return -1;
    }

    // Turn the path into a list of inputs
    if (PyUnicode_Check(inputs))
    {
        // We own the inputs. No need to increment reference count
        self->inputs = gpuimage_all_from_path(inputs);
    }
    else
    {
        // Borrowing inputs. Increment reference count
        Py_INCREF(inputs);
        self->inputs = inputs;
    }

    Py_INCREF(operations);
    self->operations = operations;
    

    return 0;
}


PyObject *
PyGPUGenerator_iter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}


PyObject *
PyGPUGenerator_next(PyObject *self)
{
    PyGPUGeneratorObject *generator = (PyGPUGeneratorObject *)self;

    if (generator->max == NO_OUTPUT_MAX || generator->i < generator->max)
    {
        PyObject *result = gpugenerator_produce_next(generator, generator->i);
        generator->i++;
        return result;
    }
    else
    {
        // Decrement reference counts, set reference to NULL
        Py_CLEAR(generator->inputs);
        Py_CLEAR(generator->operations);

        // Stop iteration error raised automatically by next() builtin
        return NULL;
    }
}

PyObject *
gpugenerator_produce_next(PyGPUGeneratorObject *generator, int i)
{
    int input_size = PyList_Size(generator->inputs);
    int device_id;
    PyObject *result = NULL;

    // GetItem produces a borrowed reference
    PyObject *input = PyList_GetItem(generator->inputs, i % input_size);
    if (!input)
    {
        return NULL;
    }

    // If no device was specified, see if one is targeted
    if (generator->device_id == DEVICE_LOC_NO_AFFINITY)
    {
        device_id = mpdev_get_target_device();
        if (device_id == DEVICE_LOC_NO_AFFINITY)
        {
            // Default to recommended device
            device_id = mpdev_get_recommended_device();
        }
    }
    // A device was specified at instantiation. Use that one
    else
    {
        device_id = generator->device_id;
    }

    // Create the right output type based on input type
    if (gpuarray_check(input))
    {
        result = gpuarray_clone((PyGPUArrayObject *)input, device_id, 0);
    }
    else if (gpuimage_check(input))
    {
        result = gpuimage_clone((PyGPUImageObject *)input, device_id, 0);
    }
    else
    {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(TYPE_ERROR_NON_GPUOBJ));
        return NULL;
    }

    int operation_size = PyList_Size(generator->operations);
    for (int i = 0; i < operation_size; ++i)
    {   
        PyGPUOperationObject *operation =
            (PyGPUOperationObject *)PyList_GetItem(generator->operations, i);
        if (operation == NULL)
        {
            // TODO: Real error type
            Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError,
                            mperr_str(-1));
            return NULL;
        }

        if (operation->requires_instance)
        {
            PyGPUOperation_run_on(operation, result);
        }
        else
        {
            PyGPUOperation_run(operation, NULL);
        }
    }
    if (generator->return_to_host)
    {
        PyObject *host_array = PyGPUArray_to_array((PyGPUArrayObject *)result, NULL);
        //result->ob_type->tp_dealloc(result);
        Py_DECREF(result);
        return host_array;
    }
    return result;
}
