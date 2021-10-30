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
    int device_id;
    Py_ssize_t num_call_args = PyTuple_Size(args);

    if(num_call_args < 2 || num_call_args > 4)
    {
        // TODO: Throw an error
        return -1;
    }

    inputs = PyTuple_GetItem(args, 0);
    operations = PyTuple_GetItem(args, 1);

    if (kwds)
    {
        device_arg = PyDict_GetItemString(kwds, "device");
        outputs_arg = PyDict_GetItemString(kwds, "outputs");

        if ((!device_arg && !outputs_arg) ||
            (!device_arg && PyDict_Size(kwds) != 1) ||
            (!outputs_arg && PyDict_Size(kwds) != 1))
        {
            PyErr_SetString(PyExc_ValueError,
                            mperr_str(GPUGENERATOR_ERROR_CONSTRUCTION_NAMED_ARGS));
            return -1;
        }

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
        generator->i++;

        PyObject *result = Py_None;
        return result;
    }
    else
    {
        Py_CLEAR(generator->inputs);
        Py_CLEAR(generator->operations);
        // Stop iteration error raised automatically by next() builtin
        return NULL;
    }
}
