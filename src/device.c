#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>

#include "device.h"
#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_objects.h"


void
PyDevice_dealloc(PyDeviceObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *
PyDevice_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyDeviceObject *self;
    self = (PyDeviceObject *)type->tp_alloc(type, 0);
    return (PyObject *) self;
}

int
PyDevice_init(PyDeviceObject *self, PyObject *args, PyObject *kwds)
{
    int device_id;

    if (!PyArg_ParseTuple(args, "i", &device_id))
    {
        return -1;
    }
    self->device_id = device_id;
    return 0;
}

PyObject *
PyDevice_enter(PyDeviceObject *self, void *closure)
{
    self->prev_device_id = mpdev_get_target_device();
    mpdev_set_target_device(self->device_id);
    return Py_BuildValue("i", self->device_id);
}

PyObject *
PyDevice_exit(PyDeviceObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *exc_type;
    PyObject *exc_val;
    PyObject *exc_tb;

    if (!PyArg_ParseTuple(args, "OOO", &exc_type, &exc_val, &exc_tb))
    {
        return NULL;
    }
    
    if(exc_type != Py_None)
    {
        PyErr_SetObject(exc_type, exc_val);
        mpdev_reset(self->device_id);
        return NULL;
    }

    mpdev_synchronize();
    mpdev_set_target_device(self->prev_device_id);
    return Py_True;
}