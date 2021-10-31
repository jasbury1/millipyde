#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#include <Python.h>
#include <sys/random.h>
#include <limits.h>

#include "use_numpy.h"
#include "gpuimage.h"
#include "gpuoperation.h"
#include "millipyde_image.h"


static PyObject *
_gpuoperation_run(PyGPUOperationObject *self, PyObject *callable);


void
PyGPUOperation_dealloc(PyGPUOperationObject *self)
{
    Py_XDECREF(self->callable);
    Py_XDECREF(self->arg_tuple);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


PyObject *
PyGPUOperation_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGPUOperationObject *self;
    self = (PyGPUOperationObject *)type->tp_alloc(type, 0);
    if(self != NULL) {
        self->callable = NULL;
        self->arg_tuple = NULL;
        self->requires_instance = MP_FALSE;
        self->probability = -1.0;
    }
    return (PyObject *) self;
    
}


int
PyGPUOperation_init(PyGPUOperationObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *prob;
    double fprob = 1.0;

    PyObject *callable = NULL;
    PyObject *call_args; 

    Py_ssize_t num_call_args = PyTuple_Size(args);

    if (num_call_args < 1)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUOPERATION_ERROR_CONSTRUCTION_NO_ARGS));
        return -1;
    }

    if (kwds)
    {
        prob = PyDict_GetItemString(kwds, "probability");

        if (prob && PyDict_Size(kwds) == 1)
        {
            if (PyFloat_Check(prob))
            {
                fprob = PyFloat_AsDouble(prob);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUOPERATION_ERROR_INVALID_PROBABILITY));
                return -1;
            }

            if (fprob <= 0.0 || fprob >= 1.0)
            {
                PyErr_SetString(PyExc_ValueError,
                                mperr_str(GPUOPERATION_ERROR_INVALID_PROBABILITY));
                return -1;
            }

            self->probability = fprob;
        }
        else
        {
            PyErr_SetString(PyExc_ValueError,
                            mperr_str(GPUOPERATION_ERROR_CONSTRUCTION_NAMED_ARGS));
            return -1;
        }
    }

    callable = PyTuple_GetItem(args, 0);

    // Instance methods are denoted by a string rather than a callable
    if (PyUnicode_Check(callable))
    {
        self->requires_instance = MP_TRUE;
    }

    // If we only have the function name, we have no args so create empty tuple
    if (num_call_args == 1)
    {
        call_args = PyTuple_New(0);
    }
    // Create tuple from all args except the function name
    else
    {
        call_args = PyTuple_GetSlice(args, 1, num_call_args);
    }

    Py_INCREF(callable);
    Py_INCREF(call_args);

    self->callable = callable;
    self->arg_tuple = call_args;

    return 0;
}


PyObject *
PyGPUOperation_run(PyGPUOperationObject *self, PyObject *Py_UNUSED(ignored))
{
    return _gpuoperation_run(self, self->callable);
}


PyObject *
PyGPUOperation_run_on(PyGPUOperationObject *self, PyObject *instance)
{
    PyObject *result;
    PyObject *callable;

    const char *method_name = PyUnicode_AsUTF8(self->callable);
    if (!method_name)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUOPERATION_ERROR_RUN_WITHOUT_STRING_METHOD));
        return NULL;
    }

    callable = PyObject_GetAttrString(instance, method_name);
    if (!callable)
    {
        PyErr_SetString(PyExc_ValueError,
                        mperr_str(GPUOPERATION_ERROR_RUN_UNKNOWN_STRING_METHOD));
        return NULL;
    }

    Py_INCREF(instance);
    result = _gpuoperation_run(self, callable);
    Py_DECREF(instance);

    return result;
}


static PyObject *
_gpuoperation_run(PyGPUOperationObject *self, PyObject *callable)
{
    PyObject *result;
    MPStatus ret_val;
    MPBool should_run;
    
    ret_val = gpuoperation_evaluate_probability(&should_run, self->probability);
    if(ret_val != MILLIPYDE_SUCCESS)
    {
        PyErr_SetString(PyExc_RuntimeError,
                            mperr_str(ret_val));
        return NULL;
    }

    if (!should_run)
    {
        return Py_None;
    }
    result = PyObject_Call(callable, self->arg_tuple, NULL);
    return result;
}

/*
MPStatus
gpuoperation_evaluate_probability(MPBool *result, double probability)
{
    double r;
    unsigned int seed;
    FILE *f;
    
    *result = MP_TRUE;

    if(probability > 0) {
        f = fopen("/dev/random", "r");
        if (!f)
        {
            return GPUOPERATION_ERROR_RUN_NO_DEV_RANDOM;
        }
        if (fread(&seed, sizeof(seed), 1, f) != sizeof(seed))
        {
            fclose(f);
            return GPUOPERATION_ERROR_RUN_CANNOT_READ_DEV_RANDOM;
        }
        fclose(f);

        srand(seed);
        r = (double)rand() / RAND_MAX;

        if (r > probability) {
            *result = MP_FALSE;
        }
    }
    
    return MILLIPYDE_SUCCESS;
}
*/

MPStatus
gpuoperation_evaluate_probability(MPBool *result, double probability)
{
    unsigned int buffer;
    ssize_t bytes;

    *result = MP_TRUE;

    if (probability < 0)
    {
        return MILLIPYDE_SUCCESS;
    }

    bytes = getrandom(&buffer, sizeof(unsigned int), 0);
    if (bytes < 0)
    {
        // TODO
        return -1;
    }

    double r = (double)buffer / UINT_MAX;

    if (r > probability)
    {
        *result = MP_FALSE;
    }
    
    return MILLIPYDE_SUCCESS;
}


MPFunc
gpuoperation_func_from_name(PyObject *uname)    
{
    const char *name = PyUnicode_AsUTF8(uname);
    
    if(name == NULL)
    {
        return NULL;
    }

    if (strcmp(name, "rgb2grey") == 0)
    {
        return mpimg_color_to_greyscale;
    }
    else if (strcmp(name, "transpose") == 0)
    {
        return mpimg_transpose;
    }
    else if (strcmp(name, "gaussian") == 0)
    {
        return mpimg_gaussian;
    }
    else if (strcmp(name, "fliplr") == 0)
    {
        return mpimg_fliplr;
    }
    else if (strcmp(name, "rotate") == 0)
    {
        return mpimg_rotate;
    }

    return NULL;
}


void *
gpuoperation_args_from_name(PyObject *uname, PyObject *arg_tuple)    
{
    const char *name = PyUnicode_AsUTF8(uname);
    
    if(name == NULL)
    {
        return NULL;
    }
    else if (strcmp(name, "rotate") == 0)
    {
        return gpuimage_rotate_args(arg_tuple);
    }
    else if (strcmp(name, "gaussian") == 0)
    {
        return gpuimage_gaussian_args(arg_tuple);
    }

    return NULL;
}
