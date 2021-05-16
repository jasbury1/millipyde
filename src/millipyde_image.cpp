#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "millipyde_image.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "use_numpy.h"

extern "C"{

void mpimg_color_to_greyscale(PyObject *array){
    printf("To Greyscale...\n");
    printf("GPUArray object : %p\n", array);
    printf("Array object : %p\n", array);
    printf("Func address : %p\n", PyArray_Squeeze);
}

} // extern "C"