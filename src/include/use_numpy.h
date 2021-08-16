/*
 * This is a "hack" to get numpy to recognize different translation units
 * so that import_array() does not have to be called in each file.
 * It also imports the arrayobject header for us.
 *
 * Credit: https://stackoverflow.com/questions/47026900/pyarray-check-gives-segmentation-fault-with-cython-c
 */

//your fancy name for the dedicated PyArray_API-symbol
#define PY_ARRAY_UNIQUE_SYMBOL Millipyde_PyArray_API 

//this macro must be defined for the translation unit              
#ifndef INIT_NUMPY_ARRAY_CPP 
    #define NO_IMPORT_ARRAY //for usual translation units
#endif

//now, everything is setup, just include the numpy-arrays:
#include <numpy/arrayobject.h>
