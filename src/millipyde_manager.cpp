#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "gpuarray.h"
#include "gpuimage.h"
#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_manager.h"
#include "millipyde_hip_util.h"


hipStream_t streams[5];

extern "C"{

MPStatus 
mpman_initialize()
{   
    int i;
    streams[0] = 0;
    for(i = 1; i < 5; ++i)
    {
        hipStreamCreate(&streams[i + 1]);
    }
    return MILLIPYDE_SUCCESS;
}

void 
mpman_teardown()
{
    int i;
    for(i = 1; i < 5; ++i)
    {
        hipStreamDestroy(streams[i]);
    }
}

void *
mpman_get_stream(int stream_num)
{
    //TODO bounds checking
    return &streams[stream_num];
}


} // extern "C"