#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

extern "C"{

int mphip_get_default_device() {
    int device_id;
    if (hipGetDevice(&device_id) != hipSuccess) {
        return -1;
    }
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id) != hipSuccess) {
        return -1;
    }
    return device_id;
}

} // extern "C"