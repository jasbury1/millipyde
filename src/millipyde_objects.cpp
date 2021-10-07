#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "use_numpy.h"
#include "gpuarray.h"
#include "gpuimage.h"
#include "gpupipeline.h"
#include "gpuoperation.h"
#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_workers.h"
#include "millipyde_objects.h"

#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"


extern "C"{

void 
mpobj_copy_from_host(PyGPUArrayObject *array, void *data, size_t nbytes)
{
    int device_id = mpdev_get_current_device();
    array->mem_loc = device_id;
    // Free any existing data
    if(array->device_data != NULL) {
        HIP_CHECK(hipFree(array->device_data));
    }
    HIP_CHECK(hipMalloc(&(array->device_data), nbytes));
    HIP_CHECK(hipMemcpy(array->device_data, data, nbytes, hipMemcpyHostToDevice));
    array->nbytes = nbytes;
}

// DO not set mem_loc. We aren't moving anything, just creating a copy
void *
mpobj_copy_to_host(PyGPUArrayObject *array)
{
    void *data = PyMem_Malloc(array->nbytes);
    HIP_CHECK(hipMemcpy(data, array->device_data, array->nbytes, hipMemcpyDeviceToHost));
    return data;
}

void 
mpobj_change_device(PyGPUArrayObject *array, int new_device_id)
{
    //TODO: Should we set it back to the old device at the end??
    int prev_device_id = array->mem_loc;

    if (prev_device_id == new_device_id || array->device_data == NULL) {
        // No need to move it if we are already on that device
        return;
    }
    if (mpdev_can_use_peer(prev_device_id, new_device_id)) {
        // Set ourselves to the old device to enable peer 2 peer
        HIP_CHECK(hipSetDevice(prev_device_id));
        HIP_CHECK(hipDeviceEnablePeerAccess(new_device_id, 0));

        // Set ourselves to the peer GPU and allocate GPU memory and initiate transfer
        HIP_CHECK(hipSetDevice(new_device_id));
        void *new_device_data;
        hipMalloc((void **)&new_device_data, array->nbytes);
        hipMemcpy(new_device_data, array->device_data, array->nbytes, hipMemcpyDeviceToDevice);

        // Set ourselves back to old GPU to disable peer 2 peer and clean up
        HIP_CHECK(hipSetDevice(prev_device_id));
        HIP_CHECK(hipDeviceDisablePeerAccess(new_device_id));
        hipFree(array->device_data);
        array->device_data = new_device_data;
        array->mem_loc = new_device_id;
    }
    else {
        // TODO: Transfer the hard way using CPU
    }
}

void 
mpobj_dealloc_device_data(PyGPUArrayObject *array) {
    if(array->device_data != NULL) {
        HIP_CHECK(hipFree(array->device_data));
    }
    array->device_data = NULL; 
}


} // extern "C"