#include <stdlib.h>
#include <stdio.h>

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_objects.h"

#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"


extern "C"{

void 
mpobj_copy_from_host(MPObjData *obj_data, void *data, size_t nbytes)
{
    int device_id = mpdev_get_target_device();
    if (device_id == DEVICE_LOC_NO_AFFINITY)
    {
        device_id = mpdev_get_recommended_device();
    }

    HIP_CHECK(hipSetDevice(device_id));
    obj_data->mem_loc = device_id;

    // Free any existing data
    if(obj_data->device_data != NULL) {
        HIP_CHECK(hipFree(obj_data->device_data));
    }
    HIP_CHECK(hipMalloc(&(obj_data->device_data), nbytes));
    HIP_CHECK(hipMemcpy(obj_data->device_data, data, nbytes, hipMemcpyHostToDevice));
    obj_data->nbytes = nbytes;
}

// DO not set mem_loc. We aren't moving anything, just creating a copy
void *
mpobj_copy_to_host(MPObjData *obj_data)
{
    void *data = PyMem_Malloc(obj_data->nbytes);
    HIP_CHECK(hipSetDevice(obj_data->mem_loc));
    HIP_CHECK(hipMemcpy(data, obj_data->device_data, obj_data->nbytes, hipMemcpyDeviceToHost));
    return data;
}

void 
mpobj_change_device(MPObjData *obj_data, int new_device_id)
{
    hipStream_t stream = (hipStream_t)obj_data->stream;

    int prev_device_id = obj_data->mem_loc;

    if (prev_device_id == new_device_id || obj_data->device_data == NULL) {
        // No need to move it if we are already on that device
        return;
    }
    if (mpdev_can_use_peer(prev_device_id, new_device_id)) {
        // Set ourselves to the old device to enable peer 2 peer
        HIP_CHECK(hipSetDevice(prev_device_id));
        // TODO: There should be a lock somewhere so that someone else cant
        // Attempt to enable it, and then disable it before we transfer
        hipDeviceEnablePeerAccess(new_device_id, 0);

        // Set ourselves to the peer GPU and allocate GPU memory and initiate transfer
        HIP_CHECK(hipSetDevice(new_device_id));
        void *new_device_data;
        hipMalloc((void **)&new_device_data, obj_data->nbytes);
        //hipMemcpy(new_device_data, obj_data->device_data, obj_data->nbytes, hipMemcpyDeviceToDevice);
        hipMemcpyPeerAsync(new_device_data, new_device_id, obj_data->device_data, prev_device_id, obj_data->nbytes, stream);

        // Set ourselves back to old GPU to disable peer 2 peer and clean up
        HIP_CHECK(hipSetDevice(prev_device_id));
        hipDeviceDisablePeerAccess(new_device_id);
        hipFree(obj_data->device_data);
        obj_data->device_data = new_device_data;
        obj_data->mem_loc = new_device_id;
    }
    else {
        // TODO: Transfer the hard way using CPU
    }
}

void 
mpobj_dealloc_device_data(MPObjData *obj_data) {
    if (obj_data == NULL){
        return;
    }
    
    if (obj_data->device_data != NULL) 
    {
        HIP_CHECK(hipFree(obj_data->device_data));
    }
    obj_data->device_data = NULL; 
}


} // extern "C"