#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "millipyde.h"
#include "millipyde_devices.h"
#include "millipyde_hip_util.h"

typedef struct mp_device {
    MPBool valid;
    hipStream_t streams[DEVICE_STREAM_COUNT];
    MPDeviceWorkPool *work_pool;
} MPDevice;


static int device_count = 0;

// A global state variable for where the user wants to place objects. Does not
// correspond to the actual current HIP device
static int current_device = 0;
static bool peer_to_peer_supported = false;

static int** peer_access_matrix;

MPDevice *device_array = NULL;

static void _setup_peer_to_peer();
static void _init_peer_access_matrix();
static void _delete_peer_access_matrix();

extern "C"{

MPStatus 
mpdev_initialize()
{
    if (hipGetDeviceCount(&device_count) != hipSuccess)
    {
        return DEV_ERROR_DEVICE_COUNT;
    }
    if (hipGetDevice(&current_device) != hipSuccess)
    {
        return DEV_ERROR_DEVICE_COUNT;
    }
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, current_device) != hipSuccess) {
        return DEV_ERROR_DEVICE_PROPERTIES;
    }

    if (device_count > 1) {
        try {
            _init_peer_access_matrix();
        }
        catch (std::bad_alloc&) {
            return DEV_ERROR_PEER_ACCESS_MATRIX_ALLOC;
        }
        
        _setup_peer_to_peer();

        // teardown will assume the matrix only has to be freed if peer to peer is supported.
        // handle the cleanup case where it we determine it cant be supported upon setup
        if(!peer_to_peer_supported) {
            _delete_peer_access_matrix();
        }
    }

    try {
        device_array = new MPDevice[device_count];
    }
    catch (std::bad_alloc&) {
        return DEV_ERROR_DEVICE_ARRAY_ALLOC;
    }

    // Initialize each device in the device array
    for (int i = 0; i < device_count; ++i)
    {
        MPDevice &device = device_array[i];
        HIP_CHECK(hipSetDevice(i));

        // Initialize the null stream
        device.streams[0] = 0;

        // Initialize remaining streams
        for (int j = 1; j < DEVICE_STREAM_COUNT; ++j)
        {
            hipStreamCreate(&(device.streams[j])); 
        }

        // Set up the pool of workers for this device
        //device.work_pool = mpwrk_create_work_pool(WORKPOOL_NUM_WORKERS);

        device.valid = MP_TRUE;
    }

    HIP_CHECK(hipSetDevice(0));
    
    return MILLIPYDE_SUCCESS;
}


void 
mpdev_teardown()
{
    if (peer_to_peer_supported) {
        _delete_peer_access_matrix();
    }

    if (device_array != NULL)
    {
        for (int i = 0; i < device_count; ++i)
        {
            MPDevice &device = device_array[i];
            HIP_CHECK(hipSetDevice(i));

            // Destroy the streams we created
            for (int j = 1; j < DEVICE_STREAM_COUNT; ++j)
            {
                hipStreamDestroy(device.streams[j]);
            }

            // Set up the pool of workers for this device
            //mpwrk_destroy_work_pool(device.work_pool);
        }
        delete[] device_array;
    }
}


MPBool 
mpdev_peer_to_peer_supported()
{
    return peer_to_peer_supported ? MP_FALSE : MP_TRUE;
}


MPBool 
mpdev_can_use_peer(int device, int peer_devce)
{
    if (peer_to_peer_supported && peer_access_matrix[device][peer_devce] == 1) {
        return MP_TRUE;
    }
    return MP_FALSE;
}


MPBool
mpdev_is_valid_device(int device_id)
{
    if (device_id < 0 || device_id > device_count)
    {
        return MP_FALSE;
    }
    if (!device_array[device_id].valid)
    {
        return MP_FALSE;
    }
    return MP_TRUE;
}

void 
mpdev_set_device(int device_id)
{
    HIP_CHECK(hipSetDevice(device_id));
}


void
mpdev_stream_synchronize(int device_id, int stream_id)
{
    HIP_CHECK(hipStreamSynchronize(device_array[device_id].streams[stream_id]));
}


int 
mpdev_get_device_count()
{
    return device_count;
}


int 
mpdev_get_current_device()
{
    return current_device;
}


void 
mpdev_set_current_device(int device_id)
{
    HIP_CHECK(hipSetDevice(device_id));
    current_device = device_id;
}


void *
mpdev_get_stream(int device_id, int stream)
{
    return (void *)(&(device_array[device_id].streams[stream]));
}


void
mpdev_submit_work(int device_id, MPWorkItem work, void *arg)
{
    mpwrk_work_queue_push(device_array[device_id].work_pool, work, arg);
}


void
mpdev_synchronize(int device_id)
{
    printf("mpdev: Synchronizing device %d\n", device_id);
    mpwrk_work_wait(device_array[device_id].work_pool);
    HIP_CHECK(hipSetDevice(device_id));
    HIP_CHECK(hipDeviceSynchronize());
} 


static void _setup_peer_to_peer()
{
    int initial_device = current_device;

    for (int device = 0; device < device_count; ++device) {
        if (hipSetDevice(device) != hipSuccess)
        {
            device_array[device].valid = MP_FALSE;
            continue;
        }
        for (int peer_device = 0; peer_device < device_count; ++peer_device) {
            if (device != peer_device)
            {
                int can_access_peer;
                if (hipDeviceCanAccessPeer(&can_access_peer, device, peer_device) != hipSuccess)
                {
                    // Testing for peer access throws an error
                    continue;
                }
                if (can_access_peer != 1)
                {
                    // The device cannot access peer's memory
                    continue;
                }
                if (hipSetDevice(peer_device) != hipSuccess)
                {
                    // The peer is not useable as a device
                    continue;
                }
                // This is a valid peer-to-peer combination
                peer_to_peer_supported = true;
                peer_access_matrix[device][peer_device] = 1;
            }
            HIP_CHECK(hipSetDevice(device));
            HIP_CHECK(hipDeviceReset());
        }
        // Revert back to the first device we were on
        HIP_CHECK(hipSetDevice(initial_device));
    }
}


static void _init_peer_access_matrix()
{
    peer_access_matrix = new int*[device_count];
    for (int i = 0; i < device_count; ++i) {
        peer_access_matrix[i] = new int[device_count];
    }
}


static void _delete_peer_access_matrix()
{
    for (int i = 0; i < device_count; ++i) {
        delete [] peer_access_matrix[i];
    }
    delete [] peer_access_matrix;
}



} // extern "C"