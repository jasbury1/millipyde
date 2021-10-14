#include <stdio.h>
#include <iostream>
#include <unistd.h>
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


// Total number of devices recognized by millipyde
static int device_count = 0;

// Refers to the device that functionality should default to as specified by the user
static int target_device = DEVICE_LOC_NO_AFFINITY;

// Refers to the device that millipyde recommends defaulting to for best performance
static int recommended_device = 0;

// Whether or not DMA peer-to-peer is supported. Gets set to 'true' during initialization if applicable
static bool peer_to_peer_supported = false;

// the value of peer_access_matrix[row][col] determines whether device 'row' and device 'col' can communicate via peer-to-peer
static int** peer_access_matrix;

MPDevice *device_array = NULL;

static int _update_recommended_device();
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

    recommended_device = _update_recommended_device();
    if (recommended_device == -1)
    {
        return DEV_ERROR_DEVICE_COUNT;
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
        device.work_pool = mpwrk_create_work_pool(WORKPOOL_NUM_WORKERS);

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
            mpwrk_destroy_work_pool(device.work_pool);
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
mpdev_get_target_device()
{
    return target_device;
}


void
mpdev_set_target_device(int device_id)
{
    if (device_id < 0 || device_id > device_count)
    {
        //TODO Throw an error
    }
    else if (device_array[device_id].valid == MP_FALSE)
    {
        //TODO throw an error
    }
    target_device = device_id;
}


int 
mpdev_get_recommended_device()
{
    return recommended_device;
}


void *
mpdev_get_stream(int device_id, int stream)
{
    return (void *)(device_array[device_id].streams[stream]);
}


void
mpdev_submit_work(int device_id, MPWorkItem work, void *arg)
{
    mpwrk_work_queue_push(device_array[device_id].work_pool, work, arg);
}


void
mpdev_synchronize(int device_id)
{
    //sleep(5);
    mpwrk_work_wait(device_array[device_id].work_pool);
    HIP_CHECK(hipSetDevice(device_id));
    HIP_CHECK(hipDeviceSynchronize());
} 

/*******************************************************************************
 * Tries every possible combination of two unique devices for all devices
 * recognized by the ROCm runtime. If two devices can communicate via peer to
 * peer DMA, then the result is updated in the peer access matrix. If at least
 * one pair exists that can communicate, then peer_to_peer_supported is updated
 * to true.
 ******************************************************************************/
static void _setup_peer_to_peer()
{
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


/*******************************************************************************
 * Iterates through all devices currently accessible by the ROCm runtime and 
 * returns the "best" one. Uses (MaximumComputeUnits * MaximumClockFrequency) 
 * as the metric for deciding the best.
 * 
 * @return an integer representing the device id of the recommended device, or
 *         -1 if no valid device could be recommended
 ******************************************************************************/
static int _update_recommended_device()
{
    int device_id = -1;
    double best_metric = 0;

    for (int i = 0; i < device_count; ++i)
    {
        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, i) != hipSuccess)
        {
            continue;
        }

        double metric = props.clockRate * props.multiProcessorCount;
        if (metric > best_metric)
        {
            device_id = i;
            best_metric = metric;
        }
    }
    return device_id;
}



} // extern "C"