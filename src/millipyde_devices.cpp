#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "millipyde_devices.h"
#include "millipyde_hip_util.h"


static int device_count = 0;
static int current_device = 0;
static bool peer_to_peer_supported = false;

static int** peer_access_matrix;

static void _setup_peer_to_peer();
static void _init_peer_access_matrix();
static void _delete_peer_access_matrix();

extern "C"{

int mpdev_initialize()
{
    if (hipGetDeviceCount(&device_count) != hipSuccess)
    {
        return -1;
    }
    if (hipGetDevice(&current_device) != hipSuccess)
    {
        return -1;
    }
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, current_device) != hipSuccess) {
        return -1;
    }

    if (device_count > 1) {
        try {
            _init_peer_access_matrix();
        }
        catch (std::bad_alloc&) {
            return -1;
        }
        
        _setup_peer_to_peer();

        // teardown will assume the matrix only has to be freed if peer to peer is supported.
        // handle the cleanup case where it we determine it cant be supported upon setup
        if(!peer_to_peer_supported) {
            _delete_peer_access_matrix();
        }
    }
    
    return 0;
}

void mpdev_teardown()
{
    if (peer_to_peer_supported) {
        _delete_peer_access_matrix();
    }
}


MPBool mpdev_peer_to_peer_supported()
{
    return peer_to_peer_supported ? MP_FALSE : MP_TRUE;
}

MPBool mpdev_can_use_peer(int device, int peer_devce)
{
    if (peer_to_peer_supported && peer_access_matrix[device][peer_devce] == 1) {
        return MP_TRUE;
    }
    return MP_FALSE;
}

int mpdev_get_device_count()
{
    return device_count;
}

int mpdev_get_current_device()
{
    return current_device;
}

void mpdev_set_current_device(int device_id)
{
    HIP_CHECK(hipSetDevice(device_id));
    current_device = device_id;
}

static void _setup_peer_to_peer()
{
    int initial_device = current_device;

    for (int device = 0; device < device_count; ++device) {
        if (hipSetDevice(device) != hipSuccess)
        {
            // TODO
            continue;
        }
        for (int peer_device = 0; peer_device < device_count; ++peer_device) {
            if (device != peer_device)
            {
                int can_access_peer;
                if (hipDeviceCanAccessPeer(&can_access_peer, device, peer_device) != hipSuccess)
                {
                    // TODO
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