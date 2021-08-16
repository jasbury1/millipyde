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
    return 0;
}

void mpdev_setup_peer_to_peer()
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
                    return;
                }
                if (can_access_peer != 1)
                {
                    // The device cannot access peer's memory
                    return;
                }
                if (hipSetDevice(peer_device) != hipSuccess)
                {
                    // The peer is not useable as a device
                    return;
                }
            }
            HIP_CHECK(hipSetDevice(device));
            HIP_CHECK(hipDeviceReset());
        }
        // Revert back to the first device we were on
        HIP_CHECK(hipSetDevice(initial_device));
    }
}

MPBool mpdev_peer_to_peer_supported()
{
    return peer_to_peer_supported ? MP_FALSE : MP_TRUE;
}

int mpdev_get_device_count()
{
    return device_count;
}

int mpdev_get_current_device()
{
    return current_device;
}




} // extern "C"