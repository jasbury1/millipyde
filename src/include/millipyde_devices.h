#ifndef MILLIPYDE_DEVICES_H
#define MILLIPYDE_DEVICES_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "millipyde.h"
#include "millipyde_workers.h"


#define DEVICE_STREAM_COUNT 5

#ifdef __cplusplus
extern "C" {
#endif


MPStatus 
mpdev_initialize();

void 
mpdev_teardown();

MPBool 
mpdev_peer_to_peer_supported();

MPBool 
mpdev_can_use_peer(int device, int peer_devce);

int mpdev_get_device_count();
int mpdev_get_current_device();
void mpdev_set_current_device(int device_id);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_DEVICES_H