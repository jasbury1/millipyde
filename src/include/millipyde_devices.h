#ifndef MILLIPYDE_DEVICES_H
#define MILLIPYDE_DEVICES_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "millipyde.h"


#ifdef __cplusplus
extern "C" {
#endif

int mpdev_initialize();

void mpdev_setup_peer_to_peer();

MPBool mpdev_peer_to_peer_supported();

int mpdev_get_device_count();

int mpdev_get_current_device();

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_DEVICES_H