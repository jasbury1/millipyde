#ifndef MILLIPYDE_DEVICES_H
#define MILLIPYDE_DEVICES_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "millipyde.h"
#include "millipyde_workers.h"

// The maximum number of streams to user per-device. Includes the NULL stream.
#define DEVICE_STREAM_COUNT 5
// The number of non-null streams of concurrency we have per device
#define THREADS_PER_DEVICE ((DEVICE_STREAM_COUNT) - 1)


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

int 
mpdev_get_device_count();

MPBool
mpdev_is_valid_device(int device_id);

void *
mpdev_get_stream(int device_id, int stream);

void
mpdev_submit_work(int device_id, MPWorkItem work, void *arg);

void
mpdev_hard_synchronize(int device_id);

void
mpdev_hard_synchronize_all();

void
mpdev_synchronize();

void
mpdev_synchronize_all();

void
mpdev_reset();

void 
mpdev_set_device(int device_id);

void
mpdev_stream_synchronize(int device_id, int stream_id);

int
mpdev_get_target_device();

int
mpdev_get_alternative_device(int device_id);

int 
mpdev_get_next_device(int device_id);

void
mpdev_set_target_device(int device_id);

int 
mpdev_get_recommended_device();

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_DEVICES_H