#ifndef MILLIPYDE_DEVICES_H
#define MILLIPYDE_DEVICES_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>


#ifdef __cplusplus
extern "C" {
#endif

int mphip_get_default_device();

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_DEVICES_H