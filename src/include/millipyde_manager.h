#ifndef MILLIPYDE_MANAGER_H
#define MILLIPYDE_MANAGER_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "millipyde.h"


#ifdef __cplusplus
extern "C" {
#endif

MPStatus
mpman_initialize();

void
mpman_teardown();

void *
mpman_get_stream(int stream_num);


#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_MANAGER_H