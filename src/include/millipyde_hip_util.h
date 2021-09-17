#ifndef MILLIPYDE_HIP_UTIL_H
#define MILLIPYDE_HIP_UTIL_H

#include <stdio.h>
#include "hip/hip_runtime.h"

#define HIP_CHECK(cmd)                                                         \
    {                                                                          \
        hipError_t error = cmd;                                                \
        if (error != hipSuccess) {                                             \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                      \
                    hipGetErrorString(error), error,                           \
                    __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#endif //MILLIPYDE_HIP_UTIL_H