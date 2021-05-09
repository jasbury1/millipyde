#ifndef MILLIPYDE_GPU_KERNELS_H
#define MILLIPYDE_GPU_KERNELS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int run_bit_extract();
int add_one(void * data, int N);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_GPU_KERNELS_H