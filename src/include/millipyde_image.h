#ifndef MILLIPYDE_IMAGE_H
#define MILLIPYDE_IMAGE_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "gpuimage.h"

#define TRANSPOSE_BLOCK_DIM 32

#ifdef __cplusplus
extern "C" {
#endif

void mpimg_color_to_greyscale(PyGPUImageObject *gpuimage);
void mpimg_transpose(PyGPUImageObject *gpuimage);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_IMAGE_H