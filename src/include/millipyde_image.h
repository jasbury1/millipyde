#ifndef MILLIPYDE_IMAGE_H
#define MILLIPYDE_IMAGE_H

// PY_SSIZE_T_CLEAN Should be defined before including Python.h
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

PyObject * mpimg_color_to_greyscale(PyObject *array);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_IMAGE_H