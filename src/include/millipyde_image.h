#ifndef MILLIPYDE_IMAGE_H
#define MILLIPYDE_IMAGE_H
#include "millipyde.h"

#define TRANSPOSE_BLOCK_DIM 32

#ifdef __cplusplus
extern "C" {
#endif

MPStatus
mpimg_color_to_greyscale(MPObjData *obj_data, void *arg);

MPStatus
mpimg_transpose(MPObjData* obj_data, void *arg);

MPStatus
mpimg_gaussian(MPObjData* obj_data, void *arg);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_IMAGE_H