#ifndef MILLIPYDE_IMAGE_H
#define MILLIPYDE_IMAGE_H
#include "millipyde.h"

/* Transposition constants */
#define TRANSPOSE_TILE_DIM 32
#define FLIP_TILE_DIM 32
#define ROTATE_BLOCK_DIM 32

/* Gaussian convolution constants */
#define KERNEL_RADIUS 8
#define KERNEL_W (2 * KERNEL_RADIUS + 1)
#define ROW_TILE_W 128
#define KERNEL_RADIUS_ALIGNED 16
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48

#ifdef __cplusplus
extern "C" {
#endif

MPStatus
mpimg_color_to_greyscale(MPObjData *obj_data, void *arg);

MPStatus
mpimg_transpose(MPObjData* obj_data, void *arg);

MPStatus
mpimg_gaussian(MPObjData* obj_data, void *arg);

MPStatus
mpimg_fliplr(MPObjData *obj_data, void *args);

MPStatus
mpimg_rotate(MPObjData *obj_data, void *args);

MPStatus
mpimg_brightness(MPObjData *obj_data, void *args);

MPStatus
mpimg_colorize(MPObjData *obj_data, void *args);

MPStatus
mpimg_adjust_gamma(MPObjData *obj_data, void *args);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_IMAGE_H