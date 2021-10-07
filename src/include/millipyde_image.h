#ifndef MILLIPYDE_IMAGE_H
#define MILLIPYDE_IMAGE_H

#define TRANSPOSE_BLOCK_DIM 32

#ifdef __cplusplus
extern "C" {
#endif

void *
mpimg_color_to_greyscale(void *arg);

void *
mpimg_transpose(void *arg);

#ifdef __cplusplus
}
#endif

#endif // MILLIPYDE_IMAGE_H