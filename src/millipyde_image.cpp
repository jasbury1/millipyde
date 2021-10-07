#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "millipyde_image.h"
#include "millipyde.h"

__global__ void g_color_to_greyscale(unsigned char * rgbImg, double * greyImg, 
        int width, int height, int channels)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int greyOffset = y * width + x;

        // 3 is for the 3 channels in rgb
        int rgbOffset = greyOffset * channels;
        unsigned char r = rgbImg[rgbOffset];
        unsigned char g = rgbImg[rgbOffset + 1];
        unsigned char b = rgbImg[rgbOffset + 2];
        greyImg[greyOffset] = fmin(1.0, (0.2125 * r + 0.7154 * g + 0.0721 * b) / 255);
    }
}

/*
 * We use this technique to optimize transposition using shared memory:
 * https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter3/HIP-Coding-3.pdf
 */
template <typename T>
__global__ void g_transpose(T *in_arr, T *out_arr, int width, int height)
{
    // Block dimensions must be square for shared memory intermediate block
    __shared__ T shared_block[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM];
	
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

	if (x < width && y < height) {
		int in_idx = y * width + x;
		shared_block[hipThreadIdx_y][hipThreadIdx_x] = in_arr[in_idx];
	}

	__syncthreads();

    x = hipThreadIdx_x + hipBlockIdx_y * TRANSPOSE_BLOCK_DIM;
    y = hipThreadIdx_y + hipBlockIdx_x * TRANSPOSE_BLOCK_DIM;


	if (x < height && y < width) {
		int out_idx = y * height + x;
		out_arr[out_idx] = shared_block[hipThreadIdx_x][hipThreadIdx_y];
	}
}

/*
 * Separable Gaussian Kernel source:
 * https://docs.nvidia.com/cuda/samples/3_Imaging/convolutionSeparable/doc/convolutionSeparable.pdf
 */
/*
template <typename T>
__global__ void g_gaussian_row(T *in_arr, T *out_arr, int width, int height)
{
    __shared__ T shared_data[2 * GAUSS_RADIUS + GAUSS_ROW_DIM];
    
    const int tile_start = hipBlockIdx_x * GAUSS_ROW_DIM;
    const int tile_end = tile_start + GAUSS_ROW_DIM - 1;
    const int apron_start = tile_start - GAUSS_RADIUS;
    const int apron_end = tile_end   + GAUSS_RADIUS;

    //Clamp tile and apron limits by image borders
    const int tile_end_clamped = min(tile_end, width - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, width - 1);

    //Row start index in d_Data[]
    const int row_start = hipBlockIdx_y * width;

    //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples 
    //of half-warp size, rowStart + apronStartAligned is also a 
    //multiple of half-warp size, thus having proper alignment 
    //for coalesced d_Data[] read.
    const int apron_start_aligned = tile_start - GAUSS_RADIUS_ALIGNED;

    const int load_pos = apron_start_aligned + hipThreadIdx_x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(load_pos >= apron_start){
        const int smem_pos = load_pos - apron_start;

        shared_data[smem_pos] = 
            ((load_pos >= apron_start_clamped) && (load_pos <= apron_end_clamped)) ?
            in_arr[row_start + load_pos] : 0;
    }


    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();

    const int write_pos = tile_start + hipThreadIdx_x;
    //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tileStart is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_Result[] write.
    if(write_pos <= tile_end_clamped){
        const int smes_pos = writePos - apronStart;
        float sum = 0;

        // TODO: Experiment with loop unrolling here
        for(int k = -1 * GAUSS_RADIUS; k <= GAUSS_RADIUS; k++)
            sum += data[smes_pos + k] * d_Kernel[KERNEL_RADIUS - k];

        out_arr[row_start + write_pos] = sum;
    }
}
*/

extern "C" {

void *
mpimg_color_to_greyscale(void *arg)
{
    GPUCapsule *capsule = (GPUCapsule *)arg;

    int channels = capsule->dims[2];
    int height = capsule->dims[0];
    int width = capsule->dims[1];

    unsigned char *d_rgb;
    double *d_grey;

    hipStream_t *stream = (hipStream_t *)capsule->stream;

    if (capsule->device_data != NULL) {
        d_rgb = (unsigned char*)(capsule->device_data);
    }
    else {
        //TODO
        return NULL;
    }

    HIP_CHECK(hipMalloc(&d_grey, (capsule->nbytes / channels) * sizeof(double)));

    capsule->nbytes = (capsule->nbytes / channels) * sizeof(double);
    capsule->ndims = 2;
    capsule->type = 12;
    capsule->dims[2] = (int)(width * sizeof(double));
    capsule->dims[3] = (int)(sizeof(double));

    hipLaunchKernelGGL(g_color_to_greyscale, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(32, 32, 1),
            0,
            *stream,
            d_rgb,
            d_grey,
            width,
            height,
            channels);

    capsule->device_data = d_grey;

    HIP_CHECK(hipFree(d_rgb));
    return NULL;
}


void *
mpimg_transpose(void *arg)
{
    GPUCapsule *capsule = (GPUCapsule *)arg;

    int height = capsule->dims[0];
    int width = capsule->dims[1];

    double *d_img;
    double *d_transpose;

    hipStream_t *stream = (hipStream_t *)capsule->stream; 

    if (capsule->device_data != NULL) {
        d_img = (double *)(capsule->device_data);
    }
    else {
        //TODO
        return NULL;
    }

    HIP_CHECK(hipMalloc(&d_transpose, capsule->nbytes));
    
    int temp = capsule->dims[0];
    capsule->dims[0] = capsule->dims[1];
    capsule->dims[1] = temp;
    capsule->dims[capsule->ndims] = capsule->dims[0] * sizeof(double);
    capsule->dims[capsule->ndims + 1] = sizeof(double);

    hipLaunchKernelGGL(g_transpose, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM, 1),
            //TODO: Double check this
            TRANSPOSE_BLOCK_DIM * TRANSPOSE_BLOCK_DIM * capsule->dims[capsule->ndims + 1],
            *stream,
            d_img,
            d_transpose,
            width,
            height);

    capsule->device_data = d_transpose;

    HIP_CHECK(hipFree(d_img));
    
    return NULL;
}


} // extern "C"