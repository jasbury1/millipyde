#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "millipyde_image.h"
#include "millipyde.h"


static
MPStatus _gaussian_greyscale(MPObjData *obj_data);



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
__device__ __constant__ double d_kernel[KERNEL_W];

__global__ void g_gaussian_row_one_channel(
    double *d_Result,
    double *d_Data,
    int dataW,
    int dataH
){
    int radius = KERNEL_RADIUS;

    //Data cache
    __shared__ double data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = hipBlockIdx_x * ROW_TILE_W;
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    //Row start index in d_Data[]
    const int          rowStart = hipBlockIdx_y * dataW;

    //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples 
    //of half-warp size, rowStart + apronStartAligned is also a 
    //multiple of half-warp size, thus having proper alignment 
    //for coalesced d_Data[] read.
    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + hipThreadIdx_x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] = 
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    const int writePos = tileStart + hipThreadIdx_x;
    //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tileStart is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_Result[] write.
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        double sum = 0;

        for (int k = -1 * radius; k <= radius; ++k) {
            sum += data[smemPos + k] * d_kernel[radius - k];
        }
        d_Result[rowStart + writePos] = sum;
    }
}


__global__ void g_gaussian_col_one_channel(
    double *d_Result,
    double *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    int radius = KERNEL_RADIUS;

    //Data cache
    __shared__ double data[COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int         tileStart = (hipBlockIdx_y * COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = (hipBlockIdx_x * COLUMN_TILE_W) + hipThreadIdx_x;

    //Shared and global memory indices for current column
    int smemPos = (hipThreadIdx_y * COLUMN_TILE_W) + hipThreadIdx_x;
    int gmemPos = ((apronStart + hipThreadIdx_y) * dataW) + columnStart;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + hipThreadIdx_y; y <= apronEnd; y += hipBlockDim_y){
        data[smemPos] = 
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ? 
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data, 
    //loaded by another threads
    __syncthreads();
    //Shared and global memory indices for current column
    smemPos = ((hipThreadIdx_y + KERNEL_RADIUS) * COLUMN_TILE_W) + hipThreadIdx_x;
    gmemPos = ((tileStart + hipThreadIdx_y) * dataW) + columnStart;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + hipThreadIdx_y; y <= tileEndClamped; y += hipBlockDim_y){
        double sum = 0;
        for (int k = -1 * radius; k <= radius; ++k) {
            sum += 
                data[smemPos + k * COLUMN_TILE_W] *
                d_kernel[radius - k];
        }
        d_Result[gmemPos] = sum;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}

extern "C" {

MPStatus
mpimg_color_to_greyscale(MPObjData* obj_data, void *args)
{
    MP_UNUSED(args);
    int device_id;
    int channels = obj_data->dims[2];
    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    unsigned char *d_rgb;
    double *d_grey;

    device_id = obj_data->mem_loc;
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = (hipStream_t)obj_data->stream;

    if (obj_data->device_data != NULL) {
        d_rgb = (unsigned char*)(obj_data->device_data);
    }
    else {
        //TODO
        return MILLIPYDE_SUCCESS;
    }

    HIP_CHECK(hipMalloc(&d_grey, (obj_data->nbytes / channels) * sizeof(double)));

    obj_data->nbytes = (obj_data->nbytes / channels) * sizeof(double);
    obj_data->ndims = 2;
    obj_data->type = 12;
    obj_data->dims[2] = (int)(width * sizeof(double));
    obj_data->dims[3] = (int)(sizeof(double));

    hipLaunchKernelGGL(g_color_to_greyscale, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(32, 32, 1),
            0,
            stream,
            d_rgb,
            d_grey,
            width,
            height,
            channels);

    obj_data->device_data = d_grey;

    HIP_CHECK(hipFree(d_rgb));
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpimg_transpose(MPObjData *obj_data, void *args)
{
    MP_UNUSED(args);
    int device_id;

    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    double *d_img;
    double *d_transpose;

    device_id = obj_data->mem_loc;
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = (hipStream_t)obj_data->stream;

    if (obj_data->device_data != NULL) {
        d_img = (double *)(obj_data->device_data);
    }
    else {
        //TODO
        return MILLIPYDE_SUCCESS;
    }

    HIP_CHECK(hipMalloc(&d_transpose, obj_data->nbytes));
    
    int temp = obj_data->dims[0];
    obj_data->dims[0] = obj_data->dims[1];
    obj_data->dims[1] = temp;
    obj_data->dims[obj_data->ndims] = obj_data->dims[0] * sizeof(double);
    obj_data->dims[obj_data->ndims + 1] = sizeof(double);

    hipLaunchKernelGGL(g_transpose, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM, 1),
            //TODO: Double check this
            TRANSPOSE_BLOCK_DIM * TRANSPOSE_BLOCK_DIM * obj_data->dims[obj_data->ndims + 1],
            stream,
            d_img,
            d_transpose,
            width,
            height);

    obj_data->device_data = d_transpose;

    HIP_CHECK(hipFree(d_img));
    
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpimg_gaussian(MPObjData *obj_data, void *args)
{
    MP_UNUSED(args);

    // If we only have x,y dimensions, we are greyscale (one channel)
    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        return _gaussian_greyscale(obj_data);
    }
    else{
        // TODO
    }
    return MILLIPYDE_SUCCESS;
}

} // extern "C"

static
MPStatus _gaussian_greyscale(MPObjData *obj_data)
{
    int sigma = 2;

    int device_id = obj_data->mem_loc;
    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    double *h_kernel;

    double *d_gaussian;
    double *d_image = (double *)(obj_data->device_data);

    HIP_CHECK(hipSetDevice(device_id));
    hipStream_t stream = (hipStream_t)obj_data->stream;

    h_kernel = (double *)malloc(KERNEL_W * sizeof(double));

    HIP_CHECK(hipMalloc(&d_gaussian, obj_data->nbytes));

    // Prepare the kernel
    double kernel_sum = 0;
    for (int i = 0; i < KERNEL_W; i++)
    {
        double dist = (double)(i - KERNEL_RADIUS) / (double)KERNEL_RADIUS;
        h_kernel[i] = expf(-1 * ((dist * dist) / (2 * sigma * sigma)));
        kernel_sum += h_kernel[i];
    }
    // Normalize the kernel
    for (int i = 0; i < KERNEL_W; ++i)
    {
        h_kernel[i] /= kernel_sum;
    }
    hipMemcpyToSymbol(HIP_SYMBOL(d_kernel), h_kernel, KERNEL_W * sizeof(double));

        hipLaunchKernelGGL(
            g_gaussian_row_one_channel,
            dim3(ceil(width / ROW_TILE_W), height, 1),
            dim3(ceil(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS)),
            (KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS) * sizeof(double),
            stream,
            d_gaussian,
            d_image,
            width,
            height);

        hipStreamSynchronize(stream);

        hipLaunchKernelGGL(
            g_gaussian_col_one_channel,
            dim3(ceil(width / COLUMN_TILE_W), ceil(height / COLUMN_TILE_H), 1),
            dim3(COLUMN_TILE_W, 8),
            (COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)) * sizeof(double),
            stream,
            d_image,
            d_gaussian,
            width,
            height,
            COLUMN_TILE_W * 8,
            width * 8);

    HIP_CHECK(hipFree(d_gaussian));
    
    return MILLIPYDE_SUCCESS;
}