#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "millipyde_image.h"
#include "millipyde.h"

__device__ __constant__ double d_kernel[KERNEL_W];


static
MPStatus _gaussian_greyscale(MPObjData *obj_data, int sigma);

static
MPStatus _gaussian_rgba(MPObjData *obj_data, int sigma);



__global__ void g_color_to_greyscale(unsigned char * d_rgb_data, double * d_grey_data, 
        int width, int height, int channels)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int greyOffset = y * width + x;

        // 3 is for the 3 channels in rgb
        int rgbOffset = greyOffset * channels;
        unsigned char r = d_rgb_data[rgbOffset];
        unsigned char g = d_rgb_data[rgbOffset + 1];
        unsigned char b = d_rgb_data[rgbOffset + 2];
        d_grey_data[greyOffset] = fmin(1.0, (0.2125 * r + 0.7154 * g + 0.0721 * b) / 255);
    }
}

/*
 * We use this technique to optimize transposition using shared memory:
 * https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter3/HIP-Coding-3.pdf
 */
template <typename T>
__global__ void g_transpose(T *d_data, T *d_result, int width, int height)
{
    // Block dimensions must be square for shared memory intermediate block
    __shared__ T shared_block[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM];
	
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

	if (x < width && y < height) {
		int in_idx = y * width + x;
		shared_block[hipThreadIdx_y][hipThreadIdx_x] = d_data[in_idx];
	}

	__syncthreads();

    x = hipThreadIdx_x + hipBlockIdx_y * TRANSPOSE_BLOCK_DIM;
    y = hipThreadIdx_y + hipBlockIdx_x * TRANSPOSE_BLOCK_DIM;


	if (x < height && y < width) {
		int out_idx = y * height + x;
		d_result[out_idx] = shared_block[hipThreadIdx_x][hipThreadIdx_y];
	}
}

/*
 * Separable Gaussian Kernel source:
 * https://docs.nvidia.com/cuda/samples/3_Imaging/convolutionSeparable/doc/convolutionSeparable.pdf
 */
__global__ void g_gaussian_row_one_channel(
    double *d_result,
    double *d_data,
    int width,
    int height)
{
    //Data cache
    __shared__ double data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int tile_start = hipBlockIdx_x * ROW_TILE_W;
    const int tile_end = tile_start + ROW_TILE_W - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int tile_end_clamped = min(tile_end, width - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, width - 1);

    //Row start index in d_data[]
    const int rowStart = hipBlockIdx_y * width;

    //Aligned apron start. Assuming width and ROW_TILE_W are multiples
    //of half-warp size, rowStart + apron_startAligned is also a
    //multiple of half-warp size, thus having proper alignment
    //for coalesced d_data[] read.
    const int apron_startAligned = tile_start - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apron_startAligned + hipThreadIdx_x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if (loadPos >= apron_start)
    {
        const int smemPos = loadPos - apron_start;

        data[smemPos] =
            ((loadPos >= apron_start_clamped) && (loadPos <= apron_end_clamped)) ? d_data[rowStart + loadPos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    const int writePos = tile_start + hipThreadIdx_x;
    //Assuming width and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tile_start is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_result[] write.
    if (writePos <= tile_end_clamped)
    {
        const int smemPos = writePos - apron_start;
        double sum = 0;

        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            sum += data[smemPos + k] * d_kernel[KERNEL_RADIUS - k];
        }
        d_result[rowStart + writePos] = sum;
    }
}

__global__ void g_gaussian_col_one_channel(
    double *d_result,
    double *d_data,
    int width,
    int height,
    int smemStride,
    int gmemStride)
{
    //Data cache
    __shared__ double data[COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int tile_start = (hipBlockIdx_y * COLUMN_TILE_H);
    const int tile_end = tile_start + COLUMN_TILE_H - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int tile_end_clamped = min(tile_end, height - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, height - 1);

    //Current column index
    const int col_start = (hipBlockIdx_x * COLUMN_TILE_W) + hipThreadIdx_x;

    //Shared and global memory indices for current column
    int smemPos = (hipThreadIdx_y * COLUMN_TILE_W) + hipThreadIdx_x;
    int gmemPos = ((apron_start + hipThreadIdx_y) * width) + col_start;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for (int y = apron_start + hipThreadIdx_y; y <= apron_end; y += hipBlockDim_y)
    {
        data[smemPos] =
            ((y >= apron_start_clamped) && (y <= apron_end_clamped)) ? d_data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    //Shared and global memory indices for current column
    smemPos = ((hipThreadIdx_y + KERNEL_RADIUS) * COLUMN_TILE_W) + hipThreadIdx_x;
    gmemPos = ((tile_start + hipThreadIdx_y) * width) + col_start;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for (int y = tile_start + hipThreadIdx_y; y <= tile_end_clamped; y += hipBlockDim_y)
    {
        double sum = 0;
        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            sum +=
                data[smemPos + k * COLUMN_TILE_W] *
                d_kernel[KERNEL_RADIUS - k];
        }
        d_result[gmemPos] = sum;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}

__global__ void g_gaussian_row_four_channel(
    uint32_t *d_result,
    uint32_t *d_data,
    int width,
    int height)
{
    //Data cache
    __shared__ uint32_t data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int tile_start = hipBlockIdx_x * ROW_TILE_W;
    const int tile_end = tile_start + ROW_TILE_W - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int tile_end_clamped = min(tile_end, width - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, width - 1);

    //Row start index in d_data[]
    const int rowStart = hipBlockIdx_y * width;

    //Aligned apron start. Assuming width and ROW_TILE_W are multiples
    //of half-warp size, rowStart + apron_startAligned is also a
    //multiple of half-warp size, thus having proper alignment
    //for coalesced d_data[] read.
    const int apron_startAligned = tile_start - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apron_startAligned + hipThreadIdx_x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if (loadPos >= apron_start)
    {
        const int smemPos = loadPos - apron_start;

        data[smemPos] =
            ((loadPos >= apron_start_clamped) && (loadPos <= apron_end_clamped)) ? d_data[rowStart + loadPos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    const int writePos = tile_start + hipThreadIdx_x;
    //Assuming width and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tile_start is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_result[] write.
    if (writePos <= tile_end_clamped)
    {
        const int smemPos = writePos - apron_start;
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int a_sum = 0;
        
        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            r_sum +=
                (int)(((data[smemPos + k] >> 24) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            g_sum +=
                (int)(((data[smemPos + k] >> 16) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            b_sum +=
                (int)(((data[smemPos + k] >> 8) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            a_sum +=
                (int)(((data[smemPos + k]) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
        }

        d_result[rowStart + writePos] = (0 |
                             ((r_sum & 0xff) << 24) |
                             ((g_sum & 0xff) << 16) |
                             ((b_sum & 0xff) << 8) |
                             (a_sum & 0xff));
    }
}


__global__ void g_gaussian_col_four_channel(
    uint32_t *d_result,
    uint32_t *d_data,
    int width,
    int height,
    int smemStride,
    int gmemStride)
{
    //Data cache
    __shared__ uint32_t data[COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int tile_start = (hipBlockIdx_y * COLUMN_TILE_H);
    const int tile_end = tile_start + COLUMN_TILE_H - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int tile_end_clamped = min(tile_end, height - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, height - 1);

    //Current column index
    const int col_start = (hipBlockIdx_x * COLUMN_TILE_W) + hipThreadIdx_x;

    //Shared and global memory indices for current column
    int smemPos = (hipThreadIdx_y * COLUMN_TILE_W) + hipThreadIdx_x;
    int gmemPos = ((apron_start + hipThreadIdx_y) * width) + col_start;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for (int y = apron_start + hipThreadIdx_y; y <= apron_end; y += hipBlockDim_y)
    {
        data[smemPos] =
            ((y >= apron_start_clamped) && (y <= apron_end_clamped)) ? d_data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    //Shared and global memory indices for current column
    smemPos = ((hipThreadIdx_y + KERNEL_RADIUS) * COLUMN_TILE_W) + hipThreadIdx_x;
    gmemPos = ((tile_start + hipThreadIdx_y) * width) + col_start;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for (int y = tile_start + hipThreadIdx_y; y <= tile_end_clamped; y += hipBlockDim_y)
    {
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int a_sum = 0;
        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            r_sum +=
                (int)(((data[smemPos + k * COLUMN_TILE_W] >> 24) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            g_sum +=
                (int)(((data[smemPos + k * COLUMN_TILE_W] >> 16) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            b_sum +=
                (int)(((data[smemPos + k * COLUMN_TILE_W] >> 8) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            a_sum +=
                (int)(((data[smemPos + k * COLUMN_TILE_W]) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);

                    /*sum +=
                data[smemPos + k * COLUMN_TILE_W] *
                d_kernel[KERNEL_RADIUS - k];*/
        }
        d_result[gmemPos] = (0 |
                             ((r_sum & 0xff) << 24) |
                             ((g_sum & 0xff) << 16) |
                             ((b_sum & 0xff) << 8) |
                             (a_sum & 0xff));
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
    int sigma = 2;

    // If we only have x,y dimensions, we are greyscale (one channel)
    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        return _gaussian_greyscale(obj_data, sigma);
    }
    else{
        // TODO
        return _gaussian_rgba(obj_data, sigma);
    }
    return MILLIPYDE_SUCCESS;
}

} // extern "C"

static
MPStatus _gaussian_greyscale(MPObjData *obj_data, int sigma)
{
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


static
MPStatus _gaussian_rgba(MPObjData *obj_data, int sigma)
{
    int device_id = obj_data->mem_loc;
    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    double *h_kernel;

    uint32_t *d_gaussian;
    uint32_t *d_image = (uint32_t *)(obj_data->device_data);

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
            g_gaussian_row_four_channel,
            dim3(ceil(width / ROW_TILE_W), height, 1),
            dim3(ceil(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS)),
            (KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS) * sizeof(uint32_t),
            stream,
            d_gaussian,
            d_image,
            width,
            height);

        hipStreamSynchronize(stream);

        hipLaunchKernelGGL(
            g_gaussian_col_four_channel,
            dim3(ceil(width / COLUMN_TILE_W), ceil(height / COLUMN_TILE_H), 1),
            dim3(COLUMN_TILE_W, 8),
            (COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)) * sizeof(uint32_t),
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
