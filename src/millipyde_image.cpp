#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "millipyde_hip_util.h"
#include "millipyde_image.h"
#include "millipyde.h"

__device__ __constant__ double d_kernel[KERNEL_W];


static MPStatus 
_gaussian_greyscale(MPObjData *obj_data, double sigma);

static MPStatus 
_gaussian_rgba(MPObjData *obj_data, double sigma);

template <typename T>  MPStatus 
_transpose(MPObjData *obj_data);

template <typename T>  MPStatus 
_fliplr(MPObjData *obj_data);

template <typename T>  MPStatus 
_rotate(MPObjData *obj_data, double angle);

static MPStatus 
_brightness_greyscale(MPObjData *obj_data, double delta);

static MPStatus 
_brightness_rgba(MPObjData *obj_data, char delta);

static MPStatus 
_colorize_rgba(MPObjData *obj_data, double r_mult, double g_mult, double b_mult);


/*******************************************************************************
* GPU KERNELS
*******************************************************************************/


__global__ void g_color_to_greyscale(unsigned char * d_rgb_data, double * d_grey_data, 
        int width, int height, int channels)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int grey_offset = y * width + x;

        // 3 is for the 3 channels in rgb
        int rgb_offset = grey_offset * channels;
        unsigned char r = d_rgb_data[rgb_offset];
        unsigned char g = d_rgb_data[rgb_offset + 1];
        unsigned char b = d_rgb_data[rgb_offset + 2];
        d_grey_data[grey_offset] = fmin(1.0, (0.2125 * r + 0.7154 * g + 0.0721 * b) / 255);
    }
}


/*
 * We use this technique to optimize transposition using shared memory:
 * https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter3/HIP-Coding-3.pdf
 */
template <typename T>
__global__ void g_transpose(T *d_data, T *d_result, int width, int height)
{
    __shared__ T shared_tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM];
	
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

	if (x < width && y < height) {
		int in_idx = y * width + x;
		shared_tile[hipThreadIdx_y][hipThreadIdx_x] = d_data[in_idx];
	}

	__syncthreads();

    x = hipThreadIdx_x + hipBlockIdx_y * TRANSPOSE_TILE_DIM;
    y = hipThreadIdx_y + hipBlockIdx_x * TRANSPOSE_TILE_DIM;


	if (x < height && y < width) {
		int out_idx = y * height + x;
		d_result[out_idx] = shared_tile[hipThreadIdx_x][hipThreadIdx_y];
	}
}


template <typename T>
__global__ void g_flip_horizontal(T *d_data, T *d_result, int width, int height)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height)
    {
        int in_idx = y * width + x;
        int out_idx = y * width + (width - 1 - x);
        d_result[out_idx] = d_data[in_idx];
    }
}


template <typename T>
__global__ void g_rotate(T* d_data, T* d_result, int width, int height, double angle)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    int x_rot = ((double)x - ((double)width / 2)) * cos(angle) -
                ((double)y - ((double)height / 2)) * sin(angle) + ((double)width / 2);
    int y_rot = ((double)x - ((double)width / 2)) * sin(angle) +
                ((double)y - ((double)height / 2)) * cos(angle) + ((double)height / 2);
    
    if (x < width && y < height)
    {
        int in_idx = y * width + x;
        if (x_rot >= 0 && x_rot < width && y_rot >= 0 && y_rot < height)
        {
            
            int out_idx = y_rot * width + x_rot;
            d_result[in_idx] = d_data[out_idx];
        }
        else
        {
            d_result[in_idx] = (T)0;
        }
    }
}


/*
 * Separable Gaussian Kernel using this technique:
 * https://docs.nvidia.com/cuda/samples/3_Imaging/convolutionSeparable/doc/convolutionSeparable.pdf
 */
__global__ void g_gaussian_row_one_channel(
    double *d_result,
    double *d_data,
    int width,
    int height)
{
    __shared__ double data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    const int tile_start = hipBlockIdx_x * ROW_TILE_W;
    const int tile_end = tile_start + ROW_TILE_W - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;
    const int tile_end_clamped = min(tile_end, width - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, width - 1);
    const int row_start = hipBlockIdx_y * width;
    const int apron_start_aligned = tile_start - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apron_start_aligned + hipThreadIdx_x;
    if (loadPos >= apron_start)
    {
        const int shared_mem_pos = loadPos - apron_start;

        data[shared_mem_pos] =
            (((loadPos >= apron_start_clamped) && (loadPos <= apron_end_clamped))
                 ? d_data[row_start + loadPos]
                 : 0);
    }

    __syncthreads();

    const int write_pos = tile_start + hipThreadIdx_x;

    if (write_pos <= tile_end_clamped)
    {
        const int shared_mem_pos = write_pos - apron_start;
        double sum = 0;

        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            sum += data[shared_mem_pos + k] * d_kernel[KERNEL_RADIUS - k];
        }
        d_result[row_start + write_pos] = sum;
    }
}


__global__ void g_gaussian_col_one_channel(
    double *d_result,
    double *d_data,
    int width,
    int height,
    int shared_mem_stride,
    int global_mem_stride)
{
    __shared__ double data[COLUMN_TILE_W *
                           (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    const int tile_start = (hipBlockIdx_y * COLUMN_TILE_H);
    const int tile_end = tile_start + COLUMN_TILE_H - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;
    const int tile_end_clamped = min(tile_end, height - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, height - 1);
    const int col_start = (hipBlockIdx_x * COLUMN_TILE_W) + hipThreadIdx_x;

    int shared_mem_pos = (hipThreadIdx_y * COLUMN_TILE_W) + hipThreadIdx_x;
    int global_mem_pos = ((apron_start + hipThreadIdx_y) * width) + col_start;

    for (int y = apron_start + hipThreadIdx_y; y <= apron_end; y += hipBlockDim_y)
    {
        data[shared_mem_pos] =
            (((y >= apron_start_clamped) && (y <= apron_end_clamped))
                 ? d_data[global_mem_pos]
                 : 0);
        shared_mem_pos += shared_mem_stride;
        global_mem_pos += global_mem_stride;
    }

    __syncthreads();

    shared_mem_pos = ((hipThreadIdx_y + KERNEL_RADIUS) * COLUMN_TILE_W) + hipThreadIdx_x;
    global_mem_pos = ((tile_start + hipThreadIdx_y) * width) + col_start;

    for (int y = tile_start + hipThreadIdx_y; y <= tile_end_clamped; y += hipBlockDim_y)
    {
        double sum = 0;
        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            sum +=
                data[shared_mem_pos + k * COLUMN_TILE_W] *
                d_kernel[KERNEL_RADIUS - k];
        }
        d_result[global_mem_pos] = fmax(0.0, sum);
        shared_mem_pos += shared_mem_stride;
        global_mem_pos += global_mem_stride;
    }
}


__global__ void g_gaussian_row_four_channel(
    uint32_t *d_result,
    uint32_t *d_data,
    int width,
    int height)
{
    __shared__ uint32_t data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    const int tile_start = hipBlockIdx_x * ROW_TILE_W;
    const int tile_end = tile_start + ROW_TILE_W - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;
    const int tile_end_clamped = min(tile_end, width - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, width - 1);

    const int row_start = hipBlockIdx_y * width;
    const int apron_start_aligned = tile_start - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apron_start_aligned + hipThreadIdx_x;
    if (loadPos >= apron_start)
    {
        const int shared_mem_pos = loadPos - apron_start;

        data[shared_mem_pos] =
            (((loadPos >= apron_start_clamped) && (loadPos <= apron_end_clamped))
                 ? d_data[row_start + loadPos]
                 : 0);
    }

    __syncthreads();
    const int write_pos = tile_start + hipThreadIdx_x;

    if (write_pos <= tile_end_clamped)
    {
        const int shared_mem_pos = write_pos - apron_start;
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int a_sum = 0;
        
        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            r_sum +=
                (int)(((data[shared_mem_pos + k] >> 24) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            g_sum +=
                (int)(((data[shared_mem_pos + k] >> 16) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            b_sum +=
                (int)(((data[shared_mem_pos + k] >> 8) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            a_sum +=
                (int)(((data[shared_mem_pos + k]) & 0xff) *
                      d_kernel[KERNEL_RADIUS - k]);
        }
        d_result[row_start + write_pos] = (0 |
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
    int shared_mem_stride,
    int global_mem_stride)
{
    __shared__ uint32_t data[COLUMN_TILE_W *
                             (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    const int tile_start = (hipBlockIdx_y * COLUMN_TILE_H);
    const int tile_end = tile_start + COLUMN_TILE_H - 1;
    const int apron_start = tile_start - KERNEL_RADIUS;
    const int apron_end = tile_end + KERNEL_RADIUS;
    const int tile_end_clamped = min(tile_end, height - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, height - 1);

    const int col_start = (hipBlockIdx_x * COLUMN_TILE_W) + hipThreadIdx_x;

    int shared_mem_pos = (hipThreadIdx_y * COLUMN_TILE_W) + hipThreadIdx_x;
    int global_mem_pos = ((apron_start + hipThreadIdx_y) * width) + col_start;

    for (int y = apron_start + hipThreadIdx_y; y <= apron_end; y += hipBlockDim_y)
    {
        data[shared_mem_pos] =
            (((y >= apron_start_clamped) && (y <= apron_end_clamped))
                 ? d_data[global_mem_pos]
                 : 0);
        shared_mem_pos += shared_mem_stride;
        global_mem_pos += global_mem_stride;
    }

    __syncthreads();

    shared_mem_pos = ((hipThreadIdx_y + KERNEL_RADIUS) * COLUMN_TILE_W) + hipThreadIdx_x;
    global_mem_pos = ((tile_start + hipThreadIdx_y) * width) + col_start;

    for (int y = tile_start + hipThreadIdx_y; y <= tile_end_clamped; y += hipBlockDim_y)
    {
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int a_sum = 0;
        for (int k = -1 * KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
        {
            r_sum +=
                (int)(((data[shared_mem_pos + k * COLUMN_TILE_W] >> 24) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            g_sum +=
                (int)(((data[shared_mem_pos + k * COLUMN_TILE_W] >> 16) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            b_sum +=
                (int)(((data[shared_mem_pos + k * COLUMN_TILE_W] >> 8) & 0xff) * 
                d_kernel[KERNEL_RADIUS - k]);
            a_sum +=
                (int)(((data[shared_mem_pos + k * COLUMN_TILE_W]) & 0xff) *
                      d_kernel[KERNEL_RADIUS - k]);
        }
        d_result[global_mem_pos] = (0 |
                             ((0xff) << 24) |
                             ((g_sum & 0xff) << 16) |
                             ((b_sum & 0xff) << 8) |
                             (a_sum & 0xff));

        shared_mem_pos += shared_mem_stride;
        global_mem_pos += global_mem_stride;
    }
}


__global__ void g_brightness_one_channel(double *d_data, double *d_result, double delta,
                                         int width, int height)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int idx = y * width + x;

        if (delta > 0)
        {
            d_result[idx] = fmin(1.0, d_data[idx] + delta);
        }
        else
        {
            d_result[idx] = fmax(0, d_data[idx] + delta); 
        }
        
    }
}

__global__ void g_brightness_four_channel(uint32_t *d_data, uint32_t *d_result,
                                          char delta, int width, int height)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int idx = y * width + x;

        unsigned char r, g, b, a;
        if (delta > 0)
        {
            r = min(255, ((int)((d_data[idx]) & 0xff) + delta));
            g = min(255, ((int)((d_data[idx] >> 8) & 0xff) + delta));
            b = min(255, ((int)((d_data[idx] >> 16) & 0xff) + delta));
            a = ((d_data[idx] >> 24) & 0xff);
        }
        else
        {
            r = max(0, ((int)((d_data[idx]) & 0xff) + delta));
            g = max(0, ((int)((d_data[idx] >> 8) & 0xff) + delta));
            b = max(0, ((int)((d_data[idx] >> 16) & 0xff) + delta));
            a = ((d_data[idx] >> 24) & 0xff);
        }

        uint32_t result = a;
        result = result << 8;
        result = result | b;
        result = result << 8;
        result = result | g;
        result = result << 8;
        result = result | r;
        d_result[idx] = result;
    }
}


__global__ void g_colorize_four_channel(uint32_t *d_data, uint32_t *d_result,
                                          double r_mult, double g_mult, double b_mult, 
                                          int width, int height)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int idx = y * width + x;

        unsigned char r, g, b, a;

        r = (unsigned char)(min(255, (((double)(d_data[idx] & 0xff)) * r_mult)));
        g = (unsigned char)(min(255, (((double)((d_data[idx] >> 8) & 0xff)) * g_mult)));
        b = (unsigned char)(min(255, (((double)((d_data[idx] >> 16) & 0xff)) * b_mult)));
        a = ((d_data[idx] >> 24) & 0xff);

        uint32_t result = a;
        result = result << 8;
        result = result | b;
        result = result << 8;
        result = result | g;
        result = result << 8;
        result = result | r;
        d_result[idx] = result;
    }
}

/*******************************************************************************
* API FUNCTIONS WITH C-LINKAGE
*******************************************************************************/

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

    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        return _transpose<double>(obj_data);
    }
    else if (channels == 4)
    {
        return _transpose<uint32_t>(obj_data);
    }
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpimg_brightness(MPObjData *obj_data, void *args)
{
    double delta = ((BrightnessArgs *)args)->delta; 

    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        return _brightness_greyscale(obj_data, delta);
    }
    else if (channels == 4)
    {
        char delta_n = (char)(delta * 255);
        return _brightness_rgba(obj_data, delta_n);
    }
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpimg_colorize(MPObjData *obj_data, void *args)
{
    double r_mult = ((ColorizeArgs *)args)->r_mult;
    double g_mult = ((ColorizeArgs *)args)->g_mult; 
    double b_mult = ((ColorizeArgs *)args)->b_mult; 

    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        // No colorization for grey images
        return MILLIPYDE_SUCCESS;
    }
    else if (channels == 4)
    {
        return _colorize_rgba(obj_data, r_mult, g_mult, b_mult);
    }
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpimg_gaussian(MPObjData *obj_data, void *args)
{
    double sigma = ((GaussianArgs *)args)->sigma;

    // If we only have x,y dimensions, we are greyscale (one channel)
    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        return _gaussian_greyscale(obj_data, sigma);
    }
    else if (channels == 4) {
        return _gaussian_rgba(obj_data, sigma);
    }
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpimg_fliplr(MPObjData *obj_data, void *args)
{
    MP_UNUSED(args);

    // If we only have x,y dimensions, we are greyscale (one channel)
    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        return _fliplr<double>(obj_data);
    }
    else if (channels == 4) {
        return _fliplr<uint32_t>(obj_data);
    }
    return MILLIPYDE_SUCCESS;
}


MPStatus
mpimg_rotate(MPObjData *obj_data, void *args)
{
    MP_UNUSED(args);
    double angle = ((RotateArgs *)args)->angle;

    double angle_rad = angle * 0.01745329252;

    // If we only have x,y dimensions, we are greyscale (one channel)
    int channels = obj_data->ndims == 2 ? 1 : obj_data->dims[2];
    if (channels == 1)
    {
        return _rotate<double>(obj_data, angle_rad);
    }
    else if (channels == 4) {
        return _rotate<uint32_t>(obj_data, angle_rad);
    }
    return MILLIPYDE_SUCCESS;
}


} // extern "C"


/*******************************************************************************
* STATIC METHODS
*******************************************************************************/


static MPStatus 
_gaussian_greyscale(MPObjData *obj_data, double sigma)
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
        int dist = -1 * (KERNEL_RADIUS - i);
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
        dim3(ceil(width / (double)ROW_TILE_W), height, 1),
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
        dim3(ceil(width / (double)COLUMN_TILE_W), ceil(height / (double)COLUMN_TILE_H), 1),
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

static MPStatus 
_gaussian_rgba(MPObjData *obj_data, double sigma)
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
        int dist = -1 * (KERNEL_RADIUS - i);
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
            dim3(ceil(width / (double)ROW_TILE_W), height, 1),
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
            dim3(ceil(width / (double)COLUMN_TILE_W),
                 ceil(height / (double)COLUMN_TILE_H), 1),
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


template <typename T>  MPStatus 
_transpose(MPObjData *obj_data)
{
    int device_id;

    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    T *d_img;
    T *d_transpose;

    device_id = obj_data->mem_loc;
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = (hipStream_t)obj_data->stream;

    if (obj_data->device_data != NULL) {
        d_img = (T *)(obj_data->device_data);
    }
    else {
        //TODO
        return MILLIPYDE_SUCCESS;
    }

    HIP_CHECK(hipMalloc(&d_transpose, obj_data->nbytes));
    
    int temp = obj_data->dims[0];
    obj_data->dims[0] = obj_data->dims[1];
    obj_data->dims[1] = temp;
    obj_data->dims[obj_data->ndims] = obj_data->dims[0] * sizeof(T);
    obj_data->dims[obj_data->ndims + 1] = sizeof(T);

    hipLaunchKernelGGL(g_transpose, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM, 1),
            //TODO: Double check this
            TRANSPOSE_TILE_DIM * TRANSPOSE_TILE_DIM * obj_data->dims[obj_data->ndims + 1],
            stream,
            d_img,
            d_transpose,
            width,
            height);

    obj_data->device_data = d_transpose;

    HIP_CHECK(hipFree(d_img));
    
    return MILLIPYDE_SUCCESS;
}


template <typename T>  MPStatus 
_fliplr(MPObjData *obj_data)
{
    int device_id;

    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    T *d_img;
    T *d_flipped;

    device_id = obj_data->mem_loc;
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = (hipStream_t)obj_data->stream;

    if (obj_data->device_data != NULL) {
        d_img = (T *)(obj_data->device_data);
    }
    else {
        //TODO
        return MILLIPYDE_SUCCESS;
    }

    HIP_CHECK(hipMalloc(&d_flipped, obj_data->nbytes));

    hipLaunchKernelGGL(g_flip_horizontal, 
            dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
            dim3(FLIP_TILE_DIM, FLIP_TILE_DIM, 1),
            //TODO: Use coalesced shared memory in the future
            //FLIP_TILE_DIM * FLIP_TILE_DIM * sizeof(T) * 2,
            0,
            stream,
            d_img,
            d_flipped,
            width,
            height);

    obj_data->device_data = d_flipped;

    HIP_CHECK(hipFree(d_img));
    
    return MILLIPYDE_SUCCESS;
}


template <typename T>  MPStatus 
_rotate(MPObjData *obj_data, double angle)
{
    int device_id;

    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    T *d_img;
    T *d_rot;

    device_id = obj_data->mem_loc;
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = (hipStream_t)obj_data->stream;

    if (obj_data->device_data != NULL) {
        d_img = (T *)(obj_data->device_data);
    }
    else {
        //TODO
        return MILLIPYDE_SUCCESS;
    }

    HIP_CHECK(hipMalloc(&d_rot, obj_data->nbytes));

    hipLaunchKernelGGL(
        g_rotate,
        dim3(ceil(width / (float)ROTATE_BLOCK_DIM),
             ceil(height / (float)ROTATE_BLOCK_DIM), 1),
        dim3(ROTATE_BLOCK_DIM, ROTATE_BLOCK_DIM, 1),
        0,
        stream,
        d_img,
        d_rot,
        width,
        height,
        angle);

    obj_data->device_data = d_rot;

    HIP_CHECK(hipFree(d_img));
    
    return MILLIPYDE_SUCCESS;
}


static MPStatus 
_brightness_greyscale(MPObjData *obj_data, double delta)
{
    int device_id = obj_data->mem_loc;
    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    double *d_result;
    double *d_image = (double *)(obj_data->device_data);

    HIP_CHECK(hipSetDevice(device_id));
    hipStream_t stream = (hipStream_t)obj_data->stream;

    HIP_CHECK(hipMalloc(&d_result, obj_data->nbytes));

    hipLaunchKernelGGL(
        g_brightness_one_channel,
        dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
        dim3(32, 32, 1),
        0,
        stream,
        d_image,
        d_result,
        delta,
        width,
        height);

    obj_data->device_data = d_result;

    HIP_CHECK(hipFree(d_image));
    
    return MILLIPYDE_SUCCESS;
}


static MPStatus 
_brightness_rgba(MPObjData *obj_data, char delta)
{
    int device_id = obj_data->mem_loc;
    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    uint32_t *d_result;
    uint32_t *d_image = (uint32_t *)(obj_data->device_data);

    HIP_CHECK(hipSetDevice(device_id));
    hipStream_t stream = (hipStream_t)obj_data->stream;

    HIP_CHECK(hipMalloc(&d_result, obj_data->nbytes));

    hipLaunchKernelGGL(
        g_brightness_four_channel,
        dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
        dim3(32, 32, 1),
        0,
        stream,
        d_image,
        d_result,
        delta,
        width,
        height);

    obj_data->device_data = d_result;

    HIP_CHECK(hipFree(d_image));
    
    return MILLIPYDE_SUCCESS;
}


static MPStatus 
_colorize_rgba(MPObjData *obj_data, double r_mult, double g_mult, double b_mult)
{
    int device_id = obj_data->mem_loc;
    int height = obj_data->dims[0];
    int width = obj_data->dims[1];

    uint32_t *d_result;
    uint32_t *d_image = (uint32_t *)(obj_data->device_data);

    HIP_CHECK(hipSetDevice(device_id));
    hipStream_t stream = (hipStream_t)obj_data->stream;

    HIP_CHECK(hipMalloc(&d_result, obj_data->nbytes));

    hipLaunchKernelGGL(
        g_colorize_four_channel,
        dim3(ceil(width / 32.0), ceil(height / 32.0), 1),
        dim3(32, 32, 1),
        0,
        stream,
        d_image,
        d_result,
        r_mult,
        g_mult,
        b_mult,
        width,
        height);

    obj_data->device_data = d_result;

    HIP_CHECK(hipFree(d_image));
    
    return MILLIPYDE_SUCCESS;
}

