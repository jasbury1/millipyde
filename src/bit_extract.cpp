/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

#include "GPUKernels.h"
#include "millipyde_hip_util.h"

__global__ void bit_extract_kernel(uint32_t* C_d, const uint32_t* A_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
#ifdef __HIP_PLATFORM_HCC__
        C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVCC__ or other path */
        C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
    }
}

__global__ void add_one_kernel(long* data_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        data_d[i] = data_d[i] + 1;
    }
}

__global__ void color_to_greyscale_kernel(unsigned char * greyImg, 
        unsigned char * rgbImg, int width, int height)
{
    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    if (x < width && y < height) {
        int greyOffset = y * width + x;

        // 3 is for the 3 channels in rgb
        int rgbOffset = greyOffset * 3;
        unsigned char r = rgbImg[rgbOffset];
        unsigned char g = rgbImg[rgbOffset + 1];
        unsigned char b = rgbImg[rgbOffset + 2];
        greyImg[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

extern "C" {

int run_bit_extract() {
    uint32_t *A_d, *C_d;
    uint32_t *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(uint32_t);

#ifdef __HIP_ENABLE_PCH
    // Verify hip_pch.o
    const char* pch = nullptr;
    unsigned int size = 0;
    __hipGetPCH(&pch, &size);
    printf("pch size: %u\n", size);
    if (size == 0) {
        printf("__hipGetPCH failed!\n");
        return -1;
    } else {
        printf("__hipGetPCH succeeded!\n");
    }
#endif

    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
    printf("info: running on device #%d %s\n", deviceId, props.name);


    printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    A_h = (uint32_t*)malloc(Nbytes);
    HIP_CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    C_h = (uint32_t*)malloc(Nbytes);
    HIP_CHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

    for (size_t i = 0; i < N; i++) {
        A_h[i] = i;
    }

    printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    HIP_CHECK(hipMalloc(&A_d, Nbytes));
    HIP_CHECK(hipMalloc(&C_d, Nbytes));

    printf("info: copy Host2Device\n");
    HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    printf("info: launch 'bit_extract_kernel' \n");
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL(bit_extract_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

    printf("info: copy Device2Host\n");
    HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    printf("info: check result\n");
    for (size_t i = 0; i < N; i++) {
        unsigned Agold = ((A_h[i] & 0xf00) >> 8);
        if (C_h[i] != Agold) {
            fprintf(stderr, "mismatch detected.\n");
            printf("%zu: %08x =? %08x (Ain=%08x)\n", i, C_h[i], Agold, A_h[i]);
            HIP_CHECK(hipErrorUnknown);
        }
    }
    printf("PASSED!\n");
    return 0;
}



int add_one(void * data, int N)
{
    size_t Nbytes = N * sizeof(long);
    long *data_d;

    HIP_CHECK(hipMalloc(&data_d, Nbytes));
    printf("info: copy Host2Device\n");
    HIP_CHECK(hipMemcpy(data_d, data, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    printf("info: launch 'add_one_kernel' \n");
    hipLaunchKernelGGL(add_one_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, data_d, N);
    
    printf("info: copy Device2Host\n");
    HIP_CHECK(hipMemcpy(data, data_d, Nbytes, hipMemcpyDeviceToHost));
    return 0;
}

/*
int color_to_greyscale(void * source, void * destination, size_t Nbytes, int width, int height)
{
    unsigned char *rgbImg;
    unsigned char *greyImg;
    HIP_CHECK(hipMalloc(&rgbImg, Nbytes));
    HIP_CHECK(hipMalloc(&greyImg, Nbytes / 3));

    HIP_CHECK(hipMemcpy(rgbImg, source, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(color_to_greyscale_kernel, 
            dim3(ceil(width / 16.0), ceil(height/16.0), 1),
            dim3(16, 16, 1)
            0,
            0,
            greyImg,
            rgbImg,
            width,
            height);

    HIP_CHECK(hipMemcpy(destination, greyImg, Nbytes, hipMemcpyDeviceToHost)); 
}
*/


} // extern "C"
