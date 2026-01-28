#include "commons.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#define PI 3.14159265358979323846f

__global__ void dft_kernel(float2 *in, float2 *out, int n) 
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        for (int j = 0; j < n; j++) {
            float angle = -2.0f * PI * ((float)k * j) / (float)n;
            float s, c;
            __sincosf(angle, &s, &c);
            sum_r += in[j].x * c - in[j].y * s;
            sum_i += in[j].x * s + in[j].y * c;
        }
        out[k].x = sum_r;
        out[k].y = sum_i;
    }
}

double cuDFT(float* d_in_ptr, float* d_out_ptr, int n) 
{
    float2* d_in = (float2*)d_in_ptr;
    float2* d_out = (float2*)d_out_ptr;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dft_kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    
    return (double)ms / 1000.0;
}

// La mejor opción con cuda
double cuFFTW3(float* d_in_ptr, float* d_out_ptr, int n) 
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem < (100 * 1024 * 1024)) 
        return -1.0; // Mínimo 100MB libres

    cufftComplex* d_in = (cufftComplex*)d_in_ptr;
    cufftComplex* d_out = (cufftComplex*)d_out_ptr;

    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        return -1.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cufftDestroy(plan);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    
    return (double)ms / 1000.0;
}
