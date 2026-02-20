/***
 * fft_gpu.cu
 * Implementaciones de FFT en GPU usando CUDA
 * Se inclueyen:
 * - cuDFT: DFT directo sin optimizaciones
 * - cuFFTW3: Wrapper para la biblioteca cuFFT de NVIDIA
 * - cuFFT_basic: FFT iterativa básica sin optimizaciones
 * - cuFFT_shuffle: FFT optimizada con warp shuffle y shared memory
 */


#include "commons.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#define PI 3.14159265358979323846f

// Kernel para la versión directa de la DFT sin optimizaciones
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

// Versión directa de la DFT sin optimizaciones
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

// Versión de uso industrial, la mejor
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

// ============================================================
// Utilidades compartidas para FFTs
// ============================================================

__device__ int bit_reverse(int x, int bits)
{
    int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__global__ void bit_reverse_copy_kernel(float2* in, float2* out, int n, int log2N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int rev = bit_reverse(tid, log2N);
        out[tid] = in[rev];
    }
}

// Kernel para cuFFT_basic
__global__ void butterfly_stage_basic_kernel(float2* src, float2* dst, int n, int stage)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int stride = 1 << stage;      // 2^stage
    int group_size = stride << 1; // 2^(stage+1)
    int j = tid & (stride - 1);   // posición en medio-grupo
    int partner = tid ^ stride;   // XOR para obtener pareja butterfly

    // Twiddle factor: w = exp(-2πij/group_size)
    float angle = -2.0f * PI * j / (float)group_size;
    float tw_re, tw_im;
    __sincosf(angle, &tw_im, &tw_re);

    float2 my_val = src[tid];
    float2 pv = src[partner];

    float2 result;
    if ((tid & stride) == 0) {
        // Top: even + w*odd
        float t_re = pv.x * tw_re - pv.y * tw_im;
        float t_im = pv.x * tw_im + pv.y * tw_re;
        result.x = my_val.x + t_re;
        result.y = my_val.y + t_im;
    } else {
        // Bottom: even - w*odd
        float t_re = my_val.x * tw_re - my_val.y * tw_im;
        float t_im = my_val.x * tw_im + my_val.y * tw_re;
        result.x = pv.x - t_re;
        result.y = pv.y - t_im;
    }

    dst[tid] = result;
}

// Versión iterativa básica de la FFT sin optimizaciones
// Equivalente a fft_rec_naive() de CPU, pero iterativo
double cuFFT_basic(float* d_in_ptr, float* d_out_ptr, int n)
{
    float2* d_in = (float2*)d_in_ptr;
    float2* d_out = (float2*)d_out_ptr;

    // Calcular log2(n)
    int log2N = 0;
    for (int tmp = n; tmp > 1; tmp >>= 1) log2N++;

    // Allocar buffer temporal para ping-pong
    float2* d_temp;
    cudaMalloc(&d_temp, n * sizeof(float2));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Paso 1: Bit-reverse permutation (d_in → d_temp)
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bit_reverse_copy_kernel<<<blocks, threads>>>(d_in, d_temp, n, log2N);

    // Paso 2: Iterative butterfly stages con ping-pong
    float2* src = d_temp;
    float2* dst = d_out;

    for (int stage = 0; stage < log2N; stage++) {
        butterfly_stage_basic_kernel<<<blocks, threads>>>(src, dst, n, stage);

        // Swap src/dst para próxima etapa
        float2* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Resultado está en 'src' después del último swap
    // Si log2N es impar: src=d_out ✓
    // Si log2N es par: src=d_temp → copiar a d_out
    if (log2N % 2 == 0) {
        cudaMemcpy(d_out, d_temp, n * sizeof(float2), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)ms / 1000.0;
}


// FFT con Warp Shuffle — Cooley-Tukey radix-2 en GPU

// Kernel principal: sub-FFT de hasta 1024 puntos por bloque
// Etapas 0-4: warp shuffle | Etapas 5-9: shared memory
__global__ void fft_shuffle_kernel(float2* in, float2* out, int n,
                                   int block_fft_size, int log2_block)
{
    extern __shared__ float2 smem[];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * block_fft_size;

    // Carga lineal (los datos ya están en orden bit-reversed)
    float2 val = in[block_offset + tid];
    float val_re = val.x;
    float val_im = val.y;

    // --- Etapas warp shuffle (0..min(4, log2_block-1)) ---
    int max_shuffle_stage = (log2_block < 5) ? log2_block : 5;
    for (int s = 0; s < max_shuffle_stage; s++) {
        int stride = 1 << s;
        int group_size = stride << 1;
        int j = tid & (stride - 1);

        float angle = -2.0f * PI * j / (float)group_size;
        float tw_re, tw_im;
        __sincosf(angle, &tw_im, &tw_re);

        float partner_re = __shfl_xor_sync(0xFFFFFFFF, val_re, stride);
        float partner_im = __shfl_xor_sync(0xFFFFFFFF, val_im, stride);

        if ((tid & stride) == 0) {
            // Top: val + twiddle * partner
            float t_re = partner_re * tw_re - partner_im * tw_im;
            float t_im = partner_re * tw_im + partner_im * tw_re;
            val_re += t_re;
            val_im += t_im;
        } else {
            // Bottom: partner - twiddle * val
            float t_re = val_re * tw_re - val_im * tw_im;
            float t_im = val_re * tw_im + val_im * tw_re;
            val_re = partner_re - t_re;
            val_im = partner_im - t_im;
        }
    }

    // --- Etapas shared memory (5..log2_block-1) ---
    if (log2_block > 5) {
        smem[tid] = make_float2(val_re, val_im);
        __syncthreads();

        for (int s = 5; s < log2_block; s++) {
            int stride = 1 << s;
            int group_size = stride << 1;
            int j = tid & (stride - 1);
            int partner = tid ^ stride;

            float angle = -2.0f * PI * j / (float)group_size;
            float tw_re, tw_im;
            __sincosf(angle, &tw_im, &tw_re);

            float2 my_val = smem[tid];
            float2 pv = smem[partner];
            __syncthreads();

            float2 result;
            if ((tid & stride) == 0) {
                float t_re = pv.x * tw_re - pv.y * tw_im;
                float t_im = pv.x * tw_im + pv.y * tw_re;
                result.x = my_val.x + t_re;
                result.y = my_val.y + t_im;
            } else {
                float t_re = my_val.x * tw_re - my_val.y * tw_im;
                float t_im = my_val.x * tw_im + my_val.y * tw_re;
                result.x = pv.x - t_re;
                result.y = pv.y - t_im;
            }

            smem[tid] = result;
            __syncthreads();
        }

        out[block_offset + tid] = smem[tid];
    } else {
        out[block_offset + tid] = make_float2(val_re, val_im);
    }
}

// Kernel global: una etapa butterfly sobre el array completo (ping-pong)
__global__ void fft_global_stage_kernel(float2* src, float2* dst, int n, int stage)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int stride = 1 << stage;
    int group_size = stride << 1;
    int j = tid & (stride - 1);
    int partner = tid ^ stride;

    float angle = -2.0f * PI * j / (float)group_size;
    float tw_re, tw_im;
    __sincosf(angle, &tw_im, &tw_re);

    float2 my_val = src[tid];
    float2 pv = src[partner];

    float2 result;
    if ((tid & stride) == 0) {
        float t_re = pv.x * tw_re - pv.y * tw_im;
        float t_im = pv.x * tw_im + pv.y * tw_re;
        result.x = my_val.x + t_re;
        result.y = my_val.y + t_im;
    } else {
        float t_re = my_val.x * tw_re - my_val.y * tw_im;
        float t_im = my_val.x * tw_im + my_val.y * tw_re;
        result.x = pv.x - t_re;
        result.y = pv.y - t_im;
    }

    dst[tid] = result;
}

// Wrapper: orquesta los kernels y mide tiempo
double cuFFT_shuffle(float* d_in_ptr, float* d_out_ptr, int n)
{
    float2* d_in  = (float2*)d_in_ptr;
    float2* d_out = (float2*)d_out_ptr;

    int log2N = 0;
    for (int tmp = n; tmp > 1; tmp >>= 1) log2N++;

    int block_fft_size = (n < 1024) ? n : 1024;
    int log2_block = (log2N < 10) ? log2N : 10;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 1. Bit-reverse copy: d_in → d_out
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        bit_reverse_copy_kernel<<<blocks, threads>>>(d_in, d_out, n, log2N);
    }

    // 2. Sub-FFTs por bloques: d_out → d_in
    {
        int threads = block_fft_size;
        int blocks = n / block_fft_size;
        int smem_bytes = (log2_block > 5) ? block_fft_size * sizeof(float2) : 0;
        fft_shuffle_kernel<<<blocks, threads, smem_bytes>>>(
            d_out, d_in, n, block_fft_size, log2_block);
    }

    // 3. Etapas globales (10..log2N-1) con ping-pong
    int global_stages = log2N - log2_block;
    float2* src = d_in;
    float2* dst = d_out;
    for (int s = log2_block; s < log2N; s++) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        fft_global_stage_kernel<<<blocks, threads>>>(src, dst, n, s);
        float2* tmp = src; src = dst; dst = tmp;
    }

    // Si el resultado quedó en d_in, copiarlo a d_out
    // Después del ping-pong, src apunta al resultado.
    // Si global_stages == 0: resultado en d_in → copiar
    // Si global_stages impar: resultado en d_out → ok
    // Si global_stages par (>0): resultado en d_in → copiar
    if (global_stages == 0 || (global_stages % 2 == 0)) {
        cudaMemcpy(d_out, d_in, n * sizeof(float2), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)ms / 1000.0;
}
