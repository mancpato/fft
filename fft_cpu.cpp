/**
 * fft_cpu.cpp
 * Implementaciones CPU de DFT/FFT para benchmarking
 * Incluye:
 * - DFT directa 
 * - DFT con OpenMP
 * - FFT recursiva naive
 * - FFTW3, de uso industrial, la mejor opción CPU
 */

#include "commons.h"
#include <fftw3.h>
#include <omp.h>
#include <math.h>
#include <string.h> // memcpy

// Versión directa de la DFT, sin optimizaciones
double DFT(float* in, float* out, int n) 
{
    double start = omp_get_wtime();
    
    // Punteros tipo Complex para aritmética fácil
    Complex* x = (Complex*)in;
    Complex* X = (Complex*)out;

    for (int k = 0; k < n; k++) {
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        for (int t = 0; t < n; t++) {
            // Ángulo negativo para Forward DFT
            float angle = -2.0f * (float)M_PI * t * k / n;
            float c = cosf(angle);
            float s = sinf(angle);
            
            // (a + ib) * (c + is) = (ac - bs) + i(as + bc)
            sum_r += x[t].x * c - x[t].y * s;
            sum_i += x[t].x * s + x[t].y * c;
        }
        X[k].x = sum_r;
        X[k].y = sum_i;
    }
    
    return omp_get_wtime() - start;
}

// Versión paralelizada de la DFT con OpenMP
double DFT_omp(float* in, float* out, int n) 
{
    double start = omp_get_wtime();
    Complex* x = (Complex*)in;
    Complex* X = (Complex*)out;

    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        for (int t = 0; t < n; t++) {
            float angle = -2.0f * (float)M_PI * t * k / n;
            float c = cosf(angle);
            float s = sinf(angle);
            sum_r += x[t].x * c - x[t].y * s;
            sum_i += x[t].x * s + x[t].y * c;
        }
        X[k].x = sum_r;
        X[k].y = sum_i;
    }
    
    return omp_get_wtime() - start;
}

// Helper para FFT Recursiva (Cooley-Tukey Naive) ---
void fft_recursive_core(Complex* buffer, int n, int step) 
{
    if (n < 2) return;

    // Divide: Separar pares e impares lógicamente usando 'step'
    // (Esta versión es in-place sobre el buffer, más eficiente que mallocs constantes)
    fft_recursive_core(buffer, n/2, 2*step);        // Pares
    fft_recursive_core(buffer + step, n/2, 2*step); // Impares

    // Combine: Mariposa
    for (int k = 0; k < n/2; k++) {
        Complex* even = buffer + 2*step*k;
        Complex* odd  = buffer + 2*step*k + step;

        float angle = -2.0f * (float)M_PI * k / n;
        float c = cosf(angle);
        float s = sinf(angle);

        // t = w * odd
        float t_re = odd->x * c - odd->y * s;
        float t_im = odd->x * s + odd->y * c;

        // odd = even - t
        odd->x = even->x - t_re;
        odd->y = even->y - t_im;

        // even = even + t
        even->x = even->x + t_re;
        even->y = even->y + t_im;
    }
}

// Versión recursiva simple
void fft_rec_naive(Complex* in, Complex* out, int n) 
{
    if (n == 1) {
        out[0] = in[0];
        return;
    }

    Complex* even_in  = (Complex*)malloc(sizeof(Complex) * n/2);
    Complex* odd_in   = (Complex*)malloc(sizeof(Complex) * n/2);
    Complex* even_out = (Complex*)malloc(sizeof(Complex) * n/2);
    Complex* odd_out  = (Complex*)malloc(sizeof(Complex) * n/2);

    for (int i = 0; i < n/2; i++) {
        even_in[i] = in[2*i];
        odd_in[i]  = in[2*i + 1];
    }

    fft_rec_naive(even_in, even_out, n/2);
    fft_rec_naive(odd_in, odd_out, n/2);

    // Se combinan los resultados
    for (int k = 0; k < n/2; k++) {
        float angle = -2.0f * (float)M_PI * k / n;
        float c = cosf(angle);
        float s = sinf(angle);

        // t = exp(-2pi*i*k/N) * odd_out[k]
        float t_re = odd_out[k].x * c - odd_out[k].y * s;
        float t_im = odd_out[k].x * s + odd_out[k].y * c;

        out[k].x = even_out[k].x + t_re;
        out[k].y = even_out[k].y + t_im;

        out[k + n/2].x = even_out[k].x - t_re;
        out[k + n/2].y = even_out[k].y - t_im;
    }

    free(even_in);  free(odd_in); 
    free(even_out); free(odd_out);
}

// Versión recursiva simple con OpenMP en la fase de combinación
double FFT(float* in, float* out, int n) 
{
    //if (n > 65536) return -1.0; // Límite por stack/memoria recursiva

    double start = omp_get_wtime();
    
    // Llamamos a la implementación naive con mallocs
    fft_rec_naive((Complex*)in, (Complex*)out, n);
    
    return omp_get_wtime() - start;
}


// La mejor
double FFTW3(float* in, float* out, int n) 
{
    fftwf_complex* fin = (fftwf_complex*)in;
    fftwf_complex* fout = (fftwf_complex*)out;
    
    // ESTIMATE para no tardar años planificando
    fftwf_plan p = fftwf_plan_dft_1d(n, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);
    
    double start = omp_get_wtime();
    fftwf_execute(p);
    double end = omp_get_wtime();
    
    fftwf_destroy_plan(p);
    return end - start;
}
