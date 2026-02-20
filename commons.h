#ifndef COMMONS_H
#define COMMONS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct { 
    float x; 
    float y; 
} Complex;

// Prototipo: regresa el tiempo en segundos, -1.0 si no se ejecuta.
typedef double (*BenchFunc)(float* in, float* out, int n);

// Estructura de Metadatos del Algoritmo ---
typedef struct {
    const char* label;   // Nombre para la columna (ej: "DFT")
    BenchFunc func;      // Función fft
    bool is_gpu;         // true = usa punteros de device (d_in)
} BenchmarkAlgo;

// CPU
double DFT(float* in, float* out, int n);
double DFT_omp(float* in, float* out, int n);
double FFT(float* in, float* out, int n);
double FFTW3(float* in, float* out, int n);

// GPU
double cuDFT(float* in, float* out, int n);         // DFT Manual O(N²)
double cuFFT_basic(float* in, float* out, int n);   // FFT básico sin optimizaciones
double cuFFT_shuffle(float* in, float* out, int n); // FFT con warp shuffle
double cuFFTW3(float* in, float* out, int n);       // Librería cuFFT

#endif // COMMONS_H
