/**
 * main.cpp
 * 
 * Programa principal para benchmark de algoritmos FFT
 * 
 * Compilar con:
 * nvcc -O3 -arch=sm_86 main.cpp \ 
 *          fft_cpu.cpp fft_gpu.cu -o fft_comp \ 
 *          -lcufft -lfftw3f -lgomp -Xcompiler -fopenmp
 */

#include "commons.h"
#include <cuda_runtime.h>
#include <string.h>

#define NUM_REPS 3       // Repeticiones para promediar
#define MAX_EXP 27       // Hasta 2^27 (aprox límite de 6GB VRAM)

// --- Arreglo Maestro de Algoritmos ---
// Ordenados por bloques CPU/GPU, por eficiencia creciente
BenchmarkAlgo algorithms[] = {
    // CPU: slowest → fastest
    {"DFT",          DFT,           false},  // O(N²) serial
    {"DFT_omp",      DFT_omp,       false},  // O(N²) parallel
    {"FFT",          FFT,           false},  // O(N log N) recursivo
    {"FFTW3",        FFTW3,         false},  // O(N log N) librería

    // GPU: slowest → fastest
    {"cuDFT",        cuDFT,         true},   // O(N²) brute force
    {"cuFFT",        cuFFT_basic,   true},   // O(N log N) básico
    {"cuFFT_shfl",   cuFFT_shuffle, true},   // O(N log N) warp shuffle
    {"cuFFTW3",      cuFFTW3,       true}    // O(N log N) librería
};

// ¿No es mejor el número mágico 6？
const int num_algos = sizeof(algorithms) / sizeof(BenchmarkAlgo);

// --- Helpers de Memoria ---
void init_input(Complex* data, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        data[i].x = sinf(2.0f * (float)M_PI * i / 64.0f);
        data[i].y = 0.0f;
    }
}

// Función auxiliar para imprimir celdas alineadas
void print_cell(double t, FILE *csv) 
{
    if (t < 0.0) 
        printf("\t -      ");
    else         
        printf("\t%.6f", t);
    fflush(stdout);
    
    if (csv) {
        if (t < 0.0) 
            fprintf(csv, ",-1.0");
        else         
            fprintf(csv, ",%.6f", t);
    }   
}

int main() 
{
    const char* filename = "TiemposFFT.csv";
    FILE* csv = fopen(filename, "w");
    if (!csv) 
        printf("Error al abrir el archivo %s para escritura.\n", filename);

    bool fftActive[num_algos];
    for(int i=0; i<num_algos; i++) 
        fftActive[i] = true;

    // 1. Imprimir Encabezado Dinámico
    printf("N");
    for (int i = 0; i < num_algos; i++) 
        printf("\t%s\t", algorithms[i].label);
    printf("\n");

    //1.1 También al archivo CSV
    if (csv) {
        fprintf(csv, "N");
        for (int i = 0; i < num_algos; i++) 
            fprintf(csv, ",%s", algorithms[i].label);
    }
    fputc('\n', csv);

    // 2. Ciclo Principal (N)
    for (int exp = 14; exp <= MAX_EXP; exp++) {
        int n = 1 << exp;
        size_t size_bytes = n * sizeof(Complex);

        // --- Gestión de Memoria (Centralizada) ---
        Complex *h_in = (Complex*)malloc(size_bytes);
        Complex *h_out = (Complex*)malloc(size_bytes);
        
        // Verificación de seguridad para GPU antes de alloc
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        float *d_in = NULL, *d_out = NULL;
        bool gpu_mem_ok = false;

        // Necesitamos 2 arrays + 200MB de margen
        if (free_mem > (size_bytes * 2 + 200*1024*1024)) {
            cudaMalloc(&d_in, size_bytes);
            cudaMalloc(&d_out, size_bytes);
            gpu_mem_ok = (d_in != NULL && d_out != NULL);
        }

        // Si falló memoria CPU, abortar todo
        if (!h_in || !h_out) {
            printf("%d\t(Memoria Host Insuficiente)\n", n);
            if ( h_in ) 
                free( h_in ); 
            if ( h_out ) 
                free( h_out );
            if ( d_in ) 
                cudaFree( d_in ); 
            if ( d_out ) 
                cudaFree( d_out );
            break;
        }

        // Inicializar datos
        init_input(h_in, n);
        if (gpu_mem_ok) 
            cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice);

        printf("%d", exp);
        fflush(stdout);
        fprintf(csv, "%d", exp);

        for (int i = 0; i < num_algos; i++) {
            BenchmarkAlgo algo = algorithms[i];
            bool skip = false;

            if (!fftActive[i]) 
                skip = true;

            if (algo.is_gpu && !gpu_mem_ok) {
                fftActive[i] = false; 
                skip = true;
            }
            if (skip) {
                print_cell(-1.0, csv);
                continue;
            }

            float *ptr_in  = algo.is_gpu ? (float*)d_in : (float*)h_in;
            float *ptr_out = algo.is_gpu ? (float*)d_out : (float*)h_out;

            double t_total = 0.0;
            int valid_runs = 0;

            for (int r = 0; r < NUM_REPS; r++) {
                double t = algo.func(ptr_in, ptr_out, n);
                
                if (t < 0.0) 
                    break; // Error interno en la función
                t_total += t;
                valid_runs++;
            }

            if (valid_runs > 0) {
                double avg_time = t_total / valid_runs;
                print_cell(avg_time, csv);

                
                if (avg_time > 2.0) // Tardó más de 2 segundos, se muere
                    fftActive[i] = false;
            } else {
                
                print_cell(-1.0, csv); // Falló internamente, se muere
                fftActive[i] = false;
            }
        }
        printf("\n");
        if (csv) 
            fputc('\n', csv);

        free(h_in); 
        free(h_out);
        if ( d_in ) 
            cudaFree(d_in);
        if (d_out) 
            cudaFree(d_out);
    }

    fclose(csv);
    return 0;
}
