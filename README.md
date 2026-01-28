# FFT Benchmark: CPU vs GPU (Complexity Analysis)

Este proyecto realiza una comparación de rendimiento (benchmarking) entre diferentes implementaciones de la Transformada de Fourier, contrastando algoritmos de fuerza bruta $O(N^2)$ contra algoritmos optimizados $O(N \log N)$, tanto en CPU como en GPU (CUDA).

## Algoritmos Comparados

1.  **DFT:** Implementación clásica $O(N^2)$ en CPU.
2.  **DFTomp:** Paralelización simple de la DFT en CPU con OpenMP.
3.  **cuDFT:** Kernel de fuerza bruta $O(N^2)$ en GPU.
4.  **FFT:** Implementación Cooley-Tukey Naive $O(N \log N)$ en CPU.
5.  **FFTW3:** Librería estándar optimizada para CPU.
6.  **cuFFTW3:** Librería NVIDIA cuFFT optimizada para GPU.

## Requisitos

* NVIDIA CUDA Toolkit (nvcc)
* Librería FFTW3 (`libfftw3-dev` / `libfftw3-single3`)
* Compilador con soporte OpenMP
* R (con `ggplot2`) para la visualización

## Compilación y Ejecución

```bash
make
./bench_final
```

(c) UABCS/DASC/manc