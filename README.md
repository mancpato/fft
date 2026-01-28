# FFT Benchmark: CPU vs GPU (Complexity Analysis)
[![Makefile CI](https://github.com/mancpato/fft/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/mancpato/fft/actions/workflows/cmake-multi-platform.yml)

Este proyecto realiza una comparación de rendimiento (benchmarking) entre diferentes implementaciones de la Transformada de Fourier, contrastando algoritmos de fuerza bruta $O(N^2)$ contra algoritmos optimizados $O(N \log N)$, tanto en CPU como en GPU (CUDA).

La salida se presenta tanto en pantalla como en en un archivo CSV para su análisis posterior. El nombre del archivo es estático: cada ejecución sobreescribe el anterior.

Proyecto elaborado como material de trabajo en los cursos de Análisis de Fourier y Análisis de Algoritmos del Departamento Académico de Sistemas Computacionales de la Universidad Autónoma de Baja California Sur.

## Algoritmos Comparados

1.  **DFT:** Implementación clásica $O(N^2)$ en CPU.
2.  **DFTomp:** Paralelización simple de la DFT en CPU con OpenMP.
3.  **cuDFT:** Kernel de fuerza bruta $O(N^2)$ en GPU.
4.  **FFT:** Implementación Cooley-Tukey Naive $O(N \log N)$ en CPU.
5.  **FFTW3:** Librería estándar optimizada para CPU.
6.  **cuFFTW3:** Librería NVIDIA cuFFT optimizada para GPU.

## Requisitos
Este proyecto compila en Linux con nvcc (CUDA) empleando librerías diversas.

* NVIDIA CUDA Toolkit (nvcc)
* Librería FFTW3 (`libfftw3-dev` / `libfftw3-single3`)
* Compilador con soporte OpenMP
* R (con `ggplot2`) para la visualización (opcional)

## Compilación y Ejecución

```bash
make
./fft_comp
```
## Licencia
Este proyecto es software libre: puedes redistribuirlo y/o modificarlo bajo los términos de la **Licencia Pública General GNU (GPL)** publicada por la Free Software Foundation, ya sea la versión 3 de la Licencia, o (a tu elección) cualquier versión posterior.

Consulta el archivo [LICENSE](LICENSE) para más detalles.

(?) UABCS/DASC/manc 2026
