# Makefile - Benchmark FFT
# Autor: Miguel

# Ajusta -arch=sm_86 según tu GPU (Ampere/RTX 30xx usa sm_86)
ARCH = -arch=sm_86 
CFLAGS = -O3 -Xcompiler -fopenmp 
LDFLAGS = -lcufft -lfftw3f -lm -lgomp

TARGET = fft_comp
SRCS = main.cpp fft_cpu.cpp fft_gpu.cu
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	nvcc $(ARCH) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	nvcc $(ARCH) $(CFLAGS) -c $< -o $@

%.o: %.cu
	nvcc $(ARCH) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
