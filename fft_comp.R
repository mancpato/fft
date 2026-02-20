# Instalar paquetes si no los tienes: ----
# install.packages(c("ggplot2", "dplyr", "tidyr", "scales"))

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# 1. Cargar datos ----
data <- read.csv("TiemposFFT.csv")

# 2. Transformar a formato "Largo" (Tidy Data) para ggplot ----
#    y filtrar los valores -1.0 (que indican "No ejecutado")
data_long <- data %>%
  pivot_longer(
    cols = -N, 
    names_to = "Algoritmo", 
    values_to = "Tiempo"
  ) %>%
  filter(Tiempo > 0) # Eliminamos los códigos de error/salto

# 3. Ordenar los factores para que la leyenda salga bonita ----
#    IMPORTANTE: Debe coincidir EXACTAMENTE con el orden del CSV
data_long$Algoritmo <- factor(data_long$Algoritmo,
                              levels = c("DFT", "DFT_omp", "FFT", "FFTW3",
                                         "cuDFT", "cuFFT_basic", "cuFFT_shfl", "cuFFTW3"))

# 4. Graficar ----
p <- ggplot(data_long, aes(x = N, y = Tiempo, color = Algoritmo, shape = Algoritmo)) +
  # Líneas y Puntos
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_log10(
    breaks = trans_breaks("log2", function(x) 2^x),
    labels = trans_format("log2", math_format(2^.x))
  ) +
  scale_y_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  
  # Colores personalizados (CPU: rojos/naranjas/verdes/azules, GPU: púrpuras/marrones/rosas/negro)
  scale_color_manual(values = c(
    "DFT"          = "#E41A1C",  # Rojo (CPU más lento)
    "DFT_omp"      = "#FF7F00",  # Naranja
    "FFT"          = "#4DAF4A",  # Verde
    "FFTW3"        = "#377EB8",  # Azul (CPU más rápido)
    "cuDFT"        = "#984EA3",  # Púrpura (GPU más lento)
    "cuFFT_basic"  = "#A65628",  # Marrón (GPU básico)
    "cuFFT_shfl"   = "#F781BF",  # Rosa (GPU optimizado)
    "cuFFTW3"      = "#000000"   # Negro (GPU más rápido)
  )) +
  
  # Etiquetas y Tema
  labs(
    title = "Benchmark Transformada de Fourier: CPU vs GPU",
    subtitle = "Comparación de 8 algoritmos: O(N²) vs O(N log N), básicos vs optimizados",
    x = "Tamaño de la Señal (N)",
    y = "Tiempo de Ejecución (segundos) [Escala Log]",
    caption = "Nota: Ejes en escala logarítmica. Valores faltantes indican timeout > 2s."
  ) +
  theme_bw() +
  theme(
    legend.position = c(0.15, 0.75), # Leyenda flotante adentro a la izquierda
    legend.background = element_rect(fill=alpha('white', 0.8), color="black"),
    panel.grid.minor = element_blank() # Limpiar un poco la rejilla
  )

# 5. Mostrar y Guardar ----
print(p)
ggsave("FT_comp.png", plot = p, width = 10, height = 6, dpi = 300)
