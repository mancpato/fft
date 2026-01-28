# Instalar paquetes si no los tienes:
# install.packages(c("ggplot2", "dplyr", "tidyr", "scales"))

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# 1. Cargar datos
data <- read.csv("TiemposFFT.csv")

# 2. Transformar a formato "Largo" (Tidy Data) para ggplot
#    y filtrar los valores -1.0 (que indican "No ejecutado")
data_long <- data %>%
  pivot_longer(
    cols = -N, 
    names_to = "Algoritmo", 
    values_to = "Tiempo"
  ) %>%
  filter(Tiempo > 0) # Eliminamos los códigos de error/salto

# 3. Ordenar los factores para que la leyenda salga bonita
#    (Opcional: define el orden de aparición)
data_long$Algoritmo <- factor(data_long$Algoritmo, 
                              levels = c("DFT", "DFT_omp", "cuDFT", 
                                         "FFT", "FFTW3", "cuFFTW3"))

# 4. Graficar
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
  
  # Colores personalizados 
  scale_color_manual(values = c(
    "DFT" = "#E41A1C",      # Rojo (Lento)
    "DFT_omp" = "#FF7F00",  # Naranja
    "cuDFT" = "#984EA3",    # Morado (GPU Fuerza bruta)
    "FFT" = "#4DAF4A",      # Verde (Recursiva)
    "FFTW3" = "#377EB8",    # Azul (Pro CPU)
    "cuFFTW3" = "#000000"   # Negro (Pro GPU - El rey)
  )) +
  
  # Etiquetas y Tema
  labs(
    title = "Benchmark Transformada de Fourier: CPU vs GPU",
    subtitle = "Comparación de algoritmos O(N^2) vs O(N log N)",
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

# 5. Mostrar y Guardar
print(p)
ggsave("FT_comp.png", plot = p, width = 10, height = 6, dpi = 300)
