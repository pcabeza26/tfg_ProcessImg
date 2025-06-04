import cv2
import numpy as np

# Leer la imagen en escala de grises (2D)
img = cv2.imread("C:/Users/pcabe/tfg/imagenes/heman_sup.jpg", cv2.IMREAD_GRAYSCALE)

# Convertir a uint16 con escala
img_uint16 = np.uint16(img) * 256  # 0-255 → 0-65535

# Ver rango para depuración
print("Mín:", img_uint16.min(), "Máx:", img_uint16.max())

# Guardar en formato binario crudo
img_uint16.tofile("bin/23_03_2023__12_31_26_ir.bin")
