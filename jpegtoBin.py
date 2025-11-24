import cv2 as cv
import numpy as np

# 1️⃣ Cargar imagen JPEG en escala de grises
img = cv.imread('imagenes/heman_sup.JPG', cv.IMREAD_GRAYSCALE)  # 8 bits, 1 canal

# 2️⃣ Convertir a 16 bits simulados
# Escalamos de 0-255 a 0-65535
img_16bit = np.uint16(img) * 257  # 255 * 257 ≈ 65535

# 3️⃣ Guardar como .bin (RAW)
img_16bit.tofile('mi_foto_fake_raw.bin')
