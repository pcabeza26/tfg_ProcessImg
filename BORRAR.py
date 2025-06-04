# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:37:42 2025
Procesamiento Imagenes Clínicas
@author: pcabe
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from funciones_SP import rgb2luv
from funciones_SP import otsun
from funciones_SP import escalaImg

from scipy import signal
from scipy.stats import skew 
from scipy.stats import kurtosis 

import pandas as pd


# Lectura Imagen
l_img = cv2.imread("C:/Users/pcabe/tfg/imagenes/heman_sup.jpg")

# Escalado de imagen
factor = 0.55
s_img = escalaImg(l_img, factor)

# RECORTE
x, y, w, h = cv2.selectROI("Seleccionar Seccion",s_img, showCrosshair=True, fromCenter=False)
r_img = s_img[y:y+h,x:x+w]
cv2.imshow("Imagen Recortada", r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalización
img = r_img / 255.0

###########################################
## TRATAMIENTO DEL ORIGINAL

s_img = s_img / 255.0
imgR = s_img[:, :, 2]
imgG = s_img[:, :, 1]
imgB = s_img[:, :, 0]

filas = s_img.shape[0]
col = s_img.shape[1]
M = filas * col

imgR_vect = np.reshape(imgR, (M, 1))
imgG_vect = np.reshape(imgG, (M, 1))
imgB_vect = np.reshape(imgB, (M, 1))

RGB = np.column_stack((imgR_vect, imgG_vect, imgB_vect))

[L, u, v, teta, saturation] = rgb2luv(RGB)

###########################################
## TRATAMIENTO DEL RECORTE

imgRr = img[:, :, 2]
imgGr = img[:, :, 1]
imgBr = img[:, :, 0]

fil_recorte = img.shape[0]
col_recorte = img.shape[1]
Mr = fil_recorte * col_recorte

imgRr_vect = np.reshape(imgRr, (Mr, 1))
imgGr_vect = np.reshape(imgGr, (Mr, 1))
imgBr_vect = np.reshape(imgBr, (Mr, 1))

RGBr = np.column_stack((imgRr_vect, imgGr_vect, imgBr_vect))

[Lr, ur, vr, tetar, saturationr] = rgb2luv(RGBr)

L_cent = np.mean(Lr)
u_cent = np.mean(ur)
v_cent = np.mean(vr)

###########################################
## IMAGEN DE DISTANCIAS

dist = np.sqrt((Lr - L_cent) ** 2 + (ur - u_cent) ** 2 + (vr - v_cent) ** 2)
dist_fig = np.reshape(dist, (fil_recorte, col_recorte))
dist_fig = (dist_fig / np.max(dist_fig))

cv2.imshow("Imagen de distancias", escalaImg(dist_fig.copy(), 1.25))

dist_fig_med = signal.medfilt2d(dist_fig, kernel_size=3)
cv2.imshow("Imagen de distancias Filtrada", escalaImg(dist_fig_med.copy(), 1.25))
cv2.waitKey(0)
cv2.destroyAllWindows()

#############################################
## UMBRAL GLOBAL DE OTSU

dist_fig_med8 = (dist_fig_med * 255).astype(np.uint8).flatten()
histLuv = np.zeros((256, 20))
hist, _ = np.histogram(dist_fig_med8, bins=256, range=(0, 255))
histLuv[:, 0] = hist

level = [otsun(histLuv[:, 0])]
indice = [int(level[0] * 255 + 1)]

NN = 0
while level[NN] > 0.1 and NN < 19:
    histLuv[:, NN+1] = 0
    histLuv[:indice[NN], NN+1] = histLuv[:indice[NN], NN]
    level.append(otsun(histLuv[:, NN+1]))
    indice.append(int(level[NN+1] * 255 + 1))
    NN += 1

nnn = np.linspace(0, 1, 256)
plt.figure()
plt.stem(nnn, histLuv[:, 0], basefmt=" ")
plt.grid(True)
plt.title('Histograma Luv')
plt.stem(np.array(indice) / 256, 500 * np.ones(len(indice)), 'rD', basefmt=" ")
plt.show()

# Zona de interés según primer umbral global
umbral1 = level[0]
mask_interes = dist_fig_med <= umbral1

# Visualizar máscara
plt.imshow(mask_interes.astype(np.uint8) * 255, cmap='gray')
plt.title("Máscara - Zona de Interés")
plt.axis('off')
plt.show()

# Vectorizar valores de la zona de interés
valores_interes = dist_fig_med[mask_interes]
valores_interes8 = (valores_interes * 255).astype(np.uint8)

# Aplicar Otsu iterativo solo en esa zona
hist_local = np.zeros((256, 20))
hist_vals, _ = np.histogram(valores_interes8, bins=256, range=(0, 255))
hist_local[:, 0] = hist_vals

level_local = [otsun(hist_local[:, 0])]
indice_local = [int(level_local[0] * 255 + 1)]

NN = 0
while level_local[NN] > 0.1 and NN < 19:
    hist_local[:, NN+1] = 0
    hist_local[:indice_local[NN], NN+1] = hist_local[:indice_local[NN], NN]
    level_local.append(otsun(hist_local[:, NN+1]))
    indice_local.append(int(level_local[NN+1]*255+1))
    NN += 1

# Visualizar histograma local
plt.figure()
plt.stem(np.linspace(0, 1, 256), hist_local[:, 0], basefmt=" ")
plt.title("Histograma Local - Zona de Interés")
plt.grid(True)
plt.show()

# Clasificar subzonas dentro del área de interés
subzona1 = (dist_fig_med <= level_local[0]) & mask_interes
subzona2 = (dist_fig_med > level_local[0]) & mask_interes

# Crear imagen final con 3 regiones:
# 0 = fondo
# 1 = subzona 1
# 2 = subzona 2
region_final = np.zeros_like(dist_fig_med, dtype=np.uint8)
region_final[subzona1] = 1
region_final[subzona2] = 2

plt.imshow(region_final, cmap='jet')
plt.title("Segmentación Final con OTSU iterativo.")
plt.colorbar(label='Región')
plt.axis('off')
plt.show()
