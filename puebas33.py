# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:21:31 2025

@author: pcabe
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import signal
from scipy.stats import skew, kurtosis
from skimage.color import rgb2luv, rgb2lab
from skimage.filters import threshold_otsu

import pandas as pd

# Función de escalado mantenida
def escalaImg(imagen, factor):
    height, width = imagen.shape[:2]
    new_size = (int(width * factor), int(height * factor))
    return cv2.resize(imagen, new_size, interpolation=cv2.INTER_AREA)

# Lectura Imagen
l_img = cv2.imread("C:/Users/pcabe/tfg/imagenes/heman2.jpg")
# Blanco de referencia
un = 0.1978
vn = 0.4683

# Escalado de imagen
factor = 0.55
s_img = escalaImg(l_img, factor)

# RECORTE
x, y, w, h = cv2.selectROI("Seleccionar Seccion", s_img, showCrosshair=True, fromCenter=False)
r_img = s_img[y:y+h, x:x+w]
cv2.imshow("Imagen Recortada", r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalización
img = r_img / 255.0
s_img = s_img / 255.0

###########################################
## TRATAMIENTO DEL ORIGINAL

imgR = s_img[:, :, 0]
imgG = s_img[:, :, 1]
imgB = s_img[:, :, 2]
M = s_img.shape[0] * s_img.shape[1]

RGB = np.stack([imgR.flatten(), imgG.flatten(), imgB.flatten()], axis=-1)

luv = rgb2luv(RGB.reshape((-1, 1, 3))).reshape((-1, 3))
L, u, v = luv[:, 0], luv[:, 1], luv[:, 2]
saturation = np.sqrt((u-un)**2 + (v-vn)**2)
teta = np.arctan2(v, u) / (2 * np.pi)

###########################################
## TRATAMIENTO DEL RECORTE

imgRr = img[:, :, 0]
imgGr = img[:, :, 1]
imgBr = img[:, :, 2]

Mr = img.shape[0] * img.shape[1]

RGBr = np.stack([imgRr.flatten(), imgGr.flatten(), imgBr.flatten()], axis=-1)

luv_r = rgb2luv(RGBr.reshape((-1, 1, 3))).reshape((-1, 3))
Lr, ur, vr = luv_r[:, 0], luv_r[:, 1], luv_r[:, 2]
saturationr = np.sqrt((ur-un)**2 + (vr-vn)**2)
tetar = np.arctan2(vr, ur) / (2 * np.pi)

L_cent = np.mean(Lr)
u_cent = np.mean(ur)
v_cent = np.mean(vr)

###########################################
## IMAGEN DE DISTANCIAS

dist = np.sqrt((Lr - L_cent)**2 + (ur - u_cent)**2 + (vr - v_cent)**2)
dist_fig = dist.reshape(img.shape[0], img.shape[1])
dist_fig = dist_fig / np.max(dist_fig)

cv2.imshow("Imagen de distancias", escalaImg(dist_fig.copy(), 1.25))

# Filtrado por mediana
dist_fig_med = signal.medfilt2d(dist_fig, kernel_size=3)
cv2.imshow("Imagen de distancias Filtrada", escalaImg(dist_fig_med.copy(), 1.25))
cv2.waitKey(0)
cv2.destroyAllWindows()

###########################################
## APLICACIÓN OTSU

dist_fig_med8 = (dist_fig_med * 255).astype(np.uint8)
level = threshold_otsu(dist_fig_med8)

dist_fig_med_vec = dist_fig_med.flatten()
ind = np.where(dist_fig_med_vec <= (level / 255.0))[0]

###########################################
## IMAGEN FINAL

RGBfinal_r = imgRr.flatten().copy()
RGBfinal_g = imgGr.flatten().copy()
RGBfinal_b = imgBr.flatten().copy()

RGBfinal_r[ind] = 1.0
RGBfinal_g[ind] = 1.0
RGBfinal_b[ind] = 0.0

RGBfinal = np.stack([RGBfinal_r, RGBfinal_g, RGBfinal_b], axis=-1)
RGBfinal = RGBfinal.reshape((img.shape[0], img.shape[1], 3))
RGBfinal = (RGBfinal * 255).astype(np.uint8)

cv2.imshow("Imagen Final", escalaImg(RGBfinal.copy(), 1.5))
cv2.waitKey(0)
cv2.destroyAllWindows()

###########################################
## PARÁMETROS DE REGIÓN DETECTADA

Bfinal = imgBr.flatten()[ind]
Gfinal = imgGr.flatten()[ind]
Rfinal = imgRr.flatten()[ind]

RGB_indice = np.stack((Rfinal, Gfinal, Bfinal), axis=-1)

luv_final = rgb2luv(RGB_indice.reshape((-1, 1, 3))).reshape((-1, 3))
Lfinal, ufinal, vfinal = luv_final[:, 0], luv_final[:, 1], luv_final[:, 2]
saturationfinal = np.sqrt((ufinal-un)**2 + (vfinal-vn)**2)
tetafinal = np.arctan2(vfinal, ufinal) / (2 * np.pi)

Lab_final = rgb2lab(RGB_indice.reshape((-1, 1, 3))).reshape((-1, 3))
L_lab_final, a_final, b_final = Lab_final[:, 0], Lab_final[:, 1], Lab_final[:, 2]

###########################################
## ALMACENAMIENTO DE DATOS

data = [{
    "Media R": np.mean(Rfinal),
    "Media G": np.mean(Gfinal),
    "Media B": np.mean(Bfinal),

    "Media L": np.mean(Lfinal),
    "Media h": np.mean(tetafinal),
    "Media c": np.mean(13 * saturationfinal),

    "Std L": np.std(Lfinal, ddof=1),
    "Std h": np.std(tetafinal, ddof=1),
    "Std c": np.std(13 * saturationfinal, ddof=1),

    "Media a": np.mean(a_final),
    "Media b": np.mean(b_final),

    "Std a": np.std(a_final, ddof=1),
    "Std b": np.std(b_final, ddof=1),

    "Skew L": skew(Lfinal, bias=False),
    "Kurt L": kurtosis(Lfinal, fisher=False, bias=False),

    "Skew a": skew(a_final, bias=False),
    "Kurt a": kurtosis(a_final, fisher=False, bias=False),

    "Skew b": skew(b_final, bias=False),
    "Kurt b": kurtosis(b_final, fisher=False, bias=False),

    "Media Theta Circular": 0.5 * np.arctan2(np.mean(np.sin(2 * np.pi * tetafinal)),
                                             np.mean(np.cos(2 * np.pi * tetafinal))),

    "Media u": np.mean(ufinal),
    "Media v": np.mean(vfinal),

    "Std u": np.std(ufinal, ddof=1),
    "Std v": np.std(vfinal, ddof=1),

    "Skew u": skew(ufinal, bias=False),
    "Kurt u": kurtosis(ufinal, fisher=False, bias=False),

    "Skew v": skew(vfinal, bias=False),
    "Kurt v": kurtosis(vfinal, fisher=False, bias=False)
}]

df = pd.DataFrame(data)

csv_file = "resultados.csv"
df.to_csv(csv_file, index=False, mode='a', header=not pd.io.common.file_exists(csv_file))

print(f"Datos guardados en {csv_file}")
