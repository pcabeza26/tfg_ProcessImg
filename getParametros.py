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
from skimage import color

l_img = cv2.imread("C:/Users/pcabe/tfg/imagenes/heman_sup.jpg")


# Escalado de imagen
factor = 0.55
s_img = escalaImg(l_img, factor)


# RECORTE
x, y, w, h = cv2.selectROI("Seleccionar Seccion",s_img, showCrosshair=True, fromCenter=False)

# Imagen Recortada
r_img = s_img[y:y+h,x:x+w]


cv2.imshow("Imagen Recortada",escalaImg(r_img.copy(),1.5))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalización
img = r_img/255

## TRATAMIENTO DEL RECORTE
# Descomposición en RGB del recorte
imgRr = img[:,:,2]
imgGr = img[:,:,1]
imgBr = img[:,:,0]


# Filas y columnas del recorte para aplanamiento
fil_recorte = img.shape[0]
col_recorte = img.shape[1]
Mr = fil_recorte * col_recorte

# Aplanamiento
imgRr_vect = np.reshape(imgRr,(Mr,1))
imgGr_vect = np.reshape(imgGr,(Mr,1))
imgBr_vect = np.reshape(imgBr,(Mr,1))

# Matriz con los tres vectores
RGBr = np.column_stack((imgRr_vect, imgGr_vect, imgBr_vect))

# CÁLCULO COORDENADAS L*u*v del recorte
[Lr,ur,vr,tetar,saturationr] = rgb2luv(RGBr)

# Centroides de recortes
L_cent = np.mean(Lr)
u_cent = np.mean(ur)
v_cent = np.mean(vr)

dist = np.sqrt((Lr-L_cent)**2+(ur-u_cent)**2+(vr-v_cent)**2)
dist_fig = np.reshape(dist,(fil_recorte,col_recorte))                  # Conv a matriz
dist_fig = dist_fig/np.max(dist_fig)


dist_fig_med = cv2.bilateralFilter(dist_fig.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)


## MÉTODO OTSU
dist_fig_med8 = (dist_fig_med*255).astype(np.uint8) 
dist_fig_med8 = np.reshape(dist_fig_med8, (dist_fig_med8.shape[0]*dist_fig_med8.shape[1],1))

histLuv = np.zeros((256,20))
hist, _ = np.histogram(dist_fig_med8,bins=256, range=(0,255)) # Cúantos píxeles tiene cada nivel de intensidad 1x256
histLuv[:, 0] = hist  # Guardar el histograma en la primera columna


level = [otsun(histLuv[:, 0])] # Primer umbral


###   MÁSCARA

lab_img = color.rgb2lab(img)  # img ya está normalizada (rango [0, 1])

# Aplanamos los canales LAB
L_flat = lab_img[:, :, 0].reshape(Mr, 1)
a_flat = lab_img[:, :, 1].reshape(Mr, 1)
b_flat = lab_img[:, :, 2].reshape(Mr, 1)

color_mask = (a_flat > 10) & (b_flat < 25)


## Segmentación
dist_fig_med_vec = np.reshape(dist_fig_med,(Mr,1))
ind = np.where(dist_fig_med_vec <= level)[0]


############################
## IMAGEN FINAL



RGBfinal_r = imgRr_vect.copy()
RGBfinal_g = imgGr_vect.copy()
RGBfinal_b = imgBr_vect.copy()


RGBfinal_r[ind] = 1.0
RGBfinal_g[ind] = 1.0
RGBfinal_b[ind] = 1.0


RGBfinal_r[~color_mask] = 1.0
RGBfinal_g[~color_mask] = 1.0
RGBfinal_b[~color_mask] = 1.0


# Crear máscara booleana con valores que son hemangioma
mask_inv = np.ones(Mr, dtype=bool)
mask_inv = (RGBfinal_r != 1.0) & (RGBfinal_g != 1.0) & (RGBfinal_b != 1.0)
mask_inv = mask_inv.ravel()  # Asegurarse de que sea vector 1D


# Matriz completa
RGBfinal = np.stack([RGBfinal_r, RGBfinal_g, RGBfinal_b], axis=-1)
RGBfinal = np.reshape(RGBfinal, (fil_recorte, col_recorte, 3))
RGBfinal = (RGBfinal * 255).astype(np.uint8)

RGBfinal_BGR = cv2.cvtColor(RGBfinal, cv2.COLOR_RGB2BGR)
cv2.imshow("Imagen Final",escalaImg(RGBfinal_BGR,1.5))
cv2.waitKey(0)
cv2.destroyAllWindows()



# Aplicar la máscara inversa a los vectores de canales
Rfinal = imgRr_vect[mask_inv]
Gfinal = imgGr_vect[mask_inv]
Bfinal = imgBr_vect[mask_inv]



RGB_indice = np.column_stack((Rfinal , Gfinal , Bfinal))

# FINAL PARÁMETROS
[Lfinal,ufinal,vfinal,tetafinal,saturationfinal] = rgb2luv(RGB_indice)



RGB_normalizado = RGB_indice 

# 3. Convertir a L*a*b*
Lab_final = color.rgb2lab(RGB_normalizado)

# 4. Separar canales
L_lab_final = Lab_final[:, 0]  # L*
a_final = Lab_final[:, 1]  # a*
b_final = Lab_final[:, 2]  # b*


data = [
    {"Media R": np.mean(Rfinal), 
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
     
     "Media Theta Circular":  0.5 * np.arctan2(np.mean(np.sin(2*np.pi*tetafinal)), np.mean(np.cos(2*np.pi*tetafinal))),  
     
     "Media u": np.mean(ufinal),
     "Media v": np.mean(vfinal),
     
     "Std u": np.std(ufinal, ddof=1),
     "Std v": np.std(vfinal, ddof=1),
     
     "Skew u": skew(ufinal, bias=False),
     "Kurt u": kurtosis(ufinal, fisher=False, bias=False),
     
     "Skew v": skew(vfinal, bias=False),
     "Kurt v": kurtosis(vfinal, fisher=False, bias=False)
    }
]
caracteristicas = np.array(list(data[0].values()))
