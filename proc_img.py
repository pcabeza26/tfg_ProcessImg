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
#from skimage.filters import threshold_otsu 


# Lectura Imagen
#l_img = cv2.imread("C:/Users/pcabe/tfg/imagen8.jpg")
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

###########################################
## TRATAMIENTO DEL ORIGINAL


# Descomposición RGB
s_img = s_img/255
imgR = s_img[:,:,2]
imgG = s_img[:,:,1]
imgB = s_img[:,:,0]

# Filas * Columnas
filas =  s_img.shape[0]
col   =  s_img.shape[1]
M = filas*col

# Aplanamiento
imgR_vect = np.reshape(imgR,(M,1))
imgG_vect = np.reshape(imgG,(M,1))
imgB_vect = np.reshape(imgB,(M,1))

# Matriz con los tres vectores
RGB = np.column_stack((imgR_vect, imgG_vect, imgB_vect))

# CÁLCULO COORDENADAS L*u*v del original
#[L,u,v,teta,saturation] = rgb2luv(RGB)



###########################################
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


###########################################
## IMAGEN DE DISTANCIAS

# L*u*v - centroides recorte
dist = np.sqrt((Lr-L_cent)**2+(ur-u_cent)**2+(vr-v_cent)**2)
dist_fig = np.reshape(dist,(fil_recorte,col_recorte))                  # Conv a matriz
dist_fig = dist_fig/np.max(dist_fig)

cv2.imshow("Imagen de distancias",escalaImg(dist_fig.copy(),1.5))

# Imagen filtrada por mediana 
#dist_fig_med = signal.medfilt2d(dist_fig, kernel_size=3)
dist_fig_med = cv2.bilateralFilter(dist_fig.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
#dist_fig_med = dist_fig.copy()

cv2.imshow("Imagen de distancias Filtrada",escalaImg(dist_fig_med.copy(),1.5))
cv2.waitKey(0)
cv2.destroyAllWindows()


#############################################
## APLICACIÓN ITERATIVA MÉTODO OTSU. CALCULA IND--> NO hemangioma

# Convierte de RGB en escala de grises a Unsigned de 8 bits)
dist_fig_med8 = (dist_fig_med*255).astype(np.uint8) 
dist_fig_med8 = np.reshape(dist_fig_med8, (dist_fig_med8.shape[0]*dist_fig_med8.shape[1],1))

histLuv = np.zeros((256,20))
hist, _ = np.histogram(dist_fig_med8,bins=256, range=(0,255)) # Cúantos píxeles tiene cada nivel de intensidad 1x256
histLuv[:, 0] = hist  # Guardar el histograma en la primera columna


level = [otsun(histLuv[:, 0])] # Primer umbral
indice = [int(level[0] * 255 + 1)]


NN = 0 

while level[NN] > 0.05 and NN<19:# Evitar desbordamientos

    histLuv[:,NN+1] = 0 # Init columna en 0
    histLuv[:indice[NN],NN+1] = histLuv[:indice[NN],NN] # Valores previos
    
    level.append(otsun(histLuv[:,NN+1])) # Nuevo umbral
    indice.append(int(level[NN+1]*255+1)) # Actualizar indice
    
    NN = NN+1
    

# Generar el vector nnn con valores entre 0 y 1 (equivalente a linspace)
nnn = np.linspace(0, 1, 256)

# Graficar el histograma de Luv
plt.figure()
plt.stem(nnn, histLuv[:, 0], basefmt=" ")
plt.grid('On')
plt.title('Histograma Luv')

# Graficar los valores de índice como marcadores rojos ('rd' en MATLAB → 'rD' en Matplotlib)
plt.stem(np.array(indice) / 256, 500 * np.ones(len(indice)), 'rD', basefmt=" ")

plt.show()


#################

lab_img = color.rgb2lab(img)  # img ya está normalizada (rango [0, 1])

# Aplanamos los canales LAB
L_flat = lab_img[:, :, 0].reshape(Mr, 1)
a_flat = lab_img[:, :, 1].reshape(Mr, 1)
b_flat = lab_img[:, :, 2].reshape(Mr, 1)

color_mask = (a_flat > 10) & (b_flat < 25)


####################
dist_fig_med_vec = np.reshape(dist_fig_med,(Mr,1))
ind = np.where(dist_fig_med_vec <= level)[0]
  
 
#ind = np.where(dist_fig_med_vec <= level[umbral])[0]
#mask = dist_fig_med_vec <= level 


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



#############################################
## ALMACENAMIENTO DE DATOS

# Hay que forzar la std en python a que sea muestral como en matlab ddof=1
 
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

# Convertir a DataFrame
df = pd.DataFrame(data)

# Guardar en CSV (modo 'a' para agregar datos sin sobrescribir)
csv_file = "resultados.csv"
df.to_csv(csv_file, index=False, mode='a', header=not pd.io.common.file_exists(csv_file))

print(f"Datos guardados en {csv_file}")







