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

from scipy import signal
from scipy.stats import skew 
from scipy.stats import kurtosis 
import pandas as pd
#from skimage.filters import threshold_otsu


# Lectura Imagen
l_img = cv2.imread("C:/Users/pcabe/tfg/heman_sup.JPG")



factor = 0.55
width = int(l_img.shape[1] * factor )
height = int(l_img.shape[0] * factor)
dim = (width, height)
s_img = cv2.resize(l_img, dim, interpolation=cv2.INTER_AREA)


# RECORTE
x, y, w, h = cv2.selectROI("Seleccionar Seccion",s_img, showCrosshair=True, fromCenter=False)



# Imagen Recortada
r_img = s_img[y:y+h,x:x+w]



cv2.imshow("Imagen Recortada",r_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Normalización
img = r_img/255

###########################################
## TRATAMIENTO DEL ORIGINAL


# Descomposición RGB
s_img = s_img/255
imgR = s_img[:,:,0]
imgG = s_img[:,:,1]
imgB = s_img[:,:,2]

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
[L,u,v,teta,saturation] = rgb2luv(RGB)



###########################################
## TRATAMIENTO DEL RECORTE


# Descomposición en RGB del recorte
imgRr = img[:,:,0]
imgGr = img[:,:,1]
imgBr = img[:,:,2]


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

cv2.imshow("Imagen de distancias",dist_fig)



# Imagen filtrada por mediana 
dist_fig_med = signal.medfilt2d(dist_fig, kernel_size=3)

cv2.imshow("Imagen de distancias Filtrada",dist_fig_med)
cv2.waitKey(0)
cv2.destroyAllWindows()


#############################################
## APLICACIÓN ITERATIVA MÉTODO OTSU

# Convierte de RGB en escala de grises a Unsigned de 8 bits)
dist_fig_med8 = (dist_fig_med*255).astype(np.uint8) 
dist_fig_med8 = np.reshape(dist_fig_med8, (dist_fig_med8.shape[0]*dist_fig_med8.shape[1],1))

histLuv = np.zeros((256,20))
hist, _ = np.histogram(dist_fig_med8,bins=256, range=(0,255))
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

# Buscar umbral más cercano a 0.1
nuevo = np.where(np.array(level) >= 0.1)[0]  # np.where devuelve una tupla, obtenemos el primer índice
if len(nuevo) > 0:
    umbral = nuevo[0] 


#level = 0.15
#umbral= 1
dist_fig_med_vec = np.reshape(dist_fig_med,(Mr,1))


ind = np.where(dist_fig_med_vec <= level)[0] 
#ind2 = np.where(dist_fig_med_vec <= level)[0]
#mask = dist_fig_med_vec <= level 

############################
## IMAGEN FINAL

RGBfinal_r = imgRr_vect.copy()
RGBfinal_g = imgGr_vect.copy()
RGBfinal_b = imgBr_vect.copy()

RGBfinal_r[ind] = 1.0
RGBfinal_g[ind] = 1.0
RGBfinal_b[ind] = 0.0


# Matriz completa
RGBfinal = np.stack([RGBfinal_r, RGBfinal_g, RGBfinal_b], axis=-1)
RGBfinal = np.reshape(RGBfinal, (fil_recorte, col_recorte, 3))
RGBfinal = (RGBfinal * 255).astype(np.uint8)


cv2.imshow("Imagen Final",RGBfinal)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Vector -> Matriz
#RGBfinal_r = np.reshape(RGBfinal_r,(fil_recorte,col_recorte))
#RGBfinal_g = np.reshape(RGBfinal_g,(fil_recorte,col_recorte))
#RGBfinal_b = np.reshape(RGBfinal_b,(fil_recorte,col_recorte))

#RGB_indice = np.column_stack((imgRr_vect[ind] , imgGr_vect[ind] , imgBr_vect[ind] ))
# Alrevés coincide con imgRr_vect en lugar de imgBr_vect ***
RGB_indice = np.column_stack((imgBr_vect[ind] , imgGr_vect[ind] , imgRr_vect[ind] ))

# FINAL PARÁMETROS
[Lfinal,ufinal,vfinal,tetafinal,saturationfinal] = rgb2luv(RGB_indice)


#############################################
## ALMACENAMIENTO DE DATOS

# Hay que forzar la std en python a que sea muestral como en matlab ddof=1
data = [
    {"Media L": np.mean(Lfinal), "Media Theta": np.mean(tetafinal), "MediaSaturación": np.mean(13*saturationfinal),  
     "Std L": np.std(Lfinal,ddof=1),  "Std Theta": np.std(tetafinal,ddof=1),  "Std Saturación": np.std(13*saturationfinal,ddof=1),
     "Media u": np.mean(ufinal), "Media v": np.mean(vfinal),
     "Std u": np.std(ufinal,ddof=1), "Std v": np.std(vfinal,ddof=1),
     "Skew L": skew(Lfinal), "Kurt L": kurtosis(Lfinal),
     "Skew u": skew(ufinal), "Kurt u": kurtosis(ufinal),
     "Skew v": skew(vfinal), "Kurt v": kurtosis(vfinal),}
    ]

# Convertir a DataFrame
df = pd.DataFrame(data)

# Guardar en CSV (modo 'a' para agregar datos sin sobrescribir)
csv_file = "resultados.csv"
df.to_csv(csv_file, index=False, mode='a', header=not pd.io.common.file_exists(csv_file))

print(f"Datos guardados en {csv_file}")






