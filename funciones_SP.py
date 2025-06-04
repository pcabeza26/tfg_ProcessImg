# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:20:19 2025

Función conversión RGB to Luv
@author: pcabe
"""
import numpy as np
import cv2
 

#RGBrgb = pd.read_csv('RGB.csv', header=None).values    
def rgb2luv (RGBrgb):  
    '''
         Convierte de RGB a L*u*v 
    '''
    
    
    # Blanco de referencia
    un = 0.1978
    vn = 0.4683
    
       
    #XYZ = np.array([[0.490, 0.310, 0.200], [0.17697, 0.8124, 0.01063], [0.000, 0.010, 0.990]])
    XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.21026, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    
    XYZoriginal = RGBrgb @ np.transpose(XYZ)
    
    # Matrices X,Y,Z
    X = XYZoriginal[:,0]
    Y = XYZoriginal[:,1]
    Z = XYZoriginal[:,2]
         
    uprima = 4*X / (X+15*Y+3*Z)
    vprima = 9*Y / (X+15*Y+3*Z)
        
    p = np.where((X==0)&(Y==0)&(Z==0))
    uprima[p] =   4/19
    vprima[p] =  9/19
    
    # Cálculo y ajuste de L    
    L = 116*(Y**(1/3))-16
    t = np.where(Y <= 0.008856) 
    L[t] = 903.3 * Y[t]
    
    # Cálculo de u*v    
    u = 13*L * (uprima-un)
    v = 13*L * (vprima-vn)
        
    teta = np.arctan2(v,u)
    # El *13 se la mete luego
    saturation = np.sqrt((uprima-un)**2+(vprima-vn)**2) 
    
    
    return L,u,v,teta,saturation
    
    
# FUNCIÓN OTSUN                          
def otsun (histLuv):
    """
    Método de Otsu para encontrar un umbral de binarización basado en un histograma.
    :param histLuv: Histograma de la imagen.
    :return: Nivel de umbral normalizado en el rango [0, 1].
    """
    
    num_bins = 256
    counts   = np.array(histLuv, dtype = np.float64)
    
    # Probabilidad de cada nivel de gris
    p = counts / sum(counts)
    omega = np.cumsum(p)   # Suma acumulativa
    mu = np.cumsum(p * np.arange(1,num_bins+1))
    mu_t = mu[-1]
 
    
    # Evitar divisiones por 0 con np.errstate
    
    with np.errstate(divide='ignore', invalid = 'ignore'):
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
        
    # Encontrar el indice del máximo valor de sigma_b squared
    
    maxval = np.nanmax(sigma_b_squared)
    if np.isfinite(maxval):
        idx = np.mean(np.where(sigma_b_squared == maxval))
        level = (idx - 1 ) / (num_bins -1 ) # Normalización
    else:
        level = 0.0
    
    
    return level

    
def escalaImg(img,factor):
    """
      Escala imagenes por factor, para dimensionar a tamaño ventana cv2
      y interpolación de sus valores
    """
    width = int(img.shape[1] * factor )
    height = int(img.shape[0] * factor)
    dim = (width, height)
    s_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    return s_img
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    