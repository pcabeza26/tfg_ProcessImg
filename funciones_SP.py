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
    p = counts / (sum(counts)+1e-8)
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
    
    
def skewP(x):
    """
      Calculo de la asimetría o skew
      Basado en el cálculo de la biblioteca de Python
    """
    
    x = np.asarray(x)
    n = len(x)
    if n < 3:
           return np.nan
    
    mean_x = np.mean(x)
    m2 = np.sum((x - mean_x)**2) / (n - 1)  
    m3 = np.sum((x - mean_x)**3) / n        
    g1 = m3 / m2**1.5
    skewness = (np.sqrt(n * (n - 1)) / (n - 2)) * g1
    return skewness

def kurtP(x):
    """
      Calculo de la curtosis
      Basado en el cálculo de la biblioteca de Python
      fisher=False, bias=False
    """
    
    x = np.asarray(x)
    n = len(x)
    
    if n < 4:
        return np.nan  # igual que scipy: necesita al menos 4 valores para kurtosis sin sesgo

    mean_x = np.mean(x)
    
    m2 = np.sum((x - mean_x) ** 2) / (n - 1)
    m4 = np.sum((x - mean_x) ** 4) / n  # igual que scipy


    g2 = m4 / (m2 ** 2)

    # Corrección por sesgo:
    kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * ((n - 1) * g2 - 3 * (n - 1)) + 3

    return kurtosis


def RGB2labP(img):
    """
    Conversión de imagen RGB a CIE Lab.
    Equivalente a skimage.color.rgb2lab(img)
    img debe estar en el rango [0, 1] como float.
    """
    # Asegurar que está en el rango correcto
    img = np.clip(img, 0, 1)
    
    # Matriz de conversión RGB -> XYZ (sRGB, D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])

    # Corrección gamma inversa (de sRGB a lineal RGB)
    def gamma_inv(c):
        mask = c > 0.04045
        c_linear = np.empty_like(c)
        c_linear[mask] = ((c[mask] + 0.055) / 1.055) ** 2.4
        c_linear[~mask] = c[~mask] / 12.92
        return c_linear

    img_lin = gamma_inv(img)

    # Reordenar para aplicar la matriz: (H, W, 3) → (H*W, 3)
    shape = img.shape
    img_flat = img_lin.reshape(-1, 3)
    XYZ = np.dot(img_flat, M.T)

    # Normalizar a referencia blanca D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X = XYZ[:, 0] / Xn
    Y = XYZ[:, 1] / Yn
    Z = XYZ[:, 2] / Zn

    # Función auxiliar f(t)
    def f(t):
        delta = 6/29
        mask = t > delta**3
        f_t = np.empty_like(t)
        f_t[mask] = np.cbrt(t[mask])
        f_t[~mask] = t[~mask] / (3 * delta**2) + 4/29
        return f_t

    fX = f(X)
    fY = f(Y)
    fZ = f(Z)

    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    lab = np.stack([L, a, b], axis=1).reshape(shape)

    return lab


        
    
    
    
    
    
    
    
    
    
    
    
    
    