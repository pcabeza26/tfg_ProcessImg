# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:20:19 2025

Función conversión RGB to Luv
@author: pcabe
"""
import numpy as np

 

#RGBrgb = pd.read_csv('RGB.csv', header=None).values    
def rgb2luv (RGBrgb):  
    '''
         Convierte de RGB a L*u*v 
    '''
    
    
    # Blanco de referencia
    un = 0.1978
    vn = 0.4683
    
       
    XYZ = np.array([[0.490, 0.310, 0.200], [0.17697, 0.8124, 0.01063], [0.000, 0.010, 0.990]])
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

    
def noisy(noise_typ,image):
        
        
       if noise_typ == "gauss":
          row,col,ch= image.shape
          mean = 0
          var = 0.1
          sigma = var**0.5
          gauss = np.random.normal(mean,sigma,(row,col,ch))
          gauss = gauss.reshape(row,col,ch)
          noisy = image + gauss
          return noisy
       elif noise_typ == "s&p":
          row,col,ch = image.shape
          s_vs_p = 0.5
          amount = 0.004
          out = np.copy(image)
          # Salt mode
          num_salt = np.ceil(amount * image.size * s_vs_p)
          coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
          out[coords] = 1
    
          # Pepper mode
          num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
          coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
          out[coords] = 0
          return out
       elif noise_typ == "poisson":
          vals = len(np.unique(image))
          vals = 2 ** np.ceil(np.log2(vals))
          noisy = np.random.poisson(image * vals) / float(vals)
          return noisy
       elif noise_typ =="speckle":
          row,col,ch = image.shape
          gauss = np.random.randn(row,col,ch)
          gauss = gauss.reshape(row,col,ch)        
          noisy = image + image * gauss
          return noisy
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    