import numpy as np
import cv2
import matplotlib.pyplot as plt

from funciones_SP import rgb2luv
from funciones_SP import otsun
from funciones_SP import escalaImg

from scipy import signal
from scipy.stats import skew 
from scipy.stats import kurtosis 
from getFeatures import getFeatures

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
features = getFeatures(r_img)

cv2.imshow("Imagen Recortada",escalaImg(r_img.copy(),1.5))
cv2.waitKey(0)
cv2.destroyAllWindows()