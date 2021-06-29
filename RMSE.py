# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:36:24 2021

@author: Joana
"""


import numpy as np
import pandas as pd
import sys
import math as m

#tiempo
import time

#Visualizar
import matplotlib.pyplot as plt
import matplotlib.image as img



def RMSE(mejora, inicial):
    dim = np.array(inicial.shape)
    num_pixeles = dim[0]*dim[1]
    
    suma = 0
    t = 0
    
    for c in range(0,dim[2]):
        for i in range(0,dim[1]):
            for j in range(0,dim[0]):
                suma = suma + (mejora[j,i,c]-inicial[j,i,c])**2
        suma = m.sqrt(suma/num_pixeles)
        t = t + suma 
    
    t = t/dim[2]
    return t



start2 = time.time()

img_c_o = img.imread("imatges\image1\image1_interpolated.png")
img_mej = img.imread("imatges\image1\img1_mejorada.png")

img_c = np.copy(img_c_o)
img_mejorada = np.copy(img_mej)
# mejora = img_mejorada
# inicial=img_c

# dim = np.array(inicial.shape)
# num_pixeles = dim[0]*dim[1]
    
# suma = 0
# t = 0
    
# for c in (0, dim[2]):
#     for i in (0, dim[1]):
#         print("i=", i)
#         for j in (0, dim[0]):
#             print(dim[0])
#             print("j=", j)
#             print(mejora[j,i,c])
#             print(inicial[j,i,c])
#             suma = suma + (mejora[j,i,c]-inicial[j,i,c])**2
#             print(suma)
#     suma = m.sqrt(suma/num_pixeles)
#     t = t + suma 
    
# t = t/dim[2]




print(RMSE(img_mejorada, img_c))

end2 = time.time()
print("El tiempo de ejecución es de "+ str(end2 - start2) + " segundos.")
