# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:10:02 2021

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
import scipy.misc
from PIL import Image, ImageOps



def centrarmatriz(X):
    
    X = np.array(X)
    
    #calculamos la mediana por columna
    mu = np.mean(X,0)
    
    # Restamos la mediana de cada fila a cada elemento de la fila 
    D = X-mu[None, :]
    
    return D, mu


def var_acumulada(vaps):
    
    vaps = np.array(vaps)
    
#    for i in range(0, len(vaps)):
#        if vaps[i] <1 :
#            vaps[i]=0
            
    suma_vaps = np.cumsum(vaps)
    total = suma_vaps[len(suma_vaps)-1]
    prop_varian = 100*(suma_vaps/total)
    plt.plot(prop_varian)
    #plt.savefig('imatges\dades_compresio_gris\var.png')


def PCA(D, num_componentes):
    #X es la matriz de datos MxN
    #num_componentes es el numero de componentes con los que te quieres quedar
    
    N=len(D[0]) #num canales
    M=len(D)
    
          
    cov_X_np = np.zeros((N,N))
    # cov_X_np = np.cov(D.T)
    
    for c1 in range(0, N):
        for c2 in range(0, N):
            for i in range(0, M):
                cov_X_np[c1][c2] = cov_X_np[c1][c2] + ( D[i][c1]*D[i][c2]  )
                
    for c1 in range(0, N):
        for c2 in range(0, N):
            cov_X_np[c2][c1] = cov_X_np[c2][c1] / (M-1.0)  

    
    #Calculamos los vaps y veps de la matriz de covarianzas
    #Uso eigh porque la matriz de cov es simetrica
    #los veps son por columnas
    vaps, veps = np.linalg.eig(cov_X_np.T)   
    #print(vaps)
    
    vaps_ordenados = vaps[np.argsort(vaps)[::-1]]
    
    #ordenamos los veps en orden decreciente
    veps_ordenados = veps[:,np.argsort(vaps)[::-1]]
    
    #Escogemos los (num_componentes) componenetes primeros
    veps_reducidos = veps_ordenados[:,0:num_componentes]
    
    
    #inicializamos cada uno de los componentes 
    PC = np.zeros((M,num_componentes))
    
    #Donat que els veps venen amb columna i son les proporcions de cada color per cada component
    #determinam quina proporció de cada color li correspon a cada pixel segons cada component
    
    for c in range(0,num_componentes):
        for i in range(0, M):
            for j in range(0, N):
                PC[i][c] = PC[i][c] + veps_reducidos[j][c]*D[i][j] 
#                if isinstance(veps_reducidos[j][c]*D[i][j],complex) == True:
#                    print("veps = ", veps_reducidos[j][c])
#                    print("D = ",D[i][j])
    
    return vaps_ordenados , veps_reducidos, PC



def inverse_PCA(veps, mu, pca):
    
    num_componentes = len(pca[0]) #columnes
    M = len(pca) #files
    N = len(veps) 
    
    X=np.zeros((M,N))
    
    for c in range(0, N): 
        for i in range(0, M): 
            for j in range(0, num_componentes):
                X[i][c] = X[i][c] + veps[c][j]*pca[i][j]
    
    for i in range(0,M):
        for j in range(0,N):
            X[i][j] = X[i][j] + mu[j]
    
    return X
    


def ecualizador_histogramas(img_g, PC1):
    
    #dimension de PC1 = dimension img_g
    dim = len(img_g)
    
    muImg_g, sdImg_g, muPC1, sdPC1 = 0.0, 0.0, 0.0, 0.0
    
    for i in range(0, dim):
        muImg_g = muImg_g + img_g[i]
        muPC1 = muPC1 + PC1[i]
        sdImg_g = sdImg_g + (img_g[i])*(img_g[i])
        sdPC1 = sdPC1 + (PC1[i])*(PC1[i])
    
    muImg_g = muImg_g / dim
    muPC1 = muPC1 / dim
    
    sdImg_g = m.sqrt( (sdImg_g/dim) - muImg_g*muImg_g )
    sdPC1 = m.sqrt( (sdPC1/dim) - muPC1*muPC1 )
    
    for i in range(0, dim):
        img_g[i] = (img_g[i] - muImg_g)*(sdPC1 / sdImg_g) + muPC1

    return img_g


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




start = time.time()

#leemos la imagen 
img_vall_o = img.imread("imatges\dades_compresio_gris\y.png")
img_vall = np.copy(img_vall_o)
dim = img_vall.shape


#centralizamos los datos y 
img_vall_c, mu = centrarmatriz(img_vall)
vaps, veps, pca = PCA(img_vall_c, 405)

var_acumulada(vaps)

img_invertida = inverse_PCA(veps, mu, pca)

plt.imsave("imatges\dades_compresio_gris\y405.png", img_invertida, cmap='gray')
#scipy.misc.toimage(img_invertida, cmin=0.0).save('imatges\dades_compresio_gris\y5.png')


end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")

