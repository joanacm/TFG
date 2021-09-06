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
img_palma_o = img.imread("imatges\dades_traffic\y.png") 

img_palma = np.copy(img_palma_o)

R0 = img_palma[:,:,0]
#plt.imsave("imatges\dades_traffic\canal1.png", R, cmap='gray')
G0 = img_palma[:,:,1]
#plt.imsave("imatges\dades_traffic\canal2.png", G, cmap='gray')
B0 = mg_palma[:,:,2]
#plt.imsave("imatges\dades_traffic\canal3.png", B, cmap='gray')

R = np.copy(R0)
G = np.copy(G0)
B = np.copy(B0)

dim = R.shape

#centralizamos los datos y 
R_c, muR = centrarmatriz(R)
G_c, muG = centrarmatriz(G)
B_c, muB = centrarmatriz(B)

vapsR, vepsR, pcaR = PCA(R_c, 5)
vapsG, vepsG, pcaG = PCA(G_c, 5)
vapsB, vepsB, pcaB = PCA(B_c, 5)


img_invertidaR = inverse_PCA(vepsR, muR, pcaR)
#plt.imsave("imatges\dades_traffic\canal1pca.png", img_invertidaR, cmap='gray')
img_invertidaG = inverse_PCA(vepsG, muG, pcaG)
#plt.imsave("imatges\dades_traffic\canal2pca.png", img_invertidaG, cmap='gray')
img_invertidaB = inverse_PCA(vepsB, muB, pcaB)
#plt.imsave("imatges\dades_traffic\canal3pca.png", img_invertidaB, cmap='gray')

img_invertida = np.zeros((dim[0],dim[1],3))
img_invertida[:,:,0] = img_invertidaR
img_invertida[:,:,1] = img_invertidaG
img_invertida[:,:,2] = img_invertidaB

#img.imsave("imatges\dades_traffic\img_invertida50.png", img_invertida)
scipy.misc.toimage(img_invertida, cmin=0.0).save('imatges\dades_traffic\img_invertida5misc.png')

end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")


