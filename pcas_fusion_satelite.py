# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 12:33:33 2021

@author: joana
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



def centrarmatriz(X):
    
    X = np.array(X)
    
    #calculamos la mediana por columna
    mu = np.mean(X,0)
    
    # Restamos la mediana de cada fila a cada elemento de la fila 
    D = X-mu[None, :]
    
    return D, mu


def PCA_svd(D, num_componentes = 3):
    #X es la matriz de datos MxN
    #num_componentes es el numero de componentes con los que te quieres quedar
    
    N=len(D[0]) #num variables
    M=len(D)
    
    cov_X_np = np.zeros((N,N))
    
    for c1 in range(0, N):
        for c2 in range(0, N):
            for i in range(0, M):
                cov_X_np[c1][c2] = cov_X_np[c1][c2] + ( D[i][c1]*D[i][c2]  )
                
    for c1 in range(0, N):
        for c2 in range(0, N):
            cov_X_np[c2][c1] = cov_X_np[c2][c1] / (M-1.0) 
    
    #calculamos SVD de D
    U,s,Vt = np.linalg.svd(cov_X_np)
    print("U = ", U)
    print("s = ", s)
    print("Vt = ", Vt)
    
    #els vaps de X es el cuadrat dels valors singulars (que ja venen ordenats)
    #Son los vaps de D^T * D
    vaps_odenados = s**2
    
    #escogemos los (num_componentes) mas grandes
    vaps_reducidos = vaps_odenados[0:num_componentes]
    veps_reducidos = Vt[:,0:num_componentes]

    PC = np.zeros((M,num_componentes))
    
    for c in range(0,num_componentes):
        for i in range(0, M):
            for j in range(0, N):
                PC[i][c] = PC[i][c] + veps_reducidos[j][c]*D[i][j] 
    
    
    return vaps_odenados, veps_reducidos, PC




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





def img_satelite(img_c, img_g):
    
    #dimension de la imagen en color y de cada canal
    dim = img_c.shape
    
    # for i in range(0,3):
    #     print("condicionamentiento para el canal " + str(i+1)+ " = " + str(kappa(img_c[:,:,i])))
    
    #construimos una matriz con todos los pixeles de cada canal por columna
    img_RGB = np.reshape(img_c, ( (img_c.shape[0])*(img_c.shape[1]), 3 )  )
    
    #centralizamos los datos y 
    img_RGB_c, mu = centrarmatriz(img_RGB)
    
    #veps de img_RGB_c y matriz de componentes
    vaps, veps, pca = PCA(img_RGB_c, 3)
    
    img_g = img_g.reshape((dim[0])*(dim[1]))
    
    pca[:,0] = ecualizador_histogramas(img_g, pca[:, 0])
    
    img_invertida = inverse_PCA(veps, mu, pca)
    img_invertida = img_invertida.reshape(dim)
    
    return img_invertida


def img_satelite_svd(img_c, img_g):
    
    #dimension de la imagen en color y de cada canal
    dim = img_c.shape

    #construimos una matriz con todos los pixeles de cada canal por columna
    img_RGB = np.reshape(img_c, ( (img_c.shape[0])*(img_c.shape[1]), 3 )  )
    
    #centralizamos los datos y 
    img_RGB_c, mu = centrarmatriz(img_RGB)
    
    #veps de img_RGB_c y matriz de componentes
    vaps, veps, pca = PCA_svd(img_RGB_c, 3)
    
    img_g = img_g.reshape((dim[0])*(dim[1]))
    
    pca[:,0] = ecualizador_histogramas(img_g, pca[:, 0])
    
    img_invertida = inverse_PCA(veps, mu, pca)
    img_invertida = img_invertida.reshape(dim)
    
    return img_invertida



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








#leemos la imagen 
img_g_o = img.imread("imatges\image2\image2_gray.png") 
img_c_o = img.imread("imatges\image2\image2_interpolated.png")


#creo una copia de la imagen original (por si acaso)
img_g = np.copy(img_g_o)
img_c = np.copy(img_c_o)

start = time.time()
img_mejorada_svd = img_satelite(img_c, img_g)

plt.imsave("imatges\image2\img2_mejorada0_svd.png", img_mejorada_svd)
plt.imsave("imatges\image2\img2_mejorada0_svd.tiff", img_mejorada_svd)

end = time.time()
print("El tiempo de ejecución con SVD es de "+ str(end - start) + " segundos.")

start = time.time()
img_mejorada = img_satelite(img_c, img_g)

plt.imsave("imatges\image1\img1_mejorada0.png", img_mejorada)
plt.imsave("imatges\image1\img1_mejorada0.tiff", img_mejorada)

end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")

start2 = time.time()

print("L'error per RMSE es de ", RMSE(img_mejorada, img_c))
print("L'error per RMSE_SVD es de ", RMSE(img_mejorada_svd, img_c))

end2 = time.time()
print("El tiempo de ejecución es de "+ str(end2 - start2) + " segundos.")
