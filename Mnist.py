# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:14:33 2021

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

#Hacer PCA
from sklearn.decomposition import PCA





def crear_imagenes(dataframe):
    original = dataframe.iloc[0].values.reshape([28,28])
    plt.imshow(original, cmap='gray_r')
    plt.savefig("5.png")
    
    for i in range(0, dataframe.shape[1], 50 ):
        pca = PCA(i)
        transformado = pca.fit_transform(dataframe)
        reducido = pca.inverse_transform(transformado)
    
        img_pca_i = reducido[0,:].reshape([28,28])
        plt.imshow(img_pca_i, cmap='gray_r')
        plt.savefig("image_pca_%d_sin_titulo.png"%(i+1)) 
        


def calculo_var_acumulada(pca):
    plt.grid()
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel('Numero de componentes')
    plt.ylabel('Varianza Explicada')
    plt.savefig('Var_plot.png')
  
  
def RMSE(mejora, inicial):
    dim = np.array(inicial.shape)
    num_pixeles = dim[0]*dim[1]
    
    suma = 0
    t = 0
    
    for i in range(0,dim[1]):
        for j in range(0,dim[0]):
            suma = suma + (mejora[j,i]-inicial[j,i])**2
    suma = m.sqrt(suma/num_pixeles)
    t = t + suma 
    
    return t






start = time.time()

mnist = pd.read_csv('mnist.csv')
del(mnist['label'])


#calculo del RMSE a medida que sumo componentes
vecRMSE = np.zeros(28*28)
original = mnist.iloc[0].values.reshape([28,28])

for i in range(0, mnist.shape[1]):
    pca = PCA(i)
    transformado = pca.fit_transform(mnist)
    reducido = pca.inverse_transform(transformado)
    
    img_pca_i = reducido[0,:].reshape([28,28])
    vecRMSE[i] = RMSE(img_pca_i, original)
    print(i)
    
plt.grid()
plt.plot(vecRMSE)
plt.xlabel('Numero de componentes')
plt.ylabel('RMSE')
plt.savefig('RMSE_mnist_plot.png')



end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")





