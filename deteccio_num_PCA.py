# -*- coding: utf-8 -*-
"""
Created on Tue Jul 06 13:50:22 2021

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

#Realizar PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def distancia(test, train):
    errores = np.zeros(len(train))
    for i in range(0,len(train)):
        for j in range(0, len(test)):
            errores[i] = errores[i] + (train[i][j] - test[j])*(train[i][j] - test[j])
        errores[i] = m.sqrt(errores[i])
    return errores


start = time.time()

test = pd.read_csv('data/mnist/test.csv')
train = pd.read_csv('data/mnist/train.csv')

#guardamos que número representa cada imagen (fila)
num_train = train['label']
#eliminamos la variable label 
del(train['label'])

#guardamos que número representa cada imagen (fila)
num_test = test['label']
#eliminamos la variable label 
del(test['label'])

#realitzamos una PCA en train
pca = PCA()
train_proy = pca.fit_transform(train)
veps = pca.components_

for i in range(0, len(num_test)):
    #cargamos una imagen de un dígito desconocido
    ejemplo_test = test.iloc[i].values

    #projectam l'exemple a les coordenades anteriors (veps)
    test_proy = np.dot(veps, ejemplo_test)
    
    errores = distancia(test_proy, train_proy)
    
    index_train = num_train[np.argmin(errores)]
    index_text = num_test[i]
    
    #comprobamos si se ha acertado
    if index_text != index_train:
        test_x = test.iloc[i].values.reshape([28,28])
        plt.imshow(test_x, cmap='gray_r')
        plt.title(u'Resultado: %s' % str(index_train))
        plt.savefig("imatges\deteccion\image_erronea_%d.png"%(i+1))


end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")
