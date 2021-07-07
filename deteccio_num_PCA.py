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


start = time.time()

test = pd.read_csv('data/mnist/test.csv')
train = pd.read_csv('data/mnist/train.csv')

num_train = train['label']
del(train['label'])

test.head()


#realitzamos una PCA en train
pca = PCA()
train_proy = pca.fit_transform(train)
veps = pca.components_

#cargamos una imagen de un dígito desconocido
ejemplo_test = np.array(test.iloc[2].values)

#projectam l'exemple a les coordenades anteriors (veps)
test_proy = np.dot(veps, ejemplo_test)

#calculamos las distancias
errores = np.zeros(len(num_train))
for i in range(0,len(num_train)):
    for j in range(0, 28*28):
        errores[i] = errores[i] + abs(train_proy[i][j] - test_proy[j] )


#nos quedamos con aquella imagen que se parezca mas a la queremos testear
index = np.argmin(errores)

#Mostramos que numero representa la imagen con distancia minima 
#y que suponemos que es como la que queremos testear
print("El numero que se muestra en esta imagen es " + str(num_train[index]) )


nueve = test.iloc[2].values.reshape([28,28])
plt.imshow(cero, cmap='gray_r')
plt.savefig("imatges/deteccion/nosetest.png")


end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")
