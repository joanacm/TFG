# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:47:14 2021

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

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA



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
del(train['label'])

#guardamos que número representa cada imagen (fila)
num_test = test['label']
#eliminamos la variable label 
del(test['label'])


##### PREDECIR 


lda_pred = LDA()

train_proy = lda_pred.fit_transform(train, num_train)
coefs = lda_pred.scalings_

suma = 0

for i in range(0, len(num_test)):
    
    #cargamos una imagen de un dígito desconocido
    ejemplo_test = test.iloc[i].values
    
    #projectam l'exemple a les coordenades anteriors (veps)
    test_proy = np.dot(ejemplo_test, coefs)
    
    errores = distancia(test_proy, train_proy)
    index_train = num_train[np.argmin(errores)]
    index_text = num_test[i]
    
    #comprobamos si se ha acertado
    if index_text != index_train:
        suma = suma + 1
        test_x = test.iloc[i].values.reshape([28,28])
        plt.imshow(test_x, cmap='gray_r')
        plt.title(u'Resultado: %s' % str(index_train))
        plt.savefig("imatges\deteccionLDA\image_erronea_%d.png"%(i+1))



varianza_explicada_LDA = lda_pred.explained_variance_ratio_


##### PREDECIR TOTALMENT SKLEARN

lda = LDA()
datos_lda = lda.fit_transform(train, num_train)
varianza_explicada_LDA = lda.explained_variance_ratio_

prediccion=lda.predict(test)

error= 0
#vemos el numero de imagenes mal clasificadas
for i in range(0, len(num_test)):
    if num_test[i] != prediccion[i]:
        error += 1 



        
##### COMPRIMIR

k=9
lda_compr = LDA(n_components = k)
#ajustamos modelo
trainLDA = lda_compr.fit(train, num_train)
#datos proyectados en los nuevos ejes
proyectados = trainLDA.transform(train)
#veps de la matriz = coef de la funciones discriminantes = U_k
U_k = lda_compr.scalings_[:, 0:(k+1)]


#invertimos LDA
U_k_pseudoinv = np.linalg.pinv(U_k) 
#print(proyectados.shape)
X_compr = np.dot(proyectados, U_k_pseudoinv) +  lda_compr.xbar_
num = X_compr[0].reshape([28,28])
plt.imshow(num, cmap='gray_r')
plt.savefig("num.png")






end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")

































