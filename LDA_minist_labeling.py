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

lda = LDA()
datos_lda = lda.fit_transform(train, num_train)
pesos_func_discriminantes = lda.coef_
varianza_explicada_LDA = lda.explained_variance_ratio_

prediccion=lda.predict(test)

error= 0
#vemos el numero de imagenes mal clasificadas
for i in range(0, len(num_test)):
    if num_test[i] != prediccion[i]:
        error += 1 
    
end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")

