# -*- coding: utf-8 -*-
"""
Created on Fri Aug 06 17:47:25 2021

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



start = time.time()


test = pd.read_csv('data/mnist/test.csv')
train = pd.read_csv('data/mnist/train.csv')

n, p = train.shape
digitos = [0,1,2,3,4,5,6,7,8,9] #pd.unique(train['label'])

matriz_pixeles = np.array(train.iloc[0:n, 1:p])

#guardamos que número representa cada imagen (fila)
num_train = train['label']
#del(train_original['label'])

columnas = train.columns.values
media_por_clases = pd.DataFrame(columns = digitos)

S_w = np.zeros((p-1,p-1))
S_b = np.zeros((p-1,p-1))
N = np.zeros(len(digitos))

for c in range(0, 10):
    is_c = train['label'] == c
    train_c = train[is_c]
    N[c] = len(train_c)
    del(train_c['label'])
    media_por_clases[c] = train_c.mean() #np.mean(train_c,0)

# Calculo matriz de covarianzas de una misma clase
for i in range(0,n):
    train_menos_media = train.iloc[i, 1:p] - np.array(media_por_clases[num_train[i]])
    S_w = S_w + np.dot(np.reshape(np.array(train_menos_media), (784,1)), np.array([train_menos_media]))

media_matriz = 0
media_matriz = np.mean(matriz_pixeles)

# Calculo de la matriz de covarianzas entre clases
for i in range(0,n):
    media_i_menos_media = np.array(media_por_clases[num_train[i]]) - np.repeat(media_matriz, p-1)
    S_b = S_b + N[num_train[i]]*(np.dot(np.reshape(np.array(media_i_menos_media), (784,1)), np.array([media_i_menos_media])))


end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")


    

