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
train_original = train

n, p = train.shape
digitos = [0,1,2,3,4,5,6,7,8,9] #pd.unique(train['label'])

#guardamos que número representa cada imagen (fila)
num_train = train['label']
#del(train_original['label'])

columnas = train.columns.values
media_por_clases = pd.DataFrame(columns = digitos)

for c in range(0, 10):
    is_c = train['label'] == c
    train_c = train[is_c]
    del(train_c['label'])
    media_por_clases[c] = train_c.mean() #np.mean(train_c,0)

S_w = np.zeros((p-1,p-1))

for i in range(0,n):
    train_menos_media = train.iloc[i, 1:p] - np.array(media_por_clases[num_train[i]])
    S_w = S_w + ((np.array(train_menos_media)).T).dot(train_menos_media)
print(S_w)



end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")


    

