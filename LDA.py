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


def ADL(x, labels):
    nombre_grupos = np.unique(labels)
    
    #Calcula media por columna y grupo
    vect_mu_columna_x_grupo = []
    for i in nombre_grupos:
        vect_mu_columna_x_grupo.append(np.mean(x[labels == i], axis = 0))
    
    #media por columna
    mu_x = (np.mean(x, axis = 0))[np.newaxis, :]
    
    #inicializamos matriz covarianza entre clases
    E = np.zeros((x.shape[1],x.shape[1]))
    
    for i, mu_vect in enumerate(vect_mu_columna_x_grupo):
        
        #numero de observaciones por grupo
        n = x[labels == i].shape[0]
        mu_vect = mu_vect[np.newaxis, :]
        mu1_mu2 = np.array(mu_vect - mu_x)
        E = E + n*np.dot(mu1_mu2.T, mu1_mu2)
    
    #inicializamos matriz covarianza dentro de clases
    D = np.zeros((x.shape[1],x.shape[1]))
    Dg = np.zeros((x.shape[1],x.shape[1]))

    for g in range(0,10):
        for i in range (0, len(labels)):
            if labels[i] == g:
                x_mu = np.array(x.iloc[i] -  vect_mu_columna_x_grupo[g], ndmin = 2)
                x_muT = x_mu.T
                Dg = Dg + np.dot(x_muT, x_mu)
        D = D + Dg
    
    D_inv = np.linalg.inv(D)
    W= np.dot(D_inv, E)
    
    vaps, veps = np.linalg.eig(W)
    return veps



test = pd.read_csv('data/mnist/test.csv')
train = pd.read_csv('data/mnist/train.csv')

#guardamos que número representa cada imagen (fila)
num_train = train['label']
del(train['label'])


veps = ADL(train, num_train)

end = time.time()
print("El tiempo de ejecución es de "+ str(end - start) + " segundos.")

