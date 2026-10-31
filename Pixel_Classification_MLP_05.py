# -*- coding: utf-8 -*-

#%% Exercicio Perceptron com 1 Neuronioe função de ativação gaussiana

#%% Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from keras import ops as K

# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Activation
# import pandas as pd
# from scipy import signal
# import scipy

#%% funções
def gaussian(x, u, std):
    return K.exp(-1*(((0.5*(x-u))/std)**2))

# u = 2.5
# std = 0.01
# x = np.arange(-10,10,0.1)
# g = K.exp(-1*(((0.5*(x-u))/std)**2))
# plt.figure()
# plt.plot(g)

#%% Ler Imagens para treinamento
I_treino = skimage.img_as_float(cv2.imread('I01.pgm', 0))
GS_treino = skimage.img_as_float(cv2.imread('GS01.pgm', 0))

(M,N) = np.shape(I_treino)
# Transforma as Imagens e Colunas para a entrada da rede
Ivector1 = I_treino.flatten(order='C')
GSvector1 = GS_treino.flatten(order='C')
#Ivector1=GSvector1

#%% Ler Imagens para teste
I_test = skimage.img_as_float(cv2.imread('I02.pgm', 0))
GS_test = skimage.img_as_float(cv2.imread('GS02.pgm', 0))
# Transforma as Imagens e Colunas para a entrada da rede
Ivector2 = I_test.flatten(order='C')
GSvector2 = GS_test.flatten(order='C')

#Ivector2=GSvector2

#%% Dados
# importa os dados do excel e convert para a Matriz
x_treinamento = [Ivector1]
x_treinamento = np.transpose(x_treinamento)
y_treinamento = GSvector1

x_teste = [Ivector2]
x_teste = np.transpose(x_teste)
y_teste = GSvector2>0.5

#%% ############################################
#%% Construção da Rede
# model = Sequential() # Desta vez não vamos usar sequencial. Usar o formato Standalone Input layer

#%% Só 1 Neuronio com 1 entrada
inputs = tf.keras.layers.Input((1,)) # vetor de 1 dimensão atributos

L1 = tf.keras.layers.Dense(1, kernel_initializer='ones', bias_initializer='zeros',
input_shape=(1,))(inputs)

L1 = gaussian(L1, u = 0.835, std = 0.02) # Modelagem da Função de Ativação


#%% Compilar a Rede --> Dados / Configuração de treinamento

model = tf.keras.Model(inputs=[inputs], outputs = [L1])
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =
['accuracy'])
# optimizer --> 'adam' é uma versão do gradient descendent
# loss --> função de calculo do erro
# metrics --> Metrica de avaliação do algoritmo
#    como é um vetor, pode ser mais de uma metrica

#%% Treinamento
#model.fit(x_treinamento, y_treinamento, batch_size = (N*M), nb_epoch = 100)

model.fit(x_treinamento, y_treinamento, validation_data = (x_teste, y_teste),
batch_size = (N*M), epochs = 100)

# previsores_treinamento --> Dados de entrada
# classe_treinamento --> Dados de Saida
# batch_size --> de quantos em quantos registros a atualização dos pesos são feitas --> Por que estamos usando o optimizer = 'adam'
# nb_epoch --> numero de epocas de treinamento

#%% Etapa de Teste
previsoes = model.predict(x_teste)

#%% Binarização com limiar de 0.5
Previsoes = previsoes >0.5


#%% Avaliação, Análise da etapa de Testes

precisao = accuracy_score(y_teste, Previsoes)
# Precisão/Acuracia Final
matriz_de_confusao = confusion_matrix(y_teste, Previsoes)
# Matriz de Confusao

#%% Exibe Imagens
previsoesmatrix = previsoes.reshape(N, -1) # Matriz da imagem
Previsoesmatrix = Previsoes.reshape(N, -1) # Matriz da imagem em Booleano
# previsoesmatrix = previsoes.ravel
# Previsoesmatrix = Previsoes.ravel
#% Exibe Imagens
plt.figure()
plt.title('previsoesmatrix')
plt.imshow(previsoesmatrix, cmap='gray') # cmap='jet'

# plt.figure()
# plt.title('Previsoesmatrix')
# plt.imshow(Previsoesmatrix, cmap='gray') # cmap='jet'
