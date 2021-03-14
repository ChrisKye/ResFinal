#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:25:41 2021

@author: chriskyee
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

######## Creating Categorial Lables ########
deap_folder = '/Users/chriskye 1/Desktop/DEAP/data_preprocessed_python/'
file_list_test = ['s01.dat', 's02.dat'] ## test directory
file_list = ['s01.dat', 's02.dat', 's03.dat', 's04.dat', 's05.dat', 's06.dat',
             's07.dat', 's08.dat', 's09.dat', 's10.dat', 's11.dat', 's12.dat',
             's13.dat', 's14.dat', 's15.dat', 's16.dat', 's17.dat', 's18.dat',
             's19.dat', 's20.dat', 's21.dat', 's22.dat', 's23.dat', 's24.dat',
             's25.dat', 's26.dat', 's27.dat', 's28.dat', 's29.dat', 's30.dat',
             's31.dat', 's32.dat']

## OG label list
labels_numerical = np.zeros((1,2))
for filename in tqdm(file_list):
    data = pickle.load(open(deap_folder + filename, 'rb'), encoding = 'bytes')
    label_raw = data[b'labels'][...,0:2]
    labels_numerical = np.append(labels_numerical, label_raw, axis=0)
labels_numerical = labels_numerical[1:,...]

## Labels to categorical
labels_categorical = np.zeros((1280,2))
for row in range(1280):
    labels_categorical[row,0] = (labels_numerical[row,0] > 5)
    labels_categorical[row,1] = (labels_numerical[row,1] > 5)

labels_categorical = np.repeat(labels_categorical, [60]*len(labels_categorical), axis = 0)

labels_numerical = pd.DataFrame(labels_numerical, columns=['Valence','Arousal'])
labels_categorical = pd.DataFrame(labels_categorical, columns=['Valence','Arousal'])

## Tensorboard configuration
import os
root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H:%M:%S')
    return os.path.join(root_logdir, run_id)

######## Model Building - Feedforward Network (FNN) ########

## Reading in Features & Normalization
infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/fullFeatures.dat', 'rb'))
featureFull = pickle.load(infile)

X = keras.utils.normalize(featureFull, axis = 0)

## Val - Data Prep
X_train, X_test, y_train_val, y_test_val = train_test_split(X, labels_categorical['Valence'], test_size=0.2, random_state=42)

y_train_val_hot = keras.utils.to_categorical(y_train_val)
y_test_val_hot = keras.utils.to_categorical(y_test_val)

## VAL - Model Architecture
model = keras.models.Sequential()
model.add(keras.layers.Dense(200, input_dim = 288, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(50, activation = 'relu'))
model.add(keras.layers.Dense(2, activation = 'sigmoid'))

## VAL - Compile & Run
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

history = model.fit(X_train, y_train_val_hot, epochs = 50, 
                    validation_split = 0.2,
                    callbacks = [tensorboard_cb])

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

## VAL - Model Evaluation (Test)
model.evaluate(X_test, y_test_val_hot)

## VAL - Save Model
model.save('mlp3_val.h5')


## ARS - Data Prep
X_train, X_test, y_train_ars, y_test_ars = train_test_split(X, labels_categorical['Arousal'], test_size=0.2, random_state=42)

y_train_ars_hot = keras.utils.to_categorical(y_train_ars)
y_test_ars_hot = keras.utils.to_categorical(y_test_ars)

## ARS - Model Architecture
model = keras.models.Sequential()
model.add(keras.layers.Dense(200, input_dim = 288, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(50, activation = 'relu'))
model.add(keras.layers.Dense(2, activation = 'sigmoid'))

## ARS - Compile & Run
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

history = model.fit(X_train, y_train_ars_hot, epochs = 50, 
                    validation_split = 0.2,
                    callbacks = [tensorboard_cb])

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

## ARS - Model Evaluation (Test)
model.evaluate(X_test, y_test_ars_hot)

## ARS - Save Model
model.save('mlp3_ars.h5')

######## Model Building - Convolutional Neural Network (CNN) ########

## Reading in Features & Normalization
infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/rawFeatures.dat', 'rb'))
feature2d = pickle.load(infile)

X = keras.utils.normalize(feature2d, axis = 1)
X = X.reshape((76800, 32, 128))

## ARS - Data Prep
X_train, X_test, y_train_ars, y_test_ars = train_test_split(X, labels_categorical['Arousal'], test_size=0.2, random_state=42)

X_train = X_train.reshape(-1,32,128,1)
X_test = X_test.reshape(-1, 32, 128, 1)

y_train_ars_hot = keras.utils.to_categorical(y_train_ars)
y_test_ars_hot = keras.utils.to_categorical(y_test_ars)

## ARS - Model Architecture
# cnn_model1 = keras.models.Sequential([
#     keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', 
#                         input_shape = [32,128,1]),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'),
#     keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same'),
#     keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same'),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation = 'relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(2, activation = 'sigmoid')
#     ])

cnn_model1 = keras.models.Sequential([
    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', 
                        input_shape = [32,128,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation = 'sigmoid')
    ])
    
cnn_model1.summary()

## ARS - Compile & Run
cnn_model1.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

history = cnn_model1.fit(X_train, y_train_ars_hot, epochs = 3, 
                    validation_split = 0.2,)
                    callbacks = [tensorboard_cb])

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

## ARS - Model Evaluation (Test)
model.evaluate(X_test, y_test_ars_hot)





