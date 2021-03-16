#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:20:29 2021

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

######## Tensorboard configuration ########
import os
root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H:%M:%S')
    return os.path.join(root_logdir, run_id)

########## Reading in Features & Normalization ########
infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/labels.dat', 'rb'))
labels_categorical = pickle.load(infile)

infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/channelPSD.dat', 'rb'))
feature2d = pickle.load(infile)

X = keras.utils.normalize(feature2d, axis = 2)
X = feature2d.reshape((76800, 32, 8))

######## Model Building - Convolutional Neural Network (CNN) - Arousal ########

## ARS - Data Prep
X_train, X_test, y_train_ars, y_test_ars = train_test_split(X, labels_categorical['Arousal'], test_size=0.2, random_state=42)

# X_train = X_train.reshape(-1,32,128,1)
# X_test = X_test.reshape(-1, 32, 128, 1)
X_train = np.expand_dims(X_train, axis = 3)
X_test = np.expand_dims(X_test, axis = 3)

y_train_ars_hot = keras.utils.to_categorical(y_train_ars, 2)
y_test_ars_hot = keras.utils.to_categorical(y_test_ars, 2)

## ARS - Model Architecture
# cnn_model1 = keras.models.Sequential([
#     keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', 
#                         input_shape = [32,8,1]),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
#     keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
#     keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
#     keras.layers.MaxPooling2D(2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation = 'relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(2, activation = 'sigmoid')
#     ])

cnn_model1 = keras.models.Sequential([
    keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', 
                        input_shape = [32,8,1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation = 'sigmoid')
    ])
# cnn_model1 = keras.Sequential([
#         keras.Input(shape=[32,8,1]),
#         keras.layers.Conv2D(32, kernel_size=2, activation="relu"),
#         keras.layers.MaxPooling2D(2),
#         keras.layers.Conv2D(64, kernel_size=2, activation="relu"),
#         keras.layers.MaxPooling2D(2),
#         keras.layers.Flatten(),
#         keras.layers.Dense(2, activation="sigmoid"),
#     ])
    
cnn_model1.summary()

## ARS - Compile & Run
cnn_model1.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

history = cnn_model1.fit(X_train, y_train_ars_hot, epochs = 5, 
                    validation_split = 0.2,
                    callbacks = [tensorboard_cb],
                    )

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

## ARS - Model Evaluation (Test)
cnn_model1.evaluate(X_test, y_test_ars_hot)

## ARS - F1 Score
from sklearn.metrics import classification_report

y_pred = cnn_model1.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test_ars, y_pred_bool, digits = 4))


######## Model Building - Convolutional Neural Network (CNN) - Arousal ########

## VAL - Data Prep
X_train, X_test, y_train_ars, y_test_ars = train_test_split(X, labels_categorical['Valence'], test_size=0.2, random_state=42)

# X_train = X_train.reshape(-1,32,128,1)
# X_test = X_test.reshape(-1, 32, 128, 1)
X_train = np.expand_dims(X_train, axis = 3)
X_test = np.expand_dims(X_test, axis = 3)

y_train_ars_hot = keras.utils.to_categorical(y_train_ars, 2)
y_test_ars_hot = keras.utils.to_categorical(y_test_ars, 2)

## VAL - Model Architecture
cnn_model2 = keras.Sequential([
        keras.Input(shape=[32,8,1]),
        keras.layers.Conv2D(32, kernel_size=2, activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, kernel_size=2, activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(2, activation="sigmoid"),
    ])
    
cnn_model2.summary()

## VAL - Compile & Run
cnn_model1.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())

history = cnn_model1.fit(X_train, y_train_ars_hot, epochs = 15, 
                    validation_split = 0.2,
                    #callbacks = [tensorboard_cb],
                    )

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

## VAL - Model Evaluation (Test)
cnn_model1.evaluate(X_test, y_test_ars_hot)

## VAL - F1 Score
from sklearn.metrics import classification_report

y_pred = cnn_model1.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test_ars, y_pred_bool, digits = 4))

