#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:25:41 2021

@author: chriskyee
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

######## Reading in Features & Normalization ######
infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/labels.dat', 'rb'))
labels_categorical = pickle.load(infile)

infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/fullFeatures.dat', 'rb'))
featureFull = pickle.load(infile)

X = keras.utils.normalize(featureFull, axis = 0)


######## Tensorboard configuration ########
import os
root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H:%M:%S')
    return os.path.join(root_logdir, run_id)

######## Model Building - Feedforward Network (FNN) - Valence ########

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

## VAL - F1 Score
from sklearn.metrics import classification_report

y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test_val, y_pred_bool, digits = 4))

## VAL - Save Model
model.save('mlp3_val.h5')

######## Model Building - Feedforward Network (FNN) - Arousal ########

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

## ARS - F1 Score
from sklearn.metrics import classification_report

y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test_ars, y_pred_bool, digits = 4))


## ARS - Save Model
model.save('mlp3_ars.h5')





