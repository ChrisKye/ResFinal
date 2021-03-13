#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:25:41 2021

@author: chriskyee
"""

import numpy as np
import pandas as pd
import tqdm as tqdm
import pickle
import tensorflow as tf
from tensorflow import keras

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

labels_numerical = pd.DataFrame(labels_numerical, columns=['Valence','Arousal'])
labels_categorical = pd.DataFrame(labels_categorical, columns=['Valence','Arousal'])



##Reading in Data

##Normalize Data

##Training & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

##One Hot Encoding

##Creating model

##Compile model

##Model Evaluation (Test)
