#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:43:35 2021

@author: chriskyee
"""
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from tensorflow import keras
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

######## Reading in Features & Normalization ######
infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/labels.dat', 'rb'))
labels_categorical = pickle.load(infile)

infile = (open('/Users/chriskye 1/Desktop/ResFinal/Data/fullFeatures.dat', 'rb'))
featureFull = pickle.load(infile)

X = keras.utils.normalize(featureFull, axis = 0)

######## Model Building - SVM - Valence ########
## Val - Data Prep
X_train, X_test, y_train_val, y_test_val = train_test_split(X, labels_categorical['Valence'], test_size=0.2, random_state=42)

## Model avg
acc = 0
f1 = 0
for i in tqdm(range(200)):
    ## Model
    clf = svm.SVC(kernel='rbf') # rbf Kernel

    ## Train the model using the training sets
    clf.fit(X_train.sample(2000, random_state = i), y_train_val.sample(2000, random_state = i))
    
    ## Predict
    y_pred = clf.predict(X_test.sample(2000, random_state = i))

    ## Print Accuracy
    acc += metrics.accuracy_score(y_test_val.sample(2000, random_state = i), y_pred)
    f1 += metrics.f1_score(y_test_val.sample(2000, random_state = i), y_pred)
print("\nAccuracy: ", acc/200)
print("F1 Score: ", f1/200)

######## Model Building - SVM - Valence ########
## ARS - Data Prep
X_train, X_test, y_train_ars, y_test_ars = train_test_split(X, labels_categorical['Arousal'], test_size=0.2, random_state=42)

## Model avg
acc = 0
f1 = 0
for i in tqdm(range(200)):
    ## Model
    clf = svm.SVC(kernel='rbf') # rbf Kernel

    ## Train the model using the training sets
    clf.fit(X_train.sample(2000, random_state = i), y_train_ars.sample(2000, random_state = i))
    
    ## Predict
    y_pred = clf.predict(X_test.sample(2000, random_state = i))

    ## Print Accuracy
    acc += metrics.accuracy_score(y_test_ars.sample(2000, random_state = i), y_pred)
    f1 += metrics.f1_score(y_test_ars.sample(2000, random_state = i), y_pred)
print("\nAccuracy: ", acc/200)
print("F1 Score: ", f1/200)



