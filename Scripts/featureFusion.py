#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:55:19 2021

@author: chriskyee
"""

from scipy.io import loadmat
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

## Reading in music features from musicFeatures.mat
mf_raw = loadmat('/Users/chriskye 1/Desktop/ResFinal/Data/musicFeatures.mat')
mf_raw = mf_raw['df']

musicFeatures = np.zeros((40,33))

for i in range(40):
        for index in range(0,14):
            musicFeatures[i,index] = mf_raw[0,i][index][0][0]
        for index2 in range(14,27):
            musicFeatures[i,index2] = mf_raw[0,i][14][index2-14][0]
        for index3 in range(27,33):
            musicFeatures[i,index3] = mf_raw[0,i][index-12][0][0]

musicFeatures = pd.DataFrame(musicFeatures,columns=['RMS', 'Fluctuation Peak',
    'Fluctuation Centroid','Tempo','Pulse Clarity', 'Mean Attack Time',
    'Mean Attack Slope', 'Zero Cross Rate', 'Spectral Centroid',  'Spectral Spread',
    'Spectral Skewness', 'Spectral Kurtosis', 'Spectral Flatness', 'Spectral Entropy',
    'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9',
    'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Harmonic Change', 'Key Clarity',
    'Majorness', 'Roughness', 'Chroma Std', 'Novelty'])

## Reading in EEG Features
infile = open('/Users/chriskye 1/Desktop/ResFinal/Data/channelPSD.dat', 'rb')
ef_raw = pickle.load(infile)



## Feature Fusion

## Creating Categorical Labels
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








  