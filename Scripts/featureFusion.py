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
from sklearn.decomposition import PCA

######## Reading in music features from musicFeatures.mat to df ########
mf_raw = loadmat('/Users/chriskye 1/Desktop/ResFinal/Data/musicFeatures.mat')
mf_raw = mf_raw['df']

musicFeatures_raw = np.zeros((40,33))

##creating feature list
for i in range(40):
    for index in range(0,14):
        musicFeatures_raw[i,index] = mf_raw[0,i][index][0][0]
    for index2 in range(14,27):
        musicFeatures_raw[i,index2] = mf_raw[0,i][14][index2-14][0]
    for index3 in range(27,33):
        musicFeatures_raw[i,index3] = mf_raw[0,i][index-12][0][0]
musicFeatures_raw = np.delete(musicFeatures_raw, 6, 1)

musicFeatures = np.repeat(musicFeatures_raw, [60]*len(musicFeatures_raw), 0)
musicFeatures = np.tile(musicFeatures, (32,1))
    
musicFeatures = pd.DataFrame(musicFeatures,columns=['RMS', 'Fluctuation Peak',
    'Fluctuation Centroid','Tempo','Pulse Clarity', 'Mean Attack Time',
    'Zero Cross Rate', 'Spectral Centroid',  'Spectral Spread',
    'Spectral Skewness', 'Spectral Kurtosis', 'Spectral Flatness', 'Spectral Entropy',
    'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9',
    'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Harmonic Change', 'Key Clarity',
    'Majorness', 'Roughness', 'Chroma Std', 'Novelty'])

######## Reading in EEG Features to df ########
infile = open('/Users/chriskye 1/Desktop/ResFinal/Data/channelPSD.dat', 'rb')
ef_raw = pickle.load(infile)

## reshape to 2d
eegFeatures = ef_raw.transpose(2,0,1).reshape(76800, 256)

## assign colnames & make into pd df
colnames = []
for i in range(32):
    for j in range(8):
        name = 'Chn' + str(i+1) + '_' + 'band' + str(j+1)
        colnames.append(name)
eegFeatures = pd.DataFrame(eegFeatures, columns=colnames)

######## Feature Fusion ########
fullFeatures = musicFeatures.join(eegFeatures)

outfile = open('/Users/chriskye 1/Desktop/ResFinal/Data/fullFeatures.dat', 'wb')
pickle.dump(fullFeatures, outfile)
outfile.close()

######## PCA ########
pca = PCA(n_components = )






  