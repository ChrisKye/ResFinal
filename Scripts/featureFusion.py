#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:55:19 2021

@author: chriskyee
"""

from scipy.io import loadmat
import pandas as pd
import numpy as np

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

musicFeatures = pd.DataFrame(musicFeatures,columns=['RMS', 
                                      'Fluctuation Peak', 
                                      'Fluctuation Centroid', 
                                      'Tempo',
                                      'Pulse Clarity',
                                      'Mean Attack Time',
                                      'Mean Attack Slope',
                                      'Zero Cross Rate',
                                      'Spectral Centroid',
                                      'Spectral Spread',
                                      'Spectral Skewness',
                                      'Spectral Kurtosis',
                                      'Spectral Flatness',
                                      'Spectral Entropy',
                                      'MFCC1',
                                      'MFCC2',
                                      'MFCC3',
                                      'MFCC4',
                                      'MFCC5',
                                      'MFCC6',
                                      'MFCC7',
                                      'MFCC8',
                                      'MFCC9',
                                      'MFCC10',
                                      'MFCC11',
                                      'MFCC12',
                                      'MFCC13',
                                      'Harmonic Change',
                                      'Key Clarity',
                                      'Majorness',
                                      'Roughness',
                                      'Chroma Std',
                                      'Novelty'])
    
                             
                                      
                                      