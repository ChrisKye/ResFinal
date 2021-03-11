#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:35:40 2021

@author: chriskyee
"""

import os
import numpy as np
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

## generate 2400 training examples with 1 second epochs per participant
processedRaw = {}
deap_folder = '/Users/chriskye 1/Desktop/DEAP/data_preprocessed_python/'
file_list_test = ['s01.dat', 's02.dat'] ## test directory
file_list = ['s01.dat', 's02.dat', 's03.dat', 's04.dat', 's05.dat', 's06.dat',
             's07.dat', 's08.dat', 's09.dat', 's10.dat', 's11.dat', 's12.dat',
             's13.dat', 's14.dat', 's15.dat', 's16.dat', 's17.dat', 's18.dat',
             's19.dat', 's20.dat', 's21.dat', 's22.dat', 's23.dat', 's24.dat',
             's25.dat', 's26.dat', 's27.dat', 's28.dat', 's29.dat', 's30.dat',
             's31.dat', 's32.dat']

for filename in tqdm(file_list):
    data = pickle.load(open(deap_folder + filename, 'rb'), encoding = 'bytes')
    data = data[b'data'] ##40 x 40 x 8064 (video x channel x data)
    data = data[...,0:32,384:] ##40 x 32 x 7680 cuts the last 3 seconds

    output = np.zeros((32,128,2400)) ##32 x 128 x 2400 (channel x time x epoch)
    for i in range(32):
        for j in range(40):
            for k in range(60):
                output[i, 0:128, j*60 + k] = data[j,i, k*128:(k+1)*128]

    dictKey = "Subject_" + filename[1:3]
    processedRaw[dictKey] = output

    print("COMPLETED " + filename)

##pickle output file (dictionary with 32 elements)
outfile = open('/Users/chriskye 1/Desktop/ResFinal/Data/channelPSD.dat', 'wb')
pickle.dump(processedRaw, outfile)
outfile.close()

## Feature Extraction Function (PSD using FFT for each band), relative power
def relativeWelchPSD(data, band, win, sf):
    from scipy import signal
    from scipy import fft
    from scipy.integrate import simps

    ##PSD using Welch's Method
    window = win * sf
    freqs, psd = signal.welch(data, sf, nperseg=window)

    ##Band power using Simpson's Rule
    band = np.asarray(band)
    low, high = band
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs > low, freqs <= high)
    bp = simps(psd[idx_band], dx=freq_res)
    bp /= simps(psd, dx=freq_res) ## finds relative power instead of absolute
    return bp

## Extract for each epoch
channelPSD = np.empty((32, 8, 1)) 

keys = list(processedRaw.keys())
for i in range(32): ## subjects
    print("\nSTARTED SUBJECT NO. " + str(i+1))
    sub = processedRaw[keys[i]]
    rawPSD = np.empty((32, 8, 2400)) ## channel x band x epoch
    for channelNo in tqdm(range(32)): ## channels
        for epoch in range(2400): 
                ##Low Theta
                rawPSD[channelNo, 0, epoch] = relativeWelchPSD(sub[channelNo,...,epoch], [4,6], 1, 128) 
                
                ##High THeta
                rawPSD[channelNo, 1, epoch] = relativeWelchPSD(sub[channelNo,...,epoch], [6,8], 1, 128)

                ##Low Alpha
                rawPSD[channelNo, 2, epoch] = relativeWelchPSD(sub[channelNo,...,epoch], [8,10.5], 1, 128)

                ##High Alpha
                rawPSD[channelNo, 3, epoch] = relativeWelchPSD(sub[channelNo,...,epoch], [10.5,13], 1, 128)

                ##Low Beta
                rawPSD[channelNo, 4, epoch] =  relativeWelchPSD(sub[channelNo,...,epoch], [13,21.5], 1, 128)

                ##High Beta
                rawPSD[channelNo, 5, epoch] = relativeWelchPSD(sub[channelNo,...,epoch], [21.5,30], 1, 128)

                #Low Gamma
                rawPSD[channelNo, 6, epoch] =  relativeWelchPSD(sub[channelNo,...,epoch], [30,37.5], 1, 128)

                #High Gamma
                rawPSD[channelNo, 7, epoch] = relativeWelchPSD(sub[channelNo,...,epoch], [37.5,45], 1, 128)
    print("COMPLETED SUBJECT NO. " + str(i+1))
    channelPSD = np.dstack((channelPSD, rawPSD))
channelPSD = channelPSD[..., 0:8,1:]

##pickle output file (dictionary with 32 elements)
outfile = open('/Users/chriskye 1/Desktop/ResFinal/Data/channelPSD.dat', 'wb')
pickle.dump(channelPSD, outfile)
outfile.close()

