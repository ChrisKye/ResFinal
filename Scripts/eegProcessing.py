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

processedRaw = {}
deap_folder = '/Users/chriskye 1/Desktop/DEAP/data_preprocessed_python/'

file_list_test = ['s01.dat', 's02.dat']
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
    processedRaw[filename] = output

    print("COMPLETED " + filename)

outfile = open('epochedList.dat', 'wb')
pickle.dump(processedRaw, outfile)
outfile.close()

