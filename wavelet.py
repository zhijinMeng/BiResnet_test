#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:40:07 2019

@author: mayi
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import pywt
import math
import os
import librosa.display as display
from scipy.signal import butter, lfilter
from tqdm import tqdm
import json
import soundfile as sf


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    print(low)
    print(high)
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_txt(dir):

    a = np.loadtxt(dir)
    return a  

def wavelet(sig):
    cA, out = pywt.dwt(sig, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    A = cA
    
    for i in range(6):
        cA, cD = pywt.dwt(A, 'db8')
        A = cA
        out = np.hstack((out,cD))

    out = np.hstack((out,A))
        
    return out

def reshape(matrix):
    num = matrix.shape[0]
    length = math.ceil(np.sqrt(num))
    zero = np.zeros([np.square(length)-num,])
    matrix = np.concatenate((matrix,zero))
    out = matrix.reshape((length,length))
    return out

def Normalization(x):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    for i in range(len(x)):
        
        x[i] = float(x[i]-min_x)/(max_x-min_x)
           
    return x
#    save_pic('../data/train_set', '../analysis/wavelet/train/')
# save_pic('./Dataset/trainset/wav/', './Dataset/trainset/wavelet/')
def save_pic(wav_dir,save_dir):

    for file in tqdm(os.listdir(wav_dir)):
        filename = os.path.splitext(file)[0] # get the file name
       
# label contains ['Normal', 'Fine Crackle', 'Wheeze', 'Coarse Crackle', 'Wheeze+Crackle', 'Stridor', 'Rhonchi']
        label = wav_dir[-9:-1]

        sig, fs=sf.read(wav_dir+'/'+file)
        sig = Normalization(sig)
        # if fs>4000:
        #     sig = butter_bandpass_filter(sig, 1, 4000, fs, order=3)
            
        wave = wavelet(sig)
        xmax=max(wave)
        xmin=min(wave)
        wave=(255-0)*(wave-xmin)/(xmax-xmin)+0       
        wave = reshape(wave)
        display.specshow(wave)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
    
        if label == 'abnormal':
            plt.savefig(save_dir+'abnormal/'+filename+'.png')
        else:
            plt.savefig(save_dir+'normal/'+filename+'.png')
        plt.close()
            
if __name__ == '__main__':
    train_wav = './Dataset/trainset/wav/'
    train_stft = './Dataset/trainset/stft/'
    train_json = './Dataset/trainset/json/'

    test_wav_normal = './data/testdata/normal/'
    test_wav_abnormal = './data/testdata/abnormal/'
    test_stft = './data/bi_test_wavelet/'


    val_wav = './Dataset/validation/wav/'
    val_stft = './Dataset/validation/stft/'
    val_json = './Dataset/validation/json/'
    # save_pic(train_wav, train_wavelet, train_json)


    save_pic(test_wav_normal, test_stft)
    save_pic(test_wav_abnormal, test_stft)

