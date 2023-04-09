# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:15:20 2019

@author: xb
"""

import joblib
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def pack(dir_stft,dir_wavelet,label):       
    feature_stft_list=[]
    feature_wavelet_list=[]
    label_list=[] 
    for file in tqdm(os.listdir(dir_stft)):
        I_stft = Image.open(dir_stft+file).convert('L')
        I_wavelet = Image.open(dir_wavelet+file).convert('L')
        I_stft = np.array(I_stft)
        I_wavelet = np.array(I_wavelet)
        feature_wavelet_list.append(I_wavelet)
        feature_stft_list.append(I_stft)
        label_list.append(label)
    return feature_stft_list,feature_wavelet_list,label_list
    
def pack_hospital(dir_stft, dir_wavelet, label):
    index = 0
    stft_array = [[]]
    wavelet_array = [[]]
    label_array = [[]]
    temp_stft_array = []
    temp_wavelet_array = []
    temp_label = []
    last_filename =''
    files = os.listdir(dir_stft)
    files.sort()
    for file in tqdm(files):

        filename = os.path.splitext(file)[0]
        wavelet_file = dir_wavelet+filename+'.png'
        filename_index = filename[:-7]
        # print(filename_index)
        I_stft = Image.open(dir_stft+file).convert('L')
        I_wavelet = Image.open(dir_wavelet+file).convert('L')
        I_stft = np.array(I_stft)
        I_wavelet = np.array(I_wavelet)

        if len(temp_stft_array) == 0:
            temp_stft_array.append(I_stft)
            temp_wavelet_array.append(I_wavelet)
            temp_label.append(label)
            last_filename = filename_index


        # new filename
        elif filename_index != last_filename:
            # print(filename)
            # print(f'filename_index is {filename_index}') 
            # print(last_filename)
            joblib.dump((temp_stft_array, temp_wavelet_array, temp_label), open('./data/bi_test_pack/hospital_pack/{}.p'.format(filename), 'wb'))

            temp_label.clear()
            temp_stft_array.clear()
            temp_wavelet_array.clear()

            temp_stft_array.append(I_stft)
            temp_wavelet_array.append(I_wavelet)
            temp_label.append(label)
            last_filename = filename_index
        else:
            temp_stft_array.append(I_stft)
            temp_wavelet_array.append(I_wavelet)
            temp_label.append(label)
    


    
if __name__ == '__main__':
# ----------------
    # pack for testset
    # stft0,wavelet0,label0 = pack_hospital('./data/bi_test_stft/normal/','./data/bi_test_wavelet/normal/',0)
    # stft1,wavelet1,label1 = pack_hospital('./data/bi_test_stft/abnormal/','./data/bi_test_wavelet/abnormal/',1)
    # stft = stft0+stft1
    # wavelet = wavelet0+wavelet1
    # label = label0+label1
    # print(label[0])





    # for i in tqdm(range(len(stft))):
    #     joblib.dump((stft[i], wavelet[i], label[i]), open('./data/bi_test_pack/hospital_pack/wavelet_stft_test_{}.p'.format(i), 'wb'))
    # # joblib.dump((stft,wavelet, label), open('./data/bi_test_pack/wavelet_stft_test.p', 'wb'))


    pack_hospital('./data/bi_test_stft/normal/','./data/bi_test_wavelet/normal/',0)
    pack_hospital('./data/bi_test_stft/abnormal/','./data/bi_test_wavelet/abnormal/',1)
