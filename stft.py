# -*- coding: utf-8 -*-
import scipy.io.wavfile as wav
import os
import shutil
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display as display
from pydub import AudioSegment
from scipy.signal import butter, lfilter  
from tqdm import tqdm
import json
import soundfile as sf
from scipy.io import wavfile

def functToDeleteItems(fullPathToDir):
   for itemsInDir in os.listdir(fullPathToDir):
        if os.path.isdir(os.path.join(fullPathToDir, itemsInDir)):
            functToDeleteItems(os.path.isdir(os.path.join(fullPathToDir, itemsInDir)))
        else:
            os.remove(os.path.join(fullPathToDir,itemsInDir))

def check_fs(dir):
    """check the fe of raw record"""

    for file in os.listdir(dir):
        fs,sig,bits= wav.read(dir+file)
        print(file,fs)

            
def Normalization(x):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    for i in range(len(x)):
        
        x[i] = float(x[i]-min_x)/(max_x-min_x)
           
    return x

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def save_pic(wav_dir,save_dir):
    for file in tqdm(os.listdir(wav_dir)):
        filename = os.path.splitext(file)[0] # get the file name
        # json_file = open(json_dir+filename+'.json')
        # json_file_opened = json.load(json_file)
        # label = json_file_opened['event_annotation'] # get the label
# label contains ['Normal', 'Fine Crackle', 'Wheeze', 'Coarse Crackle', 'Wheeze+Crackle', 'Stridor', 'Rhonchi']
        label = wav_dir[-9:-1]
        sig, fs= sf.read(wav_dir+'/'+file)
        sig = Normalization(sig)
        
        # if fs>4000:
        #     sig = butter_bandpass_filter(sig, 1, 4000, fs, order=3)

        stft = librosa.stft(sig, n_fft=int(0.02*fs), hop_length=int(0.01*fs), window='hann')
        if fs>4000:
            display.specshow(librosa.amplitude_to_db(stft[0:int(len(stft)/2),:],ref=np.max),y_axis='log',x_axis='time')
        else:
            display.specshow(librosa.amplitude_to_db(stft,ref=np.max),y_axis='log',x_axis='time')

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
    test_stft = './data/bi_test_stft/'


    val_wav = './Dataset/validation/wav/'
    val_stft = './Dataset/validation/stft/'
    val_json = './Dataset/validation/json/'


    save_pic(test_wav_normal, test_stft)
    save_pic(test_wav_abnormal, test_stft)



# # clear the specific dirs
    # functToDeleteItems('./data/bi_test_stft/normal')
    # functToDeleteItems('./data/bi_test_stft/abnormal')
    # functToDeleteItems('./data/bi_test_wavelet/normal')
    # functToDeleteItems('./data/bi_test_wavelet/abnormal')
    # functToDeleteItems('./data/bi_test_pack/hospital_pack/')
    # functToDeleteItems('./data/testdata/normal')
    # functToDeleteItems('./data/testdata/abnormal')
