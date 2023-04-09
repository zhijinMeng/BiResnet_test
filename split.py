# This file is aiming to slicing the original voice data and splitting the train/test data
import os
import json
import shutil
import sklearn
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from pydub.utils import make_chunks
import wave
#load the wav file


def split_random(wav_source_pathway):
    for file in os.listdir(wav_source_pathway):
        file_name = os.path.splitext(file)[0]  
        wav_file = wav_source_pathway + file
        sound = AudioSegment.from_wav(wav_file)
        chunk_length_ms = 2000 # set the chunck lenght as 2 second
        chunks = make_chunks(sound, chunk_length_ms)
    # create the save dir
        if wav_source_pathway[-9:-1] == 'abnormal':
            save_path = './data/testdata/abnormal/'
        else:
            save_path = './data/testdata/normal/'
        # save the splited files
        for i, chunk in enumerate(chunks):
            chunk_name = "{}chunk{}.wav".format(save_path+file_name, i)
            print(f'exporting {chunk_name}')
            chunk.export(chunk_name, format='wav')

wav_source_pathway_abnormal = './data/testdata_original/abnormal/'
wav_source_pathway_normal = './data/testdata_original/normal/'
split_random(wav_source_pathway_normal)
split_random(wav_source_pathway_abnormal)




