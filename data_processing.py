# This file is aiming to slicing the original voice data and splitting the train/test data
import os
import json
import shutil
import sklearn
from pydub import AudioSegment
from sklearn.model_selection import train_test_split

# copy the original data and paste into workspace - original dataset
def mov_source (original_pathway, target_pathway):
    shutil.copytree(original_pathway, target_pathway)

# slicing the wav files
def slicing_wavs(original_wav_pathway, target_wav_pathway, original_json_pathway, target_json_pathway):
    # get the file name from the json folder
    for file in os.listdir(original_json_pathway):
        file_name = os.path.splitext(file)[0] #json file name
        # file is the json file, here we get the correspongding wav file 
        wav_file = original_wav_pathway + file_name +'.wav'
        json_file = original_json_pathway + file_name +'.json'
        target_json_file = target_json_pathway + file_name + '.json'
        target_wav_file = target_wav_pathway + file_name + '.wav'

        json_file_opened = json.load(open(json_file)) #open the json file 
        # if the video is poor quality, just move it to the folder
        if json_file_opened['record_annotation'] == "Poor Quality":
            poor_quality_json = './Dataset/poor_quality/json/'
            poor_quality_wav = './Dataset/poor_quality/wav/'
            #move the files to the poor quality folder -> './Dataset/poor_quality/'
            shutil.copy(json_file, poor_quality_json+file_name+'.json')
            shutil.copy(wav_file, poor_quality_wav+file_name+'.wav')
        # if the vidceo is in good quality, then slice it based on the events.
        else:
            # slice the wav file based on the json events period
            sound = AudioSegment.from_wav(wav_file) #read the wav file
            for i in range(len(json_file_opened['event_annotation'])):
                start_time = int(list(json_file_opened['event_annotation'][i].values())[0])
                end_time = int(list(json_file_opened['event_annotation'][i].values())[1])
                label = list(json_file_opened['event_annotation'][i].values())[2]
                word = sound[start_time:end_time]
                word.export(target_wav_pathway+file_name+'_'+str(i)+'.wav', format="wav") #write the wav file into new wav dir

                # write the dictionray for json file
                json_dictionary = {
                    'record_annotation': json_file_opened['record_annotation'],
                    'event_annotation': label
                }
                json_object = json.dumps(json_dictionary, indent=4)
                with open(target_json_pathway+file_name+'_'+str(i)+'.json', 'w') as file:
                    file.write(json_object)
                    file.close()

# split train/test/validation set 
def split_train_test_val(original_wav_pathway, original_json_pathway):
    json_list = os.listdir(original_json_pathway)
    # split the data into 0.8 trainset and 0.1 testset + 0.1 validation set
    train_list,test_list = train_test_split(json_list, test_size=0.2)
    test_list,val_list = train_test_split(test_list, test_size=0.5)

    train_wav_path = './Dataset/trainset/wav/'
    train_json_path = './Dataset/trainset/json/'
    test_wav_path = './Dataset/testset/wav/'
    test_json_path = './Dataset/testset/json/'
    val_wav_path = './Dataset/validation/wav/'
    val_json_path = './Dataset/validation/json/'
    pick_up_file(train_list, original_wav_pathway, train_wav_path, original_json_pathway, train_json_path)
    pick_up_file(test_list, original_wav_pathway, test_wav_path, original_json_pathway, test_json_path)
    pick_up_file(val_list, original_wav_pathway, val_wav_path, original_json_pathway, val_json_path)
#pick up the coresponding file in wav and json folders
def pick_up_file(list, source_wav_pathway, target_wav_pathway, source_json_pathway, target_json_pathway):
    for file in list:
        filename = os.path.splitext(file)[0]
        source_wav_file = source_wav_pathway+filename+'.wav'
        target_wav_file = target_wav_pathway+filename+'.wav'
        source_json_file = source_json_pathway+filename+'.json'
        target_json_file = target_json_pathway+filename+'.json'

        shutil.copy(source_wav_file, target_wav_file)
        shutil.copy(source_json_file, target_json_file)

# clear the folder while only keep sub-folders
def claer_folder(file_pathway):
    dir = os.listdir(file_pathway)
    for file in dir:
        file_object = file_pathway+file
        if not os.path.isdir(file_object):
            os.remove(file_object)





if __name__ == '__main__':
    wav_path = './DataBase/train_wav/'
    json_path = './DataBase/train_json/'
    source_wav_path = './Dataset/original/wav/'
    source_json_path = './Dataset/original/json/'
    splited_wav_path = './Dataset/sliced/wav/'
    splited_json_path = './Dataset/sliced/json/'
    

    # # copy the data from the original DataBase to the workspace DataSet
    # mov_source(wav_path, source_wav_path)
    # mov_source(json_path, source_json_path)

    ## slicing the wav files
    # slicing_wavs(source_wav_path, splited_wav_path, source_json_path, splited_json_path)
    
    ## split the train/test/validation set
    # split_train_test_val(splited_wav_path, splited_json_path)

num = 0
abnrmal = 0
for file in os.listdir('./Dataset/poor_quality/json/'):
    num += 1
    with open ('./Dataset/poor_quality/json/'+file) as json_file:
        data = json.load(json_file)
        if data['record_annotation'] != 'Normal':
            abnrmal+=1

print(f'音频总数量为{num}')
print(f'其中正常音数量为{num-abnrmal}')
print(f'其中异常音数量为{abnrmal}')
