import os, pickle
import numpy as np
import pandas as pd
import librosa

#args
scale = 'MISTROI'
sub = 'sub-04'
dataset = 'movie10'

#path
tracks_path = f'/home/maellef/DataBase/stimuli/{dataset}/'
data_path = f'/home/maellef/DataBase/{dataset}/Preprocessed_fMRI'

#
sessions_data = 'sub_session_data.tsv'
sessions_df = pd.read_csv(sessions_data, sep='\t')
silences_data = 'silences_data.tsv'
silences_df = pd.read_csv(silences_data, sep='\t')

path = os.path.join(data_path, scale, sub, filename)
with np.load(path) as data:
    x = data['X']
data.close()
#----------------------------------------------------------------
def find_label_value(name, label):
    label += '-' if label[-1] != '-' else label
    startval = name.find(label)+len(label)
    endval = startval
    for i in range(startval, len(name)):
        if name[i] in ['_', '.']:
            break
        endval+=1
    return name[startval:endval]
#--------------------------------------------------------------------
