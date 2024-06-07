import os, torch, librosa, pandas
import numpy as np
from torch import nn
from torch.utils.data import Dataset

import sys
sys.path.append('/home/maellef/git/cNeuromod_encoding_2020')
from models import encoding_models as encod

MIST_path = '/home/maellef/DataBase/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

#specific aux soundnet models
def load_sub_model(sub, scale, conv, models_path, no_init=False):
    for model_name in os.listdir(models_path):
        conv_bool = ('_f_' not in model_name) if conv == 'soundnet' else (conv in model_name)
        if '.pt' in model_name and conv_bool and sub in model_name and scale in model_name:
            model_path = os.path.join(models_path, model_name)
            modeldict = torch.load(model_path, map_location=torch.device('cpu'))
            model_net = encod.SoundNetEncoding_conv(out_size=modeldict['out_size'],output_layer=modeldict['output_layer'],
                                                    kernel_size=modeldict['kernel_size'], no_init=no_init)
            if not no_init:
                model_net.load_state_dict(modeldict['checkpoint'])
            return (model_name, model_net)

#individual class X_dataset(Dataset):

class encoding_dataset(Dataset):
    def __init__(self, data, tr):
        self.tr = tr
        self.X_data = [x for (x, y) in data]
        self.Y_data = [y for (x, y) in data]
    
    def __len__(self):
        return len(self.X_data)
    
    #def create_batches(self)

    def __getitem__(self, idx):
        return self.X_data[idx], self.Y_data[idx]
    
class audio_encoding_dataset(encoding_dataset):
    def __init__(self, data, tr, sr):
        super().__init__(data, tr)
        self.sr = sr
        self.tr_conversion = round(self.tr*self.sr)
    
    def resample_input(self, input_sr):
        resampled_X = [librosa.resample(x, orig_sr=input_sr, target_sr=self.sr) for x in self.X_data]
        self.X_data = resampled_X
    
    def same_size_x_y(self, cut='end'):
        for i, (x, y) in enumerate(zip(self.X_data, self.Y_data)):
            x_size = round(len(x)/self.tr_conversion)
            y_size = len(y)
            if x_size != y_size:
                min_tr = min(x_size, y_size)
                if cut == 'end':
                    self.X_data[i] = x[:min_tr*self.tr_conversion]
                    self.Y_data[i] = y[:min_tr]
                elif cut == 'start':
                    self.X_data[i] = x[(x_size-min_tr)*self.tr_conversion:]
                    self.Y_data[i] = y[y_size-min_tr:]
    
    def segment_audio_dataset(self, segment_size):
        X_segmented = []
        Y_segmented = []
        for x, y in zip(self.X_data, self.Y_data):
            x_chunk_length = self.tr_conversion*segment_size
            y_chunk_length = segment_size
            #WIP:works with x with 1D (audio mono) and y with 2D (BOLD)
            x_starts = range(0, len(x), x_chunk_length)
            y_starts = range(0, len(y), y_chunk_length)

            for x_start, y_start in zip(x_starts, y_starts):
                if x_start+x_chunk_length < len(x):            
                    x_chunk = x[x_start:x_start+x_chunk_length]
                    y_chunk = y[y_start:y_start+y_chunk_length,:]
                else:
                    x_chunk = x[x_start:]
                    y_chunk = y[y_start:,:]

                X_segmented.append(x_chunk)
                Y_segmented.append(y_chunk)

        self.X_data = X_segmented
        self.Y_data = Y_segmented

#-----------------------training-utils-----------------------------------
def test(dataloader, net, return_nodes, epoch=1, gpu=True):
    net.eval()
    out_p = {layer_name:[] for layer_name in return_nodes.values()}
    
    with torch.no_grad():
        for (x,y) in dataloader:
            y_p = net(x, epoch)
            for key, p in y_p.items():
                out_p[key].append((p, y))
    return out_p

#-------------other-utils-------------------------------------------
def select_df_index(df, **selectors):
    '''return the rows indexes of a dataframe based on selectors
    selector argument : column name = value to select
    value can be a list of values, or a single value
    if value = 'all', all rows for this column will be selected'''
    
    conditions = True
    for column_name, val in selectors.items():
        val = val if isinstance(val, (list, np.ndarray)) else [val]        
        selected_items = df[column_name].unique() if val[0]=='all' else val
        condition = df[column_name].isin(selected_items)
        conditions &= (condition)
    i_selection = df.loc[conditions].index.values
    return i_selection

def extract_selected_data(data, data_df, **selectors):
    selection_i = select_df_index(data_df, **selectors)
    selected_data = [(x, y) for i, (x, y) in enumerate(data) if i in selection_i]
    selected_df = data_df.iloc[selection_i]
    return (selected_data, selected_df)

def extract_silence_timestamps(track_df, tr, unit='second'):
    tracks_silence = []
    sr = track_df['sr']
    
    timestamps = {'duration':[track_df['S1_duration'], track_df['S2_duration'], track_df['S3_duration'], track_df['S4_duration']],
                    'start':[track_df['S1_start'], track_df['S2_start'], track_df['S3_start'], track_df['S4_start']],
                    'stop':[track_df['S1_stop'], track_df['S2_stop'], track_df['S3_stop'], track_df['S4_stop']]}
    siltt_df = pandas.DataFrame(timestamps).sort_values(by='start').dropna()

    music = 0
    for i, silence_tt in siltt_df.iterrows():
        if unit == 'second':
            start = silence_tt['start']
            stop = silence_tt['stop']
            dur = silence_tt['duration']
        elif unit == 'sr':
            start = silence_tt['start']*sr
            stop = silence_tt['stop']*sr
            dur = silence_tt['duration']*sr
        elif unit == 'tr':
            start = silence_tt['start']/tr
            stop = silence_tt['stop']/tr
            dur = silence_tt['duration']/tr
        
        silence_start = start if unit == 'second' else round(start)
        silence_stop = stop if unit == 'second' else round(stop)
        silence_duration = dur if unit == 'second' else round(dur)
        tracks_silence.append((silence_start, silence_stop, silence_duration))
            
    return tracks_silence

