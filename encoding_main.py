import pandas, pickle, os
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from torch import Tensor
from torchvision.models import feature_extraction
from torch.utils.data import DataLoader

import encoding_utils as eu
import visualisation_utils as visu

#-----------env args------------------------------------------------------------
dataset = 'friends'
sub = 'sub-06'
no_init = False
tr=1.49

#absolute paths
model_path = '/home/maellef/Results/best_models/converted' 
training_data_path = '/home/maellef/git/MuteMusic_analysis/data/training_data'

#specific to soundnet/audio
model_type = 'conv4'
resolution = 'MIST_ROI'# 'auditory_Voxels' 
sr=22050

#specific to one instance
category = 's04'
r2_max_threshold = 0.4
#repetition = 'all' for life / figures in movie10
#------------load training data-------------------------------------------------------

#load data + metadata
metadata_path = os.path.join(training_data_path, f'{dataset}_{sub}_metadata.tsv')
pairbold_path = os.path.join(training_data_path, f'{dataset}_{sub}_pairWavBold')

data_df = pandas.read_csv(metadata_path, sep='\t')
with open(pairbold_path, 'rb') as f: 
    wavbold = pickle.load(f)

#data selection
i_selection = eu.select_df_index(data_df, category=category)
selected_wavbold = [(wav, bold) for i, (wav, bold) in enumerate(wavbold) if i in i_selection]

#----------load model + convert to extract embedding-------------------------------------

#load model (specific to soundnet model)
print(sub, resolution, model_type, category)
model_name, model = eu.load_sub_model(sub, resolution, model_type, model_path, no_init=False)
print(model_name)
i = model_name.find('conv_') + len('conv_')
temporal_size = int(model_name[i:i+3]) 

#create model with extractable embeddings

#train_nodes, eval_nodes = feature_extraction.get_graph_node_names(model)
#return_nodes = {layer:layer[len('soundnet.'):-2] for layer in train_nodes if layer[-1] == '2'}
return_nodes = {'soundnet.conv7.2':'conv7', 'encoding_fmri':'encoding_conv'}
model_feat = feature_extraction.create_feature_extractor(model, return_nodes=return_nodes)

#----------create dataset and train/val/test fractions (WIP)------------------------------

#specific class for model:
class soundnet_dataset(eu.audio_encoding_dataset):
    def __init__(self, data, tr, sr):
        super().__init__(data, tr, sr)
        #self.layer = layer

    def convert_input_to_tensor(self):
        X_converted = [Tensor(x).view(1,-1,1) for x in self.X_data]
        self.X_data = X_converted
    
    def __temporal_conversion__(self, y, nb_tr, cut='end'):
        #depend on the output layer
        #for layer conv7
        if cut == 'start':
            y = y[len(y)-nb_tr:,:]
        elif cut == 'end':
            y = y[:nb_tr,:]
        return(y)

    def redimension_output(self, Y_pred, Y_real, cut='end'):
        Y_pred_converted = Y_pred.permute(2,1,0,3).squeeze(axis=(2,3)).numpy()
        Y_real_converted = Y_real.squeeze(axis=0).numpy() 
        if len(Y_pred_converted) > len(Y_real_converted):
            #print('redimension prediction outputs to real outputs')
            Y_pred_converted = self.__temporal_conversion__(Y_pred_converted, nb_tr=len(Y_real_converted), cut=cut)
        
        elif len(Y_pred_converted) < len(Y_real_converted):
            #print('redimension real outputs to predicted outputs')
            Y_real_converted = self.__temporal_conversion__(Y_real_converted, nb_tr=len(Y_pred_converted), cut=cut)

        return(Y_pred_converted, Y_real_converted)

#create dataset
encoding_dataset = soundnet_dataset(selected_wavbold, tr=tr, sr=sr)
encoding_dataset.same_size_x_y(cut='end')
encoding_dataset.segment_audio_dataset(segment_size=temporal_size)
encoding_dataset.convert_input_to_tensor()

#-------------------encoding or training---------------------------------------

#for training : criterion=nn.MSELoss(reduction='sum')
testloader = DataLoader(encoding_dataset)
out_p = eu.test(testloader, net=model_feat, return_nodes=return_nodes, gpu=False)

#convert output to numpy array and check if yp and y have the same length

Y_pred_converted, Y_real_converted = [], []
for y_p, y_r in out_p['encoding_conv']:
    (y_p_converted, y_r_converted) = encoding_dataset.redimension_output(y_p, y_r, cut='end')
    Y_pred_converted.append(y_p_converted)
    Y_real_converted.append(y_r_converted)

#---------------visualize----------------------------------------------------

predicted_y = np.vstack(Y_pred_converted)
real_y = np.vstack(Y_real_converted)
print(predicted_y.shape, real_y.shape)

r2 = r2_score(real_y, predicted_y, multioutput='raw_values')
r2 = np.where(r2<0, 0, r2)
print(max(r2))
colormap = visu.extend_colormap(original_colormap='turbo',
                          percent_start = 0.1, percent_finish=0)
visu.surface_fig(r2, vmax=r2_max_threshold, threshold=0.00005, cmap='turbo', symmetric_cbar=False)

savepath = f'./figures/{sub}_generalisation_{dataset}_{category}.png'
plt.savefig(savepath)

#-----------------a voir avec mutemusic------------------------------------------------------------- 
#extract X and Y data for prediction + check for empty data (WIP: move to previous later)
#empty_pair = []
#for i, (wav, bold) in enumerate(selected_wavbold):
#    if wav.shape[0] == 0 and bold.shape[0] == 0:
#        empty_pair.append(i)

#correct_wavbold = [(wav, bold) for (wav, bold) in selected_wavbold if wav.shape[0] != 0]
#print(empty_pair)

#correct_data_df = data_df.drop(empty_pair).reset_index()
#correct_data_df.drop(['index', 'Unnamed: 0'], axis='columns', inplace=True)