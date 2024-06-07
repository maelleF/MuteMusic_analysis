from torch import Tensor
import encoding_utils as eu

class soundnet_dataset(eu.audio_encoding_dataset):
    def __init__(self, data, tr, sr):
        super().__init__(data, tr, sr)
        #self.layer = layer

    def convert_input_to_tensor(self):
        X_converted = [Tensor(x).view(1,-1,1) for x in self.X_data]
        self.X_data = X_converted
    
    def __tr_alignment__(self, y, nb_tr, cut='end'):
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
            Y_pred_converted = self.__tr_alignment__(Y_pred_converted, nb_tr=len(Y_real_converted), cut=cut)
        
        elif len(Y_pred_converted) < len(Y_real_converted):
            #print('redimension real outputs to predicted outputs')
            Y_real_converted = self.__tr_alignment__(Y_real_converted, nb_tr=len(Y_pred_converted), cut=cut)

        return(Y_pred_converted, Y_real_converted)
