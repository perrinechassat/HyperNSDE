import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from src.utils import init_network_weights, split_last_dim, check_mask, linspace_vector, reverse

# Params config: [act_dec, n_long_var, drop_dec]

'linear', 'nonlinear'

# ========================================
# ========= DECODERS MultiNODEs ==========
# ========================================


class RecognitionRNN(nn.Module):
    # obs_dim is the number of longitudinal variables and latent_dim the z dim, 
    # nhidden is the hidden state´s dimension between input and output>

    def __init__(self, obs_dim, latent_dim, nhidden, act):
        super(RecognitionRNN, self).__init__()

        if nhidden == 0:
            nhidden = 1
        self.nhidden = nhidden
        self.obs_dim = obs_dim

        self.i2h = nn.Linear(latent_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, obs_dim)

        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:  # act == 'none'
            self.act = nn.Identity()

    def internal_forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = self.act(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def forward(self, data, time_dim=10):
        # When used as decoder, time_dim should be ajusted according to the desired time dimension
        h = torch.zeros(data.size(0), self.nhidden).to(data.device)
        out = torch.zeros(data.size(0), time_dim, self.obs_dim)
        for t in range(data.size(1)):
            z = data[:, t, :]
            out[:, t, :], h = self.internal_forward(z, h)
        return out
    

class LSTMDecoder(nn.Module):
    def __init__(self, i_size, h_size, target_size):
        super(LSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(i_size, h_size, num_layers=1, bias=True,
                            batch_first=True, dropout=0, bidirectional=False)
        self.lin = nn.Linear(h_size, target_size)

    def forward(self, z):
        out, _ = self.lstm(z)
        out = self.lin(out)  # out is pred_x
        return out
        

class Decoder_MultiNODEs(nn.Module):
    def __init__(self, config, latent_dim, nhidden_number):
        super(Decoder_MultiNODEs, self).__init__()

        self.act = getattr(torch.nn, config.act_dec)()

        self.act_dec = config.act_dec
        self.fc1 = nn.Linear(latent_dim, nhidden_number)
        self.fc2 = nn.Linear(nhidden_number, config.n_long_var)
        self.drop = nn.Dropout(config.drop_dec)

    def forward(self, z):
        out = self.fc1(z)

        # It's convenient to use dropout after activation, but 
        # in case of Relu before activation
        if self.act_dec == 'ReLU':
            out = self.act(self.drop(out))
        else:
            out = self.drop(self.act(out))

        out = self.fc2(out)  # out is pred_x
        return out
    
    
# class Decoder(nn.Module):
#     def __init__(self, config, latent_dim):
#         super(Decoder, self).__init__()

#         decoder = nn.Sequential(nn.Linear(latent_dim, config.n_long_var), )
#         init_network_weights(decoder)
#         self.decoder = decoder

#     def forward(self, z, time_dim=None):
#         return self.decoder(z)
        

class Decoder(nn.Module):
    def __init__(self, config, latent_dim):
        super(Decoder, self).__init__()

        if config.type_dec == 'linear':
            self.decoder = nn.Sequential(nn.Linear(latent_dim, config.n_long_var, bias=False), )
        else:  # nonlinear
            activation = getattr(torch.nn, config.act_dec)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, config.nhidden_dec, bias=False),
                activation(),
                nn.Dropout(config.drop_dec), # drop_dec = 0.0 for no dropout
                nn.Linear(config.nhidden_dec, config.n_long_var, bias=False)
            )
   
        # init_network_weights(self.decoder)

    def forward(self, z):
        return self.decoder(z)
    
    # def forward(self, z, time_dim=None):
    #     return self.decoder(z)



# ========================================
# ========= DECODERS with statics ========
# ========================================


class Decoder_w_Static(nn.Module):
    def __init__(self, config, latent_dim, nhidden_number):
        super(Decoder_w_Static, self).__init__()

        if config.act_dec == 'tanh':
            self.act = nn.Tanh()
        elif config.act_dec == 'relu':
            self.act = nn.ReLU()
        else:  
            self.act = nn.Identity()

        self.act_dec = config.act_dec

        self.fc1 = nn.Linear(latent_dim, nhidden_number)
        self.fc2 = nn.Linear(nhidden_number, config.n_long_var)
        self.drop = nn.Dropout(config.drop_dec)

        # self.type_sig = config.dec_sig
        # if self.type_sig == 'continue':
        #     # Variance network
        #     self.fc1_sigma = nn.Linear(latent_dim, nhidden_number)
        #     self.fc2_sigma = nn.Linear(nhidden_number, config.n_long_var)
        # else:
        #     # Variance parameters
        #     self.out_logsig = torch.nn.Parameter(torch.ones(config.n_long_var)*(-5.0))
        
        # self.sp = nn.Softplus()

    
    def forward(self, z, sample=False):
        
        # Mean 
        out = self.fc1(z)
        if self.act_dec == 'relu':
            out = self.act(self.drop(out))
        else:
            out = self.drop(self.act(out))
        out = self.fc2(out)  

        return out 

        # # Variance 
        # if self.type_sig == 'continue':
        #     out_logsig = self.fc1_sigma(z)
        #     if self.act_dec == 'relu':
        #         out_logsig = self.act(self.drop(out_logsig))
        #     else:
        #         out_logsig = self.drop(self.act(out_logsig))
        #     out_logsig = self.fc2_sigma(out_logsig)
        #     # sigma = self.sp(out_sigma)
        #     std = torch.exp(0.5*out_logsig)
        #     noise = torch.randn_like(out) * std
        #     if sample:
        #         out = out + noise
            
        #     return out, std
        
        # else: 
        #     if sample:
        #         # Add Gaussian noise based on the learned variance
        #         # std = self.sp(self.out_logsig).sqrt()  # Convert log-sigma to std dev
        #         std = torch.exp(0.5 * self.out_logsig)
        #         std = std.view(1, 1, -1).expand_as(out)
        #         out = out + torch.randn_like(out) * std
            
        #     return out
        

