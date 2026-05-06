import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import init_network_weights, split_last_dim, check_mask, linspace_vector, reverse


# Params config: []


# ========================================
# ========= ENCODERS MultiNODEs ==========
# ========================================

class LipSwish(torch.nn.Module):
    """
    LipSwish activation to control Lipschitz constant of MLP output
    """
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class VaderLayer(nn.Module):
    def __init__(self, A_init):
        super(VaderLayer, self).__init__()

        self.b = nn.Parameter(A_init)

    def forward(self, x, mask):
        # x is the data and mask is the indicator function
        # Handle missing values section of the main text
        return (1 - mask) * self.b + x * mask
	

class RecognitionRNN(nn.Module):
    # When is used as encoder obs_dim is the number of longitudinal variables and latent_dim the Z dim
    # in both cases nhidden is the hidden state´s dimension between input and output

    def __init__(self, latent_dim, obs_dim, nhidden, act):
        super(RecognitionRNN, self).__init__()

        if nhidden == 0:
            nhidden = 1
        self.nhidden = nhidden
        self.latent_dim = latent_dim

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim)

        if act != "LipSwish":
            self.act = getattr(torch.nn, act)()
        else:
            self.act = LipSwish()

    def internal_forward(self, x, h):

        combined = torch.cat((x, h), dim=1)
        h = self.act(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def forward(self, data):
        h = torch.zeros(data.size(0), self.nhidden).to(data.device)

        # data here is XW it means the longitudinal data
        # for reconRP data can be smaller but in the train script
        # we will change this dimension
        for t in reversed(range(data.size(1))):
            long_v = data[:, t, :]
            out, h = self.internal_forward(long_v, h)
            # out here is Z
        return out


class LSTMEncoder(nn.Module):
    def __init__(self, i_size, h_size, target_size):
        super(LSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(i_size, h_size, num_layers=1, bias=True,
                            batch_first=True, dropout=0, bidirectional=False)
        self.lin = nn.Linear(h_size, target_size)
        self.h_size = h_size

    def forward(self, XW):

        h = torch.zeros(1, XW.size(0), self.h_size).to(XW.device)
        c = torch.zeros(1, XW.size(0), self.h_size).to(XW.device)

        for t in reversed(range(XW.size(1))):
            long_v = XW[:, t:t+1, :]  # longitudinal variable in time t
            out, (h, c) = self.lstm(long_v, (h, c))
            # out and h_0 are the same, because just one point is going through LSTM

        out = out[:, 0, :]
        out = self.lin(out)  # out is pred_z_long
        
        # out = torch.tanh(out) # Keeps z0 between -1 and 1
        return out
	
