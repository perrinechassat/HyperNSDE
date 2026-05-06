
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, List, Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import sys
sys.modules['pykeops'] = None
from external.s4.models.s4.s4 import S4Block as S4D
from src.evaluation.utils import fill_missing_values_long, build_future_targets, build_input
import torch.jit as jit

"""
-----------------------------------------------------------------------------
                                Utility metrics
-----------------------------------------------------------------------------
"""

# -----------------------------------------------------------------------------
# Predictive Score (to do with TSTR/TRTR)
# ----------------------------------------------------------------------------- 

def prediction_score_TSTR(
    x_real, m_real, T_real, 
    x_syn, m_syn, T_syn, 
    W_real = None, W_syn = None, 
    # future_steps = 5, 
    input_len = None,
    cached_real_data = None, # (x_test_real, m_test_real, y_test_real) pre-processed
    predictor_type='S4', 
    epochs=200, 
    batch_size=32, 
    device='cuda',
    **kwargs
):
    # Unpack Cached Real Data (Ground Truth)
    # Ensure these are already on GPU to avoid transfers
    if cached_real_data is not None:
        x_test_real, m_test_real, y_test_real, ym_test_real = [t.clone() for t in cached_real_data]
    else: 
        xr_f, mr_f, _ = fill_missing_values_long(x_real, m_real, T_real, filling_type='last')
        if W_real is not None:
            Wr_rep = W_real.unsqueeze(1).expand(-1, xr_f.size(1), -1)
            xr_aug = torch.cat([xr_f, Wr_rep], dim=-1)
        else:
            xr_aug = xr_f

        if input_len is None:
            input_len = int(0.5*xr_aug.size(1))
            # input_len = xr_aug.size(1) - future_steps

        x_test_real = xr_aug[:, :input_len, :].to(device)
        m_test_real = mr_f[:, :input_len, :].to(device)
        y_test_real = xr_f[:, input_len:, :].to(device)
        ym_test_real = mr_f[:, input_len:, :].to(device)

    # Prepare Synthetic Data (Train Set)
    x_syn_f, m_syn_f, _ = fill_missing_values_long(x_syn, m_syn, T_syn, filling_type='last')
    
    # Augment with static vars
    if W_syn is not None:
        W_rep = W_syn.unsqueeze(1).expand(-1, x_syn_f.size(1), -1)
        x_syn_aug = torch.cat([x_syn_f, W_rep], dim=-1)
    else:
        x_syn_aug = x_syn_f

    # Split Past (Input) / Future (Target)
    # Input: All steps except last 'future_steps'
    # Target: Last 'future_steps' of the longitudinal features ONLY
    if input_len is None:
        # input_len = x_syn_aug.size(1) - future_steps
        input_len = int(0.5*x_syn_aug.size(1))
    x_train = x_syn_aug[:, :input_len, :]
    m_train = m_syn_f[:, :input_len, :]


    n_pts_predict = x_syn_aug.size(1) - input_len
    y_train = x_syn_f[:, input_len:input_len + n_pts_predict, :] # Predict only dynamic features
    ym_train = m_syn_f[:, input_len:input_len + n_pts_predict, :]
    y_test_real = y_test_real[:, :n_pts_predict, :].to(device)
    ym_test_real = ym_test_real[:, :n_pts_predict, :].to(device)
    
    # Shuffle Train Data
    N_train = x_train.size(0)
    perm = torch.randperm(N_train, device=device)
    x_train, m_train, y_train, ym_train = x_train[perm], m_train[perm], y_train[perm], ym_train[perm]

    # Model Setup
    input_dim = x_train.size(2)
    output_dim = y_train.size(2)
    
    if predictor_type == 'S4':
        model = S4Predictor(input_dim, n_pts_predict, d_model=64, n_layers=1, forecast_dim=output_dim, dropout=0.2).to(device)
    elif predictor_type == 'LSTM':
        model = LSTMPredictor(input_dim, n_pts_predict, d_model=32, n_layers=1, forecast_dim=output_dim).to(device)
    else:
        raise ValueError(f"Unknown predictor: {predictor_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    model.train()
    for epoch in range(epochs):
        # Manual batching (slice instead of DataLoader)
        for i in range(0, N_train, batch_size):
            xb = x_train[i:i+batch_size]
            mb = m_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            ymb = ym_train[i:i+batch_size]
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(xb, mb)
                n_obs = ymb.sum().clamp(min=1)
                loss  = ((preds - yb) ** 2 * ymb).sum() / n_obs
                # loss = criterion(preds, yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Evaluation on Real Test Set
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        # Process test set in batches to avoid OOM if real set is huge
        total_mse     = 0.0
        total_obs_pts = 0  
        
        N_test = x_test_real.size(0)
        for i in range(0, N_test, batch_size):
            xb = x_test_real[i:i+batch_size]
            mb = m_test_real[i:i+batch_size]
            yb = y_test_real[i:i+batch_size]
            ymb = ym_test_real[i:i+batch_size]
            
            preds = model(xb, mb)
            n_obs = ymb.sum().clamp(min=1)
            loss  = ((preds - yb) ** 2 * ymb).sum() / n_obs
            total_mse += loss.item() * ymb.sum().item()
            total_obs_pts += ymb.sum().item()

    final_mse = total_mse / total_obs_pts
    return {"mse_test": final_mse}


###############################################################################
# Predictor for Time-Series Forecasting
###############################################################################


class S4Predictor(nn.Module):
    def __init__(self, d_input, m, d_model=64, n_layers=2, dropout=0.0, forecast_dim=None):
        super().__init__()
        forecast_dim = forecast_dim or d_input
        self.m = m
        self.forecast_dim = forecast_dim

        self.encoder = nn.Linear(d_input, d_model)
        
        # S4 Layers (Standard S4D config)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True))
            self.norms.append(nn.LayerNorm(d_model))
        self.decoder = nn.Linear(d_model, m * forecast_dim)

    def forward(self, x, mask):
        # x: (B, T, D)
        x = self.encoder(x)
        x = x.transpose(-1, -2) # (B, D, T)
        
        for layer, norm in zip(self.s4_layers, self.norms):
            z = norm(x.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            x = x + z
            
        x = x.transpose(-1, -2) # (B, T, D)
        lengths = mask.sum(dim=1)[:, 0].long() # (B,)
        last_states = x[torch.arange(x.size(0), device=x.device), lengths - 1]
        
        # Decode
        out = self.decoder(last_states)
        return out.view(x.size(0), self.m, self.forecast_dim)


class LSTMPredictor(nn.Module):
    """
    Direct LSTM Predictor (Non-autoregressive).
    Significantly faster training for TSTR metrics.
    """
    def __init__(self, d_input, m, d_model=64, n_layers=1, dropout=0.0, forecast_dim=None):
        super().__init__()
        forecast_dim = forecast_dim or d_input
        self.m = m
        self.forecast_dim = forecast_dim
        
        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(d_model, m * forecast_dim)

    def forward(self, x, mask=None):
        out, (h, c) = self.lstm(x)
        # h: (num_layers, B, D) -> (B, D)
        last_hidden = h[-1]
        out = self.fc(last_hidden)
        return out.view(x.size(0), self.m, self.forecast_dim)
    
