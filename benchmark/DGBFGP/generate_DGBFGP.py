import torch
import numpy as np
import os
import argparse
import sys
import copy
import argparse
from sklearn.preprocessing import StandardScaler
sys.path.append('../../')
from data_loader.datasets import Simulation_Dataset_Static
from data_loader.load_data import read_static_data, read_csv_values
from src.modules.static_HIVAE import HIVAE_Decoder, HIVAE_Encoder
from train_standalone_hivae import *
# DGBFGP_PATH = "/path/to/Documents/DGBFGP"
DGBFGP_PATH = "/path/to/DGBFGP"
sys.path.append(DGBFGP_PATH)
from models import DGBFGP
    
def generate_DGBFGP(args, path_data, path_ckpts, idx_train, idx_val, x_scaler, y_scaler, n_gen, T_gen, device, s_dim_static=2, z_dim_static=2, batch_size=32):    
    y_num_dim = args.y_num_dim
    x_num_dim = args.x_num_dim
    id_embed_dim = args.id_embed_dim
    P = args.P
    M = args.M
    C = args.C
    latent_dim = args.latent_dim
    vy_fixed = args.vy_fixed
    vy_init = 1
    p_drop = args.p_drop
    basis_funcs = args.basis_funcs
    scale = args.scale
    alpha = args.alpha
    alpha_fixed = args.alpha_fixed
    scale_fixed = args.scale_fixed
    se_idx = copy.deepcopy(args.se_idx)
    ca_idx = copy.deepcopy(args.ca_idx)
    bin_idx = copy.deepcopy(args.bin_idx)
    interactions = copy.deepcopy(args.interactions)
    assert len(C) == len(interactions), "Number of categorical variables should match number of interactions (SE x CA)"
    id_covariate = args.id_covariate
    id_handler = args.id_handler
    dataset_type = args.dataset_type
    k = 1
    k_test = 1
    B = args.B
    loss_function = args.loss_function
    stochastic_flag = False
    plot_flag = False
    if loss_function == "iwae":
        k = args.k
        k_test = 500
        stochastic_flag = True

    model = DGBFGP(y_num_dim, x_num_dim, latent_dim, P, id_embed_dim, id_handler, M, C, id_covariate, se_idx, ca_idx, bin_idx, interactions, basis_funcs, 
                   scale, alpha, alpha_fixed, scale_fixed, vy_init, vy_fixed, p_drop, dataset_type, k=k, k_test=k_test, device = device).to(device)
    model.load_state_dict(torch.load(os.path.join(path_ckpts, "model_params_simu.pth"), map_location=device))
    model.eval()    

    """
      BLOCK HI-VAE + generation of 50 sets of samples of covariates

    """
    static_vals = read_csv_values(os.path.join(path_data, 'data_static.csv'), to_torch=True)
    static_types = read_csv_values(os.path.join(path_data, 'data_static_types.csv'), header=None)
    static_missing = read_csv_values(os.path.join(path_data, 'data_static_missing.csv'), to_torch=True)
    if static_types.shape[1] > 3:
        static_types = static_types[:, -3:]
    static_vals_dim = static_vals.shape[1]
    static_onehot, static_types, static_true_miss_mask = read_static_data(static_vals, static_types, static_missing)
    static_onehot_dim = static_onehot.shape[1]
    data_splits = {
        "train": [static_onehot[idx_train], static_types, static_true_miss_mask[idx_train], static_vals[idx_train]],
        "validation": [static_onehot[idx_val], static_types, static_true_miss_mask[idx_val], static_vals[idx_val]],
    }
    train_data = data_splits["train"]
    val_data = data_splits["validation"]
    train_dataset = Simulation_Dataset_Static(train_data, batch_norm_static=True)
    validation_dataset = Simulation_Dataset_Static(val_data, batch_norm_static=True)
    batch_size = int(batch_size)
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last = False, num_workers=4, pin_memory=True) 
    val_dataloader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = False, drop_last = False, num_workers=4, pin_memory=True) 
        
    # Initialize the networks
    S_Enc = HIVAE_Encoder(static_vals_dim, static_onehot_dim, s_dim_static, z_dim_static).to(device)
    S_Dec = HIVAE_Decoder(static_vals_dim, static_types, s_dim_static, z_dim_static).to(device)

    # Train the HI-VAE
    S_Enc, S_Dec = train_standalone_hivae(
        S_Enc=S_Enc, 
        S_Dec=S_Dec, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        static_types=static_types, 
        s_dim_static=s_dim_static, 
        device=device,
        epochs=50, 
        lr=2e-3,
        initial_tau=1e-3
    )

    # Generate new synthetic static covariates
    with torch.no_grad():
        static_data = val_dataloader.dataset.get_onehot_static()
        n_per_gen = len(idx_val)
        n_generated_samples = n_gen * n_per_gen
        set_idx = torch.cat([torch.arange(0, n_per_gen, 1)]*n_gen, dim=0)
        idx_gen, _ = torch.sort(torch.randperm(len(set_idx))[:n_generated_samples])
        gen_stat = []
        for i in range(n_gen): 
            _, gen_stat_i, _ = static_forward_pass(S_Enc=S_Enc, S_Dec=S_Dec, data=static_data, tau=1e-3, 
                                                    static_types=static_types, s_dim_static=s_dim_static, 
                                                    device=device, batch_norm_static=True, return_pred_stat=True)
            gen_stat.append(gen_stat_i.detach().cpu())
    gen_stat = torch.cat(gen_stat, dim=0)
    gen_stat = gen_stat[idx_gen]

    gen_stat_reshape = gen_stat.view(n_gen, n_per_gen, static_vals_dim)

    # Formated static
    gen_stat_np = gen_stat.cpu().numpy()
    T_gen_np = T_gen.cpu().numpy()
    base_patient_ids = np.tile(np.arange(n_per_gen), n_gen)
    patient_ids = np.repeat(base_patient_ids, len(T_gen))
    times = np.tile(T_gen_np, n_generated_samples)
    static_repeated = np.repeat(gen_stat_np, len(T_gen), axis=0)
    X_raw = np.column_stack((patient_ids, times, static_repeated))
    X_scaled = X_raw.copy()
    numerical_covariates = [1, 2] 
    X_scaled[:, numerical_covariates] = x_scaler.transform(X_raw[:, numerical_covariates])
    
    X_gen_tensor_cpu = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    patients_per_batch = 10
    rows_per_batch = patients_per_batch * len(T_gen)

    with torch.no_grad():
        y_generated_scaled_list = []
        for i in range(0, X_gen_tensor_cpu.shape[0], rows_per_batch):
            X_batch = X_gen_tensor_cpu[i : i + rows_per_batch].to(device)
            z_list, _, _ = model.encode(X_batch, stochastic_flag=True, train=False)
            z_summed = sum(z_list)
            y_generated_scaled = model.decode(z_summed)
            y_generated_scaled_list.append(y_generated_scaled.squeeze(1).cpu().numpy())

    y_gen_np = np.concatenate(y_generated_scaled_list, axis=0)
    y_transform = y_scaler.inverse_transform(y_gen_np)
    y_final_np = y_transform.reshape(n_gen, n_per_gen, len(T_gen), y_num_dim)
    y_tensor_final = torch.from_numpy(y_final_np)
    
    # X_gen_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    # batch_size = 32

    # # Generate Longitudinal Trajectories
    # with torch.no_grad():
    #     i = 0
    #     y_generated_scaled_list = []
    #     while i < X_gen_tensor.shape[0]:
    #         z_list, _, _ = model.encode(X_gen_tensor[i:i+batch_size], stochastic_flag=True, train=False)
    #         z_summed = sum(z_list)
    #         y_generated_scaled = model.decode(z_summed)
    #         y_generated_scaled_list.append(y_generated_scaled.squeeze(1).cpu().numpy())
    #         i = i + batch_size
    #     if X_gen_tensor.shape[0] - i > 0:
    #         z_list, _, _ = model.encode(X_gen_tensor[i:], stochastic_flag=True, train=False)
    #         z_summed = sum(z_list)
    #         y_generated_scaled = model.decode(z_summed)
    #         y_generated_scaled_list.append(y_generated_scaled.squeeze(1).cpu().numpy())

    #     y_gen_np = np.concatenate(y_generated_scaled_list, axis=0)
    #     y_transform = y_scaler.inverse_transform(y_gen_np)
    #     y_final_np = y_transform.reshape(n_gen, n_per_gen, len(T_gen), y_num_dim)
    #     y_tensor_final = torch.from_numpy(y_final_np)

    return gen_stat_reshape, y_tensor_final