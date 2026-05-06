#!/usr/bin/ipython
import os
import warnings
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import sys
sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../../')
from src.parser import base_parser
from src.utils import define_logs, load_yaml_config
from src.generative_model import Generative_Model_Longi_Static
from data_loader.load_data import load_dataset, get_data, load_dataset_CV
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
import copy


def run(MC_id, base_config):

    # ----------------------------------------------------------- #
    # -------------------------- Config ------------------------- #
    # ----------------------------------------------------------- #

    config = copy.deepcopy(base_config)
    config.seed = MC_id  # Different seed for each Monte Carlo run        
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.file_dataset = config.file_dataset + "/simulated_dataset_MC_" + str(MC_id)  # Different dataset for each Monte Carlo run

    folder_name = config.exp_name + "/MC_" + str(MC_id)
    config.save_path = os.path.join(config.init_path + config.save_dir, folder_name)
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path, 'losses')
    config.train_dir = os.path.join(config.init_path + config.train_dir, config.dataset)

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)

    config.save_path_losses = os.path.join(config.save_path_losses, 'losses.txt')

    # ----------------------------------------------------------- #
    # --------------------- Train / Generate -------------------- #
    # ----------------------------------------------------------- #
    
    print('torch.cuda is available: ', torch.cuda.is_available())

    print("\n" + "="*50)
    print(f"--- Loading data MC {MC_id}...")
    config_and_data = get_data(config)
    n_samples = config_and_data[1].shape[0]
    idx_train, idx_val = train_test_split(torch.arange(n_samples), test_size=0.2, random_state=config.seed, shuffle=True)
    if "Ntrain200" in config.exp_name:
        # I keep only 200 samples for the training set
        idx_train, idx_out = train_test_split(idx_train, test_size=0.75, random_state=config.seed, shuffle=True) 
    config, train_dataloader, val_dataloader = load_dataset_CV(config_and_data, idx_train, idx_val) 
    static_types = train_dataloader.dataset.get_static_types()
    define_logs(config)

    print("\n" + "="*50)
    print(f"--- Building model MC {MC_id}...")
    if config.type_enc != 'none':
        init_x, init_mask = train_dataloader.dataset.get_x_mask()
    else: 
        init_x, init_mask = None, None
    model = Generative_Model_Longi_Static(config, static_types=static_types, init_x=init_x, init_mask=init_mask)

    print("\n" + "="*50)
    print(f"--- Training model MC {MC_id}...")
    # model.train(train_dataloader, val_dataloader)

    print("\n" + "="*50)
    print(f"--- Generating synthetic data MC {MC_id}...")
    n_gen = 50
    n_generated_samples = n_gen * len(idx_val)
    # Generate new samples from the prior or posterior distribution, or reconstruct the input data
    # The sigma parameters are used to add noise to the latent representations
    # For generation from the prior and reconstruction, sigma_stat and sigma_long are set to 1.0
    # For generation from the posterior, sigma_stat and sigma_long are to set manually depending on the desired variability of the generated samples  
    res_gen_prior = model.generate(val_dataloader, n_generated_samples=n_generated_samples, type_gen='prior', from_drift_only=False, sigma_stat=1.0, sigma_long=1.0, save=True)
    res_gen_posterior = model.generate(val_dataloader, n_generated_samples=n_generated_samples, type_gen='posterior', from_drift_only=False, sigma_stat=1.0, sigma_long=1.0, save=True)
    res_gen_reconstruction = model.generate(val_dataloader, n_generated_samples=None, type_gen='reconstruction', from_drift_only=False, sigma_stat=1.0, sigma_long=1.0, save=True)
    res_gen_reconstruction_drift = model.generate(val_dataloader, n_generated_samples=None, type_gen='reconstruction', from_drift_only=True, sigma_stat=1.0, sigma_long=1.0, save=True)
    print(f"--- MC {MC_id} completed.")
    print("\n" + "="*50)


if __name__ == "__main__":

    base_config, unknown = base_parser(return_unknown=True)
    external_config = load_yaml_config(base_config.config_file)
    for key, value in external_config.items():
        setattr(base_config, key, value)

    if base_config.GPU == True:
        base_config.GPU = torch.cuda.is_available()
    else:
        base_config.GPU = False

    if base_config.estim_event_rate:
        if base_config.lambda_dim != 1:
            base_config.lambda_dim = base_config.n_long_var

    MC_id = int(unknown[0])
    run(MC_id, base_config)
