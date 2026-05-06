#!/usr/bin/ipython
import os
import warnings
import numpy as np
import time
import torch
# import matplotlib.pyplot as plt
# import yaml
# from datetime import datetime
from src.parser import base_parser
from src.utils import define_logs, load_yaml_config
from src.generative_model import Generative_Model_Longi_Static
import sys
sys.path.append('../')
sys.path.append('../../')
from data_loader.load_data import load_dataset, get_data, load_dataset_CV
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':

    ''' ----------------------------------------------------------- '''
    ''' -------------------------- Config ------------------------- '''
    ''' ----------------------------------------------------------- '''

    config = base_parser()
    external_config = load_yaml_config(config.config_file)
    for key, value in external_config.items():
        setattr(config, key, value)

    if config.GPU == True:
        config.GPU = torch.cuda.is_available()
    else:
        config.GPU = False

    if config.estim_event_rate:
        if config.lambda_dim != 1:
            config.lambda_dim = config.n_long_var

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # folder_name = config.file_dataset[:-4] if '.csv' in config.file_dataset else config.file_dataset
    folder_name = config.exp_name
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

    ''' ----------------------------------------------------------- '''
    ''' ---------------------- Train/Generate --------------------- '''
    ''' ----------------------------------------------------------- '''
    
    print('torch.cuda is available: ', torch.cuda.is_available())

    print("\n" + "="*50)
    print("--- Loading data...")
    # config, train_dataloader, val_dataloader, test_dataloader = load_dataset(config)
    # static_types = train_dataloader.dataset.get_static_types()
    config_and_data = get_data(config, normalize_long=False)
    n_samples = config_and_data[1].shape[0]
    idx_train, idx_val = train_test_split(torch.arange(n_samples), test_size=0.2, random_state=config.seed, shuffle=True)
    config, train_dataloader, val_dataloader = load_dataset_CV(config_and_data, idx_train, idx_val) # normalization is done here to avoid data leakage from the test set.
    static_types = train_dataloader.dataset.get_static_types()
    print(f"    Static variable types: {static_types}")
    define_logs(config)

    print("\n" + "="*50)
    print("--- Building model...")
    model = Generative_Model_Longi_Static(config, static_types=static_types)

    print("\n" + "="*50)
    print("--- Training model...")
    # model.train(train_dataloader, val_dataloader)

    print("\n" + "="*50)
    print("--- Generating synthetic data...")
    n_generated_samples = 50 * len(idx_val)
    # Generate new samples from the prior or posterior distribution, or reconstruct the input data
    # The sigma parameters are used to add noise to the latent representations
    # For generation from the prior and reconstruction, sigma_stat and sigma_long are set to 1.0
    # For generation from the posterior, sigma_stat and sigma_long are to set manually depending on the desired variability of the generated samples  
    # res_gen_prior = model.generate(val_dataloader, n_generated_samples=n_generated_samples, type_gen='prior', from_drift_only=False, sigma_stat=1.0, sigma_long=1.0, save=True)
    # del res_gen_prior
    # torch.cuda.empty_cache()
    # res_gen_posterior = model.generate(val_dataloader, n_generated_samples=n_generated_samples, type_gen='posterior', from_drift_only=False, sigma_stat=1.0, sigma_long=1.0, save=True)
    # del res_gen_posterior
    # torch.cuda.empty_cache()
    # res_gen_reconstruction = model.generate(val_dataloader, n_generated_samples=None, type_gen='reconstruction', from_drift_only=False, sigma_stat=1.0, sigma_long=1.0, save=True)
    # del res_gen_reconstruction
    # torch.cuda.empty_cache()
    # res_gen_reconstruction_drift = model.generate(val_dataloader, n_generated_samples=None, type_gen='reconstruction', from_drift_only=True, sigma_stat=1.0, sigma_long=1.0, save=True)
    # del res_gen_reconstruction_drift
    # torch.cuda.empty_cache()
    res_gen_posterior = model.generate(val_dataloader, n_generated_samples=n_generated_samples, type_gen='posterior', from_drift_only=True, sigma_stat=1.0, sigma_long=1.0, save=True)
    del res_gen_posterior
    torch.cuda.empty_cache()

    print("\n" + "="*50)
    print(f"✨ Generation completed.")
    
    
    # print("\n" + "="*50)
    # print("--- Generating synthetic data...")

    # n_val = len(idx_val)
    # n_chunks = 50

    # for gen_type, drift_only in [('posterior', True)]:
    #     print(f"\n  Generating {gen_type} ({n_chunks} chunks of {n_val} samples)...")

    #     all_X, all_M, all_W, all_Z, all_R = [], [], [], [], []
    #     T_ref, var_names, var_names_static = None, None, None

    #     for chunk_id in range(38, n_chunks):
    #         print(f"    Chunk {chunk_id+1}/{n_chunks}", end="\r")

    #         name_save = f"{chunk_id}_"+gen_type 
    #         res = model.generate(
    #             val_dataloader,
    #             n_generated_samples=n_val,
    #             type_gen=gen_type,
    #             from_drift_only=drift_only,
    #             sigma_stat=1.0,
    #             sigma_long=1.0,
    #             save=True,
    #             name_save=name_save
    #         )
    #         del res
    #         torch.cuda.empty_cache()

    # # # Reconstruction — no chunking needed (1:1 with val set)
    # # for drift_only, suffix in [(False, ''), (True, '_drift')]:
    # #     print(f"\n  Generating reconstruction{suffix}...")
    # #     res = model.generate(
    # #         val_dataloader,
    # #         n_generated_samples=None,
    # #         type_gen='reconstruction',
    # #         from_drift_only=drift_only,
    # #         sigma_stat=1.0,
    # #         sigma_long=1.0,
    # #         save=True
    # #     )
    # #     del res
    # #     torch.cuda.empty_cache()

    # print("\n" + "="*50)
    # print("✨ Generation completed.")
