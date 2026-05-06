#!/usr/bin/ipython
import os
import warnings
import numpy as np
import time
import torch
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../../')
from src.utils import load_yaml_config
import copy
from sklearn.model_selection import train_test_split

# MULTINODES_PATH = "/path/to/Documents/MultiNODEs/src"
MULTINODES_PATH = "/path/to/MultiNODEs/src"
# MULTINODES_PATH = "../../../MultiNODEs"
sys.path.append(MULTINODES_PATH)
from models.parser import base_parser
from models.utils import define_logs
from models.train import Train
from models.validation import Validation
from data.load_data import load_dataset, load_data_other, load_dataset_CV

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

# python3 ./benchmark/multiNODEs/main_multiNODEs_bis.py --init_path '/path/to/HyperNSDE' --config_file 'benchmark/multiNODEs/config_paper_package.yml'
# python3 main_multiNODEs_bis.py --init_path '/path/to/HyperNSDE' --config_file 'benchmark/multiNODEs/config_paper_package.yml'


if __name__ == "__main__":

    base_config = base_parser()
    external_config = load_yaml_config(base_config.config_file)
    for key, value in external_config.items():
        setattr(base_config, key, value)

    if base_config.GPU == True:
        # base_config.GPU = torch.cuda.is_available()
        base_config.GPU = [0] if torch.cuda.is_available() else False
    else:
        base_config.GPU = False

    config = copy.deepcopy(base_config) 

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.train_dir = os.path.join(config.init_path + config.train_dir, config.dataset)
    # config.save_path = os.path.join(config.save_path, config.exp_name)
    config.save_path = os.path.join(config.init_path + config.save_path, config.exp_name)
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path, 'losses')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)

    config.save_path_losses = os.path.join(config.save_path_losses, 'losses.txt')

    config.mode = "only_train"
    define_logs(config)

    print("\n" + "="*50)
    print(f"--- Loading data...")
    config_and_data = load_data_other(config)
    config, data = config_and_data[0], config_and_data[1:]
    n_samples = data[0].shape[0]
    idx_train, idx_val = train_test_split(torch.arange(n_samples), test_size=0.2, random_state=config.seed, shuffle=True)
    config, train_dataloader, val_dataloader = load_dataset_CV(config, data, idx_train, idx_val, only_data=False)  

    print("\n" + "="*50)
    print(f"--- Training model...")
    # Train(config, train_dataloader, val_dataloader)

    print("\n" + "="*50)
    print(f"--- Generating synthetic data...")
    config.from_best = False
    Validation(config, val_dataloader, f_train=True)

    print(f"--- Completed.")
    print("\n" + "="*50)
    