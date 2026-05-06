import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from src.parser import base_parser
from src.utils import load_yaml_config
from src.hyperopt import optuna_hyperparameter_search
import torch
import os
import numpy as np
# from datetime import datetime

if __name__ == '__main__':

    ''' ----------------------------------------------------------- '''
    ''' -------------------------- Config ------------------------- '''
    ''' ----------------------------------------------------------- '''

    config = base_parser()
    external_config = load_yaml_config(config.config_file)
    message = ''
    message += '----------------- Options ---------------\n'
    for key, value in external_config.items():
        setattr(config, key, value)
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
    message += '----------------- End -------------------'
    print(message)
    if config.GPU == True:
        config.GPU = torch.cuda.is_available()
    else:
        config.GPU = False
    if config.estim_event_rate:
        if config.lambda_dim != 1:
            config.lambda_dim = config.n_long_var
    # Seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.train_dir = os.path.join(config.init_path + config.train_dir, config.dataset)
    config.save_path_losses = ''

    ''' ----------------------------------------------------------- '''
    ''' -------------------------- Hyperopt ----------------------- '''
    ''' ----------------------------------------------------------- '''

    if config.dataset == 'Simu_OU/monte_carlo':
        params_dict = {'batch_size': [128, 256], # 128 (suggest_categorical)
                       'lr': [1e-2, 5e-3, 2e-3, 1e-3], # (suggest_categorical)
                       'z_dim_static' : [2, 6], # min, max (suggest_int, 1)
                       's_dim_static' : [2, 6], # min, max (suggest_int, 1)  
                       'latent_dim' : [2, 4], # min, max (suggest_int, 1)
                       'hidden_dim': [16, 32, 64], # 128 (suggest_categorical)
                       'num_odelayers' : [1, 2], # min, max (suggest_int, 1)
                       'num_hypernet_layers' : [1, 3], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-6, 1.0e-2], # min, max (suggest_float log scale)
                     #   'lambda_act' : ['LipSwish', 'Tanh', 'ReLU'], # (suggest_categorical)
                       'diff_mlp_num_layers' : [1, 3], # min, max (suggest_int, 1)
                       'sigma_kernel': [0.1, 1.0], # min, max (suggest_float log scale)
                     #   'act_dec' : ['ReLU', 'Tanh'], #, 'Identity'] (suggest_categorical)
                       'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [20, 800], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.01, 200], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.01, 1], # min, max (suggest_float log scale) 
        }
    elif config.dataset == 'Simu_OU/monte_carlo_N250':
        params_dict = {'batch_size': [32, 64, 128], # 128 (suggest_categorical)
                       'lr': [5e-3, 2e-3, 1e-3, 5e-4], # (suggest_categorical)
                     #   'z_dim_static' : [2, 6], # min, max (suggest_int, 1)
                     #   's_dim_static' : [2, 6], # min, max (suggest_int, 1)  
                     #   'latent_dim' : [2, 4], # min, max (suggest_int, 1)
                       'hidden_dim': [16, 32, 64], # 128 (suggest_categorical)
                       'num_odelayers' : [1, 2], # min, max (suggest_int, 1)
                       'num_hypernet_layers' : [1, 3], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-6, 1.0e-2], # min, max (suggest_float log scale)
                    #    'lambda_act' : ['LipSwish', 'Tanh', 'ReLU'], # (suggest_categorical)
                     #   'diff_mlp_num_layers' : [1, 3], # min, max (suggest_int, 1)
                    #    'sigma_kernel': [0.1, 1.0], # min, max (suggest_float log scale)
                    #    'act_dec' : ['ReLU', 'Tanh'], #, 'Identity'] (suggest_categorical)
                       'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [20, 800], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.01, 10], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.01, 1], # min, max (suggest_float log scale) 
        }
    elif config.dataset == 'Simu_OU/monte_carlo_N5000':
        params_dict = {
        # 'batch_size': [256], # 128 (suggest_categorical)
                       'lr': [1e-2, 5e-3, 2e-3, 1e-3], # (suggest_categorical)
                     #   'z_dim_static' : [2, 6], # min, max (suggest_int, 1)
                     #   's_dim_static' : [2, 6], # min, max (suggest_int, 1)  
                     #   'latent_dim' : [2, 4], # min, max (suggest_int, 1)
                       'hidden_dim': [16, 32, 64, 128], # 128 (suggest_categorical)
                       'num_odelayers' : [1, 2], # min, max (suggest_int, 1)
                       'num_hypernet_layers' : [1, 3], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-6, 1.0e-2], # min, max (suggest_float log scale)
                    #    'lambda_act' : ['LipSwish', 'Tanh', 'ReLU'], # (suggest_categorical)
                     #   'diff_mlp_num_layers' : [1, 3], # min, max (suggest_int, 1)
                    #    'sigma_kernel': [0.1, 1.0], # min, max (suggest_float log scale)
                    #    'act_dec' : ['ReLU', 'Tanh'], #, 'Identity'] (suggest_categorical)
                       'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [20, 800], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.01, 10], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.01, 1], # min, max (suggest_float log scale) 
        }
    elif config.dataset == 'Simu_OU/monte_carlo_15pct':
        params_dict = {'batch_size': [64, 128, 256], # 128 (suggest_categorical)
                       'lr': [1e-2, 5e-3, 2e-3, 1e-3], # (suggest_categorical)
                    #    'z_dim_static' : [2, 6], # min, max (suggest_int, 1)
                    #    's_dim_static' : [2, 6], # min, max (suggest_int, 1)  
                    #    'latent_dim' : [2, 4], # min, max (suggest_int, 1)
                    #    'hidden_dim': [16, 32, 64], # 128 (suggest_categorical)
                    #    'num_odelayers' : [1, 2], # min, max (suggest_int, 1)
                    #    'num_hypernet_layers' : [1, 3], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-6, 1.0e-2], # min, max (suggest_float log scale)
                    #    'lambda_act' : ['LipSwish', 'Tanh', 'ReLU'], # (suggest_categorical)
                    #    'diff_mlp_num_layers' : [1, 3], # min, max (suggest_int, 1)
                       'sigma_kernel': [0.1, 1.0], # min, max (suggest_float log scale)
                    #    'act_dec' : ['ReLU', 'Tanh'], #, 'Identity'] (suggest_categorical)
                    #    'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [20, 800], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.01, 10], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.01, 1], # min, max (suggest_float log scale) 
        }
    elif config.dataset == 'Simu_OU/monte_carlo_100pct':
        params_dict = {'batch_size': [128, 256], # 128 (suggest_categorical)
                       'lr': [1e-2, 5e-3, 2e-3, 1e-3], # (suggest_categorical)
                    #    'z_dim_static' : [2, 6], # min, max (suggest_int, 1)
                    #    's_dim_static' : [2, 6], # min, max (suggest_int, 1)  
                    #    'latent_dim' : [2, 4], # min, max (suggest_int, 1)
                    #    'hidden_dim': [16, 32, 64], # 128 (suggest_categorical)
                    #    'num_odelayers' : [1, 2], # min, max (suggest_int, 1)
                    #    'num_hypernet_layers' : [1, 3], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-6, 1.0e-2], # min, max (suggest_float log scale)
                    #    'lambda_act' : ['LipSwish', 'Tanh', 'ReLU'], # (suggest_categorical)
                    #    'diff_mlp_num_layers' : [1, 3], # min, max (suggest_int, 1)
                       'sigma_kernel': [0.1, 1.0], # min, max (suggest_float log scale)
                    #    'act_dec' : ['ReLU', 'Tanh'], #, 'Identity'] (suggest_categorical)
                    #    'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [20, 800], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.01, 10], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.01, 1], # min, max (suggest_float log scale) 
        }
    # elif config.dataset == 'ADNI':
    #     params_dict = {'batch_size': [128, 256], # 128 (suggest_categorical)
    #                    'lr': [1e-2, 5e-3, 1e-3, 1e-4], # (suggest_categorical)
    #                    'z_dim_static' : [2, 8], # min, max (suggest_int, 1)
    #                    's_dim_static' : [2, 8], # min, max (suggest_int, 1)  
    #                    'latent_dim' : [2, 20], # min, max (suggest_int, 1)
    #                    'hidden_dim': [16, 32, 64, 128], # 128 (suggest_categorical)
    #                    'num_odelayers' : [1, 3], # min, max (suggest_int, 1)
    #                    'num_hypernet_layers' : [1, 4], # min, max (suggest_int, 1)
    #                    'weight_decay' : [1.0e-6, 1.0e-2], # min, max (suggest_float log scale)
    #                    'sigma_kernel': [0.1, 1.0], # min, max (suggest_float log scale)
    #                    'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
    #                    'loss_scaling_ode' : [0.1, 100], # min, max (suggest_float log scale)
    #                    'loss_scaling_sde' : [0.1, 100], # min, max (suggest_float log scale)
    #                    'loss_scaling_poisson' : [0.001, 1], # min, max (suggest_float log scale) 

    #     }
    elif config.dataset == 'MSK-Chord':
        params_dict = {
                        # 'batch_size': [64], # 128 (suggest_categorical)
                       'lr': [5e-3, 2e-3, 1e-3, 1e-4], # (suggest_categorical)
                       'z_dim_static' : [2, 8], # min, max (suggest_int, 1)
                       's_dim_static' : [2, 8], # min, max (suggest_int, 1)  
                       'latent_dim' : [1, 5], # min, max (suggest_int, 1)
                       'hidden_dim': [16, 32, 64, 128], # 128 (suggest_categorical)
                       'num_odelayers' : [1, 4], # min, max (suggest_int, 1)
                       'num_hypernet_layers' : [1, 5], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-8, 1.0e-3], # min, max (suggest_float log scale)
                       'sigma_kernel': [0.01, 1.0], # min, max (suggest_float log scale)
                       'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [0.1, 100], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.1, 100], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.001, 1], # min, max (suggest_float log scale) 

        }
    elif config.dataset == 'Physionet_2012':
        params_dict = {
                        # 'batch_size': [64], # 128 (suggest_categorical)
                       'lr': [5e-3, 2e-3, 1e-3, 1e-4], # (suggest_categorical)
                       'z_dim_static' : [2, 8], # min, max (suggest_int, 1)
                       's_dim_static' : [2, 8], # min, max (suggest_int, 1)  
                       'latent_dim' : [2, 9], # min, max (suggest_int, 1)
                       'hidden_dim': [32, 64, 128], # 128 (suggest_categorical)
                       'num_odelayers' : [1, 4], # min, max (suggest_int, 1)
                       'num_hypernet_layers' : [1, 5], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-8, 1.0e-3], # min, max (suggest_float log scale)
                       'sigma_kernel': [0.01, 1.0], # min, max (suggest_float log scale)
                       'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [0.1, 100], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.1, 100], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.001, 1], # min, max (suggest_float log scale) 

        }
    elif config.dataset == 'PPMI':
        params_dict = {'batch_size': [128, 256], # 128 (suggest_categorical)
                       'lr': [5e-3, 2e-3, 1e-3, 1e-4], # (suggest_categorical)
                       'z_dim_static' : [2, 8], # min, max (suggest_int, 1)
                       's_dim_static' : [2, 8], # min, max (suggest_int, 1)  
                       'latent_dim' : [2, 15], # min, max (suggest_int, 1)
                       'hidden_dim': [32, 64, 128], # 128 (suggest_categorical)
                       'num_odelayers' : [1, 4], # min, max (suggest_int, 1)
                       'num_hypernet_layers' : [1, 5], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-8, 1.0e-3], # min, max (suggest_float log scale)
                       'sigma_kernel': [0.01, 1.0], # min, max (suggest_float log scale)
                       'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [0.1, 100], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.1, 100], #0.001, 1 min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.001, 1], # min, max (suggest_float log scale) 

        }
    elif config.dataset == 'AVE0005':
        params_dict = {'batch_size': [128, 256], # 128 (suggest_categorical)
                       'lr': [1e-2, 5e-3, 2e-3, 1e-3, 1e-4], # (suggest_categorical)
                       'z_dim_static' : [2, 8], # min, max (suggest_int, 1)
                       's_dim_static' : [2, 8], # min, max (suggest_int, 1)  
                       'latent_dim' : [2, 23], # min, max (suggest_int, 1)
                       'hidden_dim': [32, 64, 128], # 128 (suggest_categorical)
                       'num_odelayers' : [1, 4], # min, max (suggest_int, 1)
                       'num_hypernet_layers' : [1, 5], # min, max (suggest_int, 1)
                       'weight_decay' : [1.0e-8, 1.0e-3], # min, max (suggest_float log scale)
                       'sigma_kernel': [0.01, 1.0], # min, max (suggest_float log scale)
                       'drop_dec' : [0., 0.1, 0.2], # (suggest_categorical)
                       'loss_scaling_ode' : [0.1, 100], # min, max (suggest_float log scale)
                       'loss_scaling_sde' : [0.1, 100], # min, max (suggest_float log scale)
                       'loss_scaling_poisson' : [0.001, 1], # min, max (suggest_float log scale) 

        }
    else:
        raise ValueError("Dataset not recognized.")

    n_trials = 1
    best_params, study = optuna_hyperparameter_search(config, params_dict, n_trials, n_jobs=1, message_config=message) 