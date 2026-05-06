#!/usr/bin/ipython
import os
import warnings
import numpy as np
import time
import torch
import json
from sklearn.model_selection import KFold, train_test_split
import random
import optuna
from optuna.logging import set_verbosity, WARNING, INFO
import copy
import yaml
import gc
import traceback

import sys
sys.path.append('../')
sys.path.append('../../')
# sys.path.append('../../../')
# sys.path.append('../../../../')
# from src.utils import load_yaml_config

# MULTINODES_PATH = "/path/to/Documents/MultiNODEs/src" # MAC
MULTINODES_PATH = "/path/to/MultiNODEs/src" # CLEPS
# MULTINODES_PATH = "$WORK/external_libs/MultiNODEs/src" # JEAN-ZAY
# MULTINODES_PATH = "../../../MultiNODEs"
sys.path.append(MULTINODES_PATH)
from models.parser import base_parser
from models.hyperopt import Hyperopt
from data.load_data import get_loader, load_data_other, load_dataset_CV


# AJOUTER LA CROSS-VAL

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':

    ''' ----------------------------------------------------------- '''
    ''' -------------------------- Config ------------------------- '''
    ''' ----------------------------------------------------------- '''

    base_config = base_parser()
    external_config = load_yaml_config(base_config.config_file)

    if base_config.GPU == True:
        # base_config.GPU = torch.cuda.is_available()
        base_config.GPU = [0] if torch.cuda.is_available() else False
    else:
        base_config.GPU = False

    message = ''
    message += '----------------- Options ---------------\n'
    for key, value in external_config.items():
        setattr(base_config, key, value)
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
    message += '----------------- End -------------------'
    print(message)
    
    # Seed for reproducibility
    set_seed(base_config.seed)
    
    base_config.train_dir = os.path.join(base_config.init_path + base_config.train_dir, base_config.dataset)
    base_config.save_path_losses = ''

    ''' ----------------------------------------------------------- '''
    ''' -------------------------- Hyperopt ----------------------- '''
    ''' ----------------------------------------------------------- '''

    n_trials = 20
    n_jobs = 1

    config_and_data = load_data_other(base_config)
    base_config, data = config_and_data[0], config_and_data[1:]
    n_samples = data[0].shape[0]

    def objective(trial: optuna.Trial):

        torch.cuda.empty_cache()
        gc.collect()

        config = copy.deepcopy(base_config)
        set_seed(base_config.seed)

        # Hyperparameter search space
        config.num_epochs = 600
       
        config.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        config.latent_dim = trial.suggest_uniform('latent_dim',0.1,3.0)
        config.nhidden = trial.suggest_uniform('nhidden',0.5,5.0) #5
        config.batch_size =  trial.suggest_uniform('batch_size',0.01,0.25) # 0.25
        config.s_dim_static = trial.suggest_int('s_dim_static',2,8,1)
        config.z_dim_static = trial.suggest_int('z_dim_static',2,8,1)
        config.drop_dec =  trial.suggest_uniform('drop_dec',0,0.6)
        config.rnn_nhidden_enc =  trial.suggest_uniform('rnn_nhidden_enc',0.1,0.5)
        config.rnn_nhidden_dec =  trial.suggest_uniform('rnn_nhidden_dec',0.1,0.5)
        config.num_ode_layers =  trial.suggest_int('num_ode_layers',1,4,1)
        config.act_ode =  trial.suggest_categorical('act_ode',['tanh','relu','none'])
        config.type_enc =  trial.suggest_categorical('type_enc',['LSTM','RNN'])
        # if (config.type_enc =='RNN'):
        #     config.act_rnn =  trial.suggest_categorical('act_rnn',['tanh','none'])
        # else:
        #     config.act_rnn = None
        # config.type_dec =  trial.suggest_categorical('type_dec',['LSTM','orig']) 
        config.act_dec =  trial.suggest_categorical('act_dec',['tanh','relu','none'])
        # config.scaling_ELBO = trial.suggest_uniform('scaling_ELBO',0,2)
        config.batch_norm_static = True
        config.epoch_init = 1
        
        # K-Fold Cross Validation
        n_fold = 5
        kfold = KFold(n_splits=n_fold, shuffle=True)
        fold = 0
        avg_val_loss = 0.0

        for train_idx, val_idx in kfold.split(torch.arange(n_samples)):

            fold += 1

            # only one fold for the real datasets
            if fold > 1: 
                print(f"\n=== Trial {trial.number} | Average Validation Loss over {n_fold} folds: {avg_val_loss:.4f} ===\n")
                return avg_val_loss

            print(f"\n{'='*20} Fold {fold}/{n_fold} {'='*20}\n")

            config_k, train_dataloader, val_dataloader = load_dataset_CV(config, data, train_idx, val_idx, only_data=False)  
            print("Number of training samples:", len(train_dataloader.dataset))
            print("Number of validation samples:", len(val_dataloader.dataset))
            print("Current hyperparameters:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            print("Batch size (absolute):", train_dataloader.batch_size)

            try: 
                print("\n" + "="*50)
                print("Training model...")
                best_val_loss = Hyperopt(config_k, train_dataloader, val_dataloader).run()
                
                print("\n" + "="*50)
                print(f"Trial {trial.number}, Kfold {fold} | Score : {best_val_loss:.4f}")
                avg_val_loss += best_val_loss

            # except torch.cuda.OutOfMemoryError as e:
            #     print("Out of Memory! Penalizing this configuration.")
            #     torch.cuda.empty_cache()
            #     # On retourne une valeur très mauvaise au lieu de Pruned
            #     # Cela apprend à Optuna que ces paramètres (ex: trop gros batch) sont mauvais
            #     for key, value in trial.params.items():
            #         print(f"    {key}: {value}")
            #     return float('nan')

            except Exception as e:  # invalid set of params
                print("\n" + "!" * 80)
                print("Exception during Optuna trial")
                print("Type:", type(e).__name__)
                print("Message:", e)
                print("\nTraceback:")
                traceback.print_exc() 
                print("\nTrial failed with params:")
                for key, value in trial.params.items():
                    print(f"    {key}: {value}")
                print("\n" + "!" * 80)
                return float('nan')

            finally:
                # Nettoyage systématique
                gc.collect()
                torch.cuda.empty_cache()
            
        avg_val_loss /= n_fold
        print(f"\n=== Trial {trial.number} | Average Validation Loss over {n_fold} folds: {avg_val_loss:.4f} ===\n")
        return avg_val_loss


    # Set up Optuna study
    optuna_study_name = base_config.optuna_study
    base_config.save_path = os.path.join(base_config.init_path + base_config.save_path, f"{optuna_study_name}")
    os.makedirs(base_config.save_path, exist_ok=True)

    config_file_name = os.path.join(base_config.save_path, 'config_init.txt')
    with open(config_file_name, 'wt') as config_file:
        config_file.write(message)
        config_file.write('\n')

    # Optuna study setup
    db_file = optuna_study_name + ".db"
    db_file = os.path.join(base_config.save_path, db_file)
    study_name = os.path.join(base_config.save_path, optuna_study_name)

    # Wrapper RDBStorage INDISPENSABLE sur Jean Zay pour gérer les accès concurrents
    # Le timeout de 120s permet d'attendre si un autre job écrit dans la base
    # storage = optuna.storages.RDBStorage(
    #     url='sqlite:///'+db_file,
    #     engine_kwargs={"connect_args": {"timeout": 120}} 
    # )
    storage='sqlite:///'+db_file

    set_verbosity(INFO)

    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(
            study_name=study_name, 
            storage=storage
            )
    else:
        print("This optuna study ({}) does not exist. We create a new study.".format(db_file))
        study = optuna.create_study(
            direction='minimize', 
            study_name=study_name, 
            storage=storage, 
            load_if_exists=True
            )
        default_params = {"lr":0.001,
                          "latent_dim":1, # 1.5,
                          "nhidden":3., #5
                          "batch_size":0.1, # 0.223764,
                          "s_dim_static":5,
                          "z_dim_static":5,
                          "drop_dec":0.505877,
                          "rnn_nhidden_enc":0.317421,
                          "rnn_nhidden_dec":0.5,
                          "num_ode_layers":1}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
        
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)  # timeout=300)


    # Print best trial
    print("\n" + "="*50)
    print("✅ Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
   
    # Save the study
    print("\n" + "="*50)
    print("📝 Saving study results...")
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(base_config.save_path, 'study_results.csv'), index=False)

    with open(os.path.join(base_config.save_path, 'best_params.json'), "w") as f:
        json.dump(study.best_params, f)
        
    


