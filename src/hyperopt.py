import optuna
from optuna.logging import set_verbosity, WARNING, INFO
import os
import warnings
import time
import torch
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from data_loader.load_data import load_dataset, load_dataset_CV, get_data, get_loader
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
import gc
import json
import random
import numpy as np
import copy
import traceback
from src.generative_model import Generative_Model_Longi_Static
# from src.evaluation import *



def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

"""
=========================== Version with validation losses as score functions of the optuna optimization ===========================
"""


def optuna_hyperparameter_search(base_config, params_dict, n_trials=100, n_jobs=1, message_config=''):

    set_seed(base_config.seed)

    config_and_data = get_data(base_config, normalize_long=False)
    n_samples = config_and_data[1].shape[0]
    idx_train, idx_val = train_test_split(torch.arange(n_samples), test_size=0.2, random_state=base_config.seed, shuffle=True)
    base_config, train_dataset, val_dataset = load_dataset_CV(config_and_data, idx_train, idx_val, only_data=True)  
    static_types = train_dataset.get_static_types() 

    # Number of losses
    nb_tasks = 1
    if base_config.sde:
        nb_tasks += 1
    if base_config.static_data:
        nb_tasks += 1
    if base_config.estim_event_rate:
        nb_tasks += 1
    

    def objective(trial: optuna.Trial):

        torch.cuda.empty_cache()
        gc.collect()
        
        config = copy.deepcopy(base_config)
        set_seed(base_config.seed)
        config.epoch_init = 1

        # Hyperparameter search space
        ## Training parameters
        if 'lr' in params_dict:
            config.lr = trial.suggest_categorical('lr', params_dict['lr'])
        if 'batch_size' in params_dict:
            config.batch_size = trial.suggest_categorical('batch_size', params_dict['batch_size'])

        ## Parameters for statics 
        if "z_dim_static" in params_dict and 's_dim_static' in params_dict:     
            config.z_dim_static = trial.suggest_int('z_dim_static', *params_dict['z_dim_static'], 1)
            config.s_dim_static = trial.suggest_int('s_dim_static', *params_dict['s_dim_static'], 1)

        ## Parameters for the mean ODE network
        if 'latent_dim' in params_dict:
            config.latent_dim = trial.suggest_int('latent_dim', *params_dict['latent_dim'], 1)
        if 'hidden_dim' in params_dict:
            config.nhidden = trial.suggest_categorical('nhidden', params_dict['hidden_dim'])
        if 'num_odelayers' in params_dict:
            config.num_ode_layers =  trial.suggest_int('num_odelayers', *params_dict['num_odelayers'], 1)
        if config.latent_model == 'HyperNDEs':
            if 'num_hypernet_layers' in params_dict:
                config.num_hypernet_layers = trial.suggest_int('num_hypernet_layers', *params_dict['num_hypernet_layers'], 1)
        if 'weight_decay' in params_dict:
            config.weight_decay_drift = trial.suggest_float('weight_decay_drift', *params_dict['weight_decay'], log=True)

        ## Parameters Lambda network
        # if 'hidden_dim' in params_dict:
        #     config.lambda_mlp_size = trial.suggest_categorical('lambda_mlp_size', params_dict['hidden_dim'])
        if 'lambda_act' in params_dict:
            config.lambda_act = trial.suggest_categorical('lambda_act', params_dict['lambda_act'])

        ## Parameters for SDE network and training
        if config.sde:
            # if 'hidden_dim' in params_dict:
            #     config.diff_mlp_size = trial.suggest_categorical('diff_mlp_size', params_dict['hidden_dim'])
            # if 'diff_mlp_num_layers' in params_dict:
            #     config.diff_mlp_num_layers = trial.suggest_int('diff_mlp_num_layers', *params_dict['diff_mlp_num_layers'], 1)
            if 'weight_decay' in params_dict:
                config.weight_decay_diff = trial.suggest_float('weight_decay_diff', *params_dict['weight_decay'], log=True)
            if 'sigma_kernel' in params_dict:
                config.sigma_kernel = trial.suggest_float("sigma_kernel", *params_dict['sigma_kernel'], log=True)

        ## Decoder
        if 'act_dec' in params_dict:
            config.act_dec = trial.suggest_categorical('act_dec', params_dict['act_dec']) 
        if 'hidden_dim' in params_dict:
            config.nhidden_dec = trial.suggest_categorical('nhidden_dec', params_dict['hidden_dim'])
        # if 'drop_dec' in params_dict:
            # config.drop_dec = trial.suggest_categorical('drop_dec', params_dict['drop_dec'])

        ## Loss scalings
        config.loss_scaling_static = 1.0
        if 'loss_scaling_ode' in params_dict and config.sde_split_training == True:  
            config.loss_scaling_ode = trial.suggest_float('loss_scaling_ode', *params_dict['loss_scaling_ode'], log=True)
        if config.sde:
            if 'loss_scaling_sde' in params_dict:
                config.loss_scaling_sde = trial.suggest_float('loss_scaling_sde', *params_dict['loss_scaling_sde'], log=True)
        if config.estim_event_rate:
            if 'loss_scaling_poisson' in params_dict:
                config.loss_scaling_poisson = trial.suggest_float('loss_scaling_poisson', *params_dict['loss_scaling_poisson'], log=True)    

        print("\n" + "="*50)
        print(f'Trial {trial.number} | {trial.params}')
        train_dataloader = get_loader(config, train_dataset)
        val_dataloader = get_loader(config, val_dataset)
    
        try: 
            print("\n" + "="*50)
            print("Building model...")
            model = Generative_Model_Longi_Static(config, static_types, hyperopt_mode=True)
            print("\n" + "="*50)
            print("Training model...")
            best_val_losses = model.train(train_dataloader, val_dataloader, save_model=False, differential_privacy=False) 
            
            print("\n" + "="*50)
            print(f"Trial {trial.number} | " + " | ".join(f"Score {key}: {value:.4f}" for key, value in best_val_losses.items()))
            return tuple(best_val_losses[key] for key in sorted(best_val_losses.keys()))

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
            return tuple(float('nan') for i in range(nb_tasks))

    # Set up Optuna study
    optuna_study_name = base_config.optuna_study
    base_config.save_path = os.path.join(base_config.init_path + base_config.save_dir, f"{optuna_study_name}")
    os.makedirs(base_config.save_path, exist_ok=True)

    config_file_name = os.path.join(base_config.save_path, 'config_init.txt')
    with open(config_file_name, 'wt') as config_file:
        config_file.write(message_config)
        config_file.write('\n')

    # Optuna study setup
    db_file = optuna_study_name + ".db"
    db_file = os.path.join(base_config.save_path, db_file)
    study_name = os.path.join(base_config.save_path, optuna_study_name)

    set_verbosity(INFO)
    
    # storage = optuna.storages.RDBStorage(
    #     url='sqlite:///'+db_file,
    #     engine_kwargs={"connect_args": {"timeout": 120}} 
    # )
    storage = 'sqlite:///'+db_file
    
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        print("This optuna study ({}) does not exist. We create a new study.".format(db_file))
        # default_params = {"lr":0.002, 
        #                   "batch_size":256, 
        #                   "z_dim_static":4, 
        #                   "s_dim_static":5, 
        #                   "latent_dim":4, 
        #                   "nhidden":32, 
        #                   "num_odelayers":1, 
        #                   "num_hypernet_layers":3, 
        #                   "weight_decay_drift":0.000707,
        #                   "weight_decay_diff":0.000051, 
        #                   "sigma_kernel":0.206652,
        #                   "drop_dec":0., 
        #                   "loss_scaling_ode":475.042843, 
        #                   "loss_scaling_sde":0.013119, 
        #                   "loss_scaling_poisson":0.965897}
        # default_params = {"lr":0.001, 
        #                   "batch_size":128, 
        #                   "z_dim_static":4, 
        #                   "s_dim_static":3, 
        #                   "latent_dim":9, 
        #                   "nhidden":64, 
        #                   "num_odelayers":1, 
        #                   "num_hypernet_layers":4, 
        #                   "weight_decay_drift":0.0000707,
        #                   "weight_decay_diff":0.000051, 
        #                   "sigma_kernel":0.1,
        #                   "nhidden_dec":64,
        #                 #   "drop_dec":0., 
        #                   "loss_scaling_ode":1, 
        #                   "loss_scaling_sde":1, 
        #                   "loss_scaling_poisson":0.1}
        study = optuna.create_study(directions=["minimize"]*nb_tasks, study_name=study_name, storage=storage, load_if_exists=True) 
        # study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)  # timeout=300)
    
    print("\n" + "="*50)
    print(f"✅ Number of trials on the Pareto front: {len(study.best_trials)}")
    trial_with_highest_score_static = min(study.best_trials, key=lambda t: t.values[0])
    print("Trial with highest score static: ")
    print(f"\tnumber: {trial_with_highest_score_static.number}")
    print(f"\tparams: {trial_with_highest_score_static.params}")
    print(f"\tvalues: {trial_with_highest_score_static.values}")
    trial_with_highest_score_longi = min(study.best_trials, key=lambda t: t.values[1])
    print("Trial with highest score longitudinal: ")
    print(f"\tnumber: {trial_with_highest_score_longi.number}")
    print(f"\tparams: {trial_with_highest_score_longi.params}")
    print(f"\tvalues: {trial_with_highest_score_longi.values}")
    trial_with_highest_score_intensity = min(study.best_trials, key=lambda t: t.values[2])
    print("Trial with highest score intensities: ")
    print(f"\tnumber: {trial_with_highest_score_intensity.number}")
    print(f"\tparams: {trial_with_highest_score_intensity.params}")
    print(f"\tvalues: {trial_with_highest_score_intensity.values}")

    # Save the study
    print("\n" + "="*50)
    print("📝 Saving study results...")
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(base_config.save_path, 'study_results.csv'), index=False)
    
    pareto = [
        {"params": t.params, "values": t.values}  # also save their objective values
        for t in study.best_trials
    ]
    with open(os.path.join(base_config.save_path, 'pareto_params.json'), "w") as f:
        json.dump(pareto, f, indent=4)
    return study.best_trials, study

