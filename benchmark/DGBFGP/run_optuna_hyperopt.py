import torch
import pandas as pd
import numpy as np
import os
import argparse
import sys
import copy
from pathlib import Path
from sklearn.preprocessing import StandardScaler
# DGBFGP_PATH = "/path/to/Documents/DGBFGP"
DGBFGP_PATH = "/path/to/DGBFGP"
sys.path.append(DGBFGP_PATH)
from main import main
from options import parse_arguments
sys.path.append('./')
from preprocess_MC import preprocess_simulation_data, train_val_test_split
import optuna
from optuna.logging import set_verbosity, WARNING, INFO
import json

folder_input = "monte_carlo" # "monte_carlo"
optuna_study_name = "optuna_DGBFGP_main_v4"
n_trials = 10
current_directory = Path(__file__).resolve().parent

if __name__ == "__main__":
    args = parse_arguments()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    mc_id = 0
    args.seed = mc_id
    # optuna_study_name = args.optuna_study_name
    # folder_input = args.dataset_init
    destination_optuna = os.path.join(current_directory, f"hyperopt/{optuna_study_name}")
    os.makedirs(destination_optuna, exist_ok=True)
    # current_directory = Path(__file__).resolve().parent

    P_tot = args.P
    P_val_test = int(P_tot*0.2)
    P_test = int(P_val_test/2)
    P_val = int(P_val_test/2)
    T_test = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- Preprocessing ----
    print("     > [Preprocessing ...] \n")

    np.random.seed(mc_id)
    data = preprocess_simulation_data(folder_input, mc_id)
    x, y, y_mask = data["X"], data["y"], data["y_mask"]

    # Ensure IDs start from 0 and are continuous
    ids = np.unique(x[:, 0])
    id_dict = {id: i for i, id in enumerate(ids)}
    x[:, 0] = [id_dict[id] for id in x[:, 0]]

    # Create directories
    destination = os.path.join(current_directory, f"results/{folder_input}/MC_{mc_id}")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(destination, split), exist_ok=True)

    # Re-apply NaNs for proper standard scaling calculations
    y[~y_mask.astype(bool)] = np.nan

    # Split the data
    (x_train, x_val, x_test, y_train, y_val, y_test, 
     y_mask_train, y_mask_val, y_mask_test, 
     y_train_init, y_val_init, y_test_init, idx_train, idx_val) = train_val_test_split(
        x, y, y_mask, P_test, P_val, T_test, seed=mc_id
    )

    # Standardize Covariates (Index 1=Time, Index 2=Static_Real) 
    numerical_covariates = [1, 2] 
    x_scaler = StandardScaler()
    x_train[:, numerical_covariates] = x_scaler.fit_transform(x_train[:, numerical_covariates])
    x_val[:, numerical_covariates] = x_scaler.transform(x_val[:, numerical_covariates])
    x_test[:, numerical_covariates] = x_scaler.transform(x_test[:, numerical_covariates])

    # Standardize Targets (Y)
    y_train[~y_mask_train.astype(bool)] = np.nan
    y_val[~y_mask_val.astype(bool)] = np.nan
    y_test[~y_mask_test.astype(bool)] = np.nan

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)

    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    y_train_init = y_scaler.transform(y_train_init)
    y_val_init = y_scaler.transform(y_val_init)
    y_test_init = y_scaler.transform(y_test_init)

    # Fill NaNs back to 0 so the PyTorch dataloader doesn't break
    y_train[np.isnan(y_train)] = 0
    y_val[np.isnan(y_val)] = 0
    y_test[np.isnan(y_test)] = 0
    y_train_init = np.nan_to_num(y_train_init)
    y_val_init = np.nan_to_num(y_val_init)
    y_test_init = np.nan_to_num(y_test_init)

    # Save Arrays to CSV/NPY
    fmt_x = ['%d', '%.5f', '%.5f', '%d'] # ID(int), Time(float), StaticReal(float), StaticCat(int)
    fmt_y = '%.5f'
    fmt_mask = '%d'

    def save_split(split_name, x_arr, y_arr, mask_arr, init_arr):
        np.savetxt(os.path.join(destination, f'{split_name}/label.csv'), x_arr, delimiter=',', fmt=fmt_x)
        np.savetxt(os.path.join(destination, f'{split_name}/data.csv'), y_arr, delimiter=',', fmt=fmt_y)
        np.savetxt(os.path.join(destination, f'{split_name}/mask.csv'), mask_arr, delimiter=',', fmt=fmt_mask)
        np.save(os.path.join(destination, f'{split_name}/init_data.csv.npy'), init_arr) # Required for DGBFGP's AVI

    save_split('train', x_train, y_train, y_mask_train, y_train_init)
    save_split('val', x_val, y_val, y_mask_val, y_val_init)
    save_split('test', x_test, y_test, y_mask_test, y_test_init)

    print("Data successfully generated and saved to:", destination)

    args.train_data_source_path = os.path.join(destination, "train")
    args.val_data_source_path = os.path.join(destination, "val")
    args.test_data_source_path = os.path.join(destination, "test")
    args.output_dir = destination
    

    ''' ----------------------------------------------------------- '''
    ''' -------------------------- Hyperopt ----------------------- '''
    ''' ----------------------------------------------------------- '''

    def objective(trial: optuna.Trial):

        args_trial = copy.deepcopy(args)

        # -- GP & Model Architecture --
        # args.M = trial.suggest_int("M", 20, 100, step=10) 
        # args_trial.scale = trial.suggest_float("scale", 0.1, 0.4, step=0.1)
        args_trial.latent_dim = trial.suggest_int("latent_dim", 4, 32)
        args_trial.id_embed_dim = trial.suggest_int("id_embed_dim", 4, 32)
        # args_trial.B = trial.suggest_float("B", 0.001, 1.0, log=True) # KL Weight
        # args_trial.p_drop = trial.suggest_float("p_drop", 0, 0.3, step=0.1)
        # args_trial.lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
        # args_trial.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

        print(f"\n=== Starting Trial {trial.number} ===")
        print(f"Testing: scale={args_trial.scale:.3f}, p_drop={args_trial.p_drop:.2f}, " # M={args_trial.M}, B={args_trial.B:.5f},
            f"latent={args_trial.latent_dim}, batch={args_trial.batch_size}, lr={args_trial.lr:.5f}, embed={args_trial.id_embed_dim}")
        
        # Optional: adjust epochs for faster tuning
        args_trial.n_epoch = 500
        print(f"\n=== Starting Trial {trial.number} ===")

        try: 
            print("     > [Training ...] \n")
            val_loss = main(args_trial, return_best_loss=True)
            print(f"\n=== Trial {trial.number} | Validation Loss o: {val_loss:.4f} ===\n")
            return val_loss

        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print("Trial failed with params:")  
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            raise optuna.TrialPruned()
        
        

    # Optuna study setup
    db_file = optuna_study_name + ".db"
    db_file = os.path.join(destination_optuna, db_file)
    study_name = os.path.join(destination_optuna, optuna_study_name)
    set_verbosity(INFO)
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+db_file)
    else:
        print("This optuna study ({}) does not exist. We create a new study.".format(db_file))
        study = optuna.create_study(direction='minimize', study_name=study_name, storage='sqlite:///'+db_file)
        default_params = {
                        # "scale":0.2,
                          "latent_dim":16,
                          "id_embed_dim":13,
                        #   "p_drop":0.1,
                        #   "lr":0.009, 
                        #   "batch_size":256
                          }
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))

        
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)  # timeout=300)

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
    trials_df.to_csv(os.path.join(destination_optuna, 'study_results.csv'), index=False)

    with open(os.path.join(destination_optuna, 'best_params.json'), "w") as f:
        json.dump(study.best_params, f)
        