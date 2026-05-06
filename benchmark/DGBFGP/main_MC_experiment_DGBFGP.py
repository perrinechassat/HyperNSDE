import torch
import pandas as pd
import numpy as np
import os
import argparse
import sys
import copy
from pathlib import Path
import gc
from sklearn.preprocessing import StandardScaler
# DGBFGP_PATH = "/path/to/Documents/DGBFGP"
DGBFGP_PATH = "/path/to/DGBFGP"
sys.path.append(DGBFGP_PATH)
from main import main
from options import parse_arguments
from models import DGBFGP
sys.path.append('./')
from preprocess_MC import preprocess_simulation_data, train_val_test_split
from generate_DGBFGP import generate_DGBFGP

TRAINING = True
GENERATION = True
PREPROCESSING = True
# INIT_PATH = "/path/to/HyperNSDE"
INIT_PATH = "/path/to/HyperNSDE"

def run(args, mc_id):
    P_tot = args.P
    P_val_test = int(P_tot*0.2)
    P_test = int(P_val_test/2)
    P_val = int(P_val_test/2)

    T_test = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_input = args.dataset_init # "monte_carlo" # monte_carlo_lbda_dep_50pct
    destination = os.path.join(INIT_PATH, f"benchmark/DGBFGP/results/{folder_input}_Ntrain200/MC_{mc_id}")
    np.random.seed(mc_id)

    # ---- Preprocessing ----
    if PREPROCESSING:
        print("     > [Preprocessing ...] \n")
        
        data = preprocess_simulation_data(folder_input, mc_id)
        x, y, y_mask = data["X"], data["y"], data["y_mask"]

        # Ensure IDs start from 0 and are continuous
        ids = np.unique(x[:, 0])
        id_dict = {id: i for i, id in enumerate(ids)}
        x[:, 0] = [id_dict[id] for id in x[:, 0]]

        # 2. Create directories
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(destination, split), exist_ok=True)

        # Re-apply NaNs for proper standard scaling calculations
        y[~y_mask.astype(bool)] = np.nan

        # 3. Split the data
        (x_train, x_val, x_test, y_train, y_val, y_test, 
        y_mask_train, y_mask_val, y_mask_test, 
        y_train_init, y_val_init, y_test_init, idx_train, idx_val) = train_val_test_split(
            x, y, y_mask, P_test, P_val, T_test, seed=mc_id
        )

        # 4. Standardize Covariates (Index 1=Time, Index 2=Static_Real) 
        # Index 3 (Static_Cat) is left unscaled.
        numerical_covariates = [1, 2] 
        x_scaler = StandardScaler()
        x_train[:, numerical_covariates] = x_scaler.fit_transform(x_train[:, numerical_covariates])
        x_val[:, numerical_covariates] = x_scaler.transform(x_val[:, numerical_covariates])
        x_test[:, numerical_covariates] = x_scaler.transform(x_test[:, numerical_covariates])

        # 5. Standardize Targets (Y)
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

        # 6. Save Arrays to CSV/NPY
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

        del x, y, y_mask
        del x_train, x_val, x_test
        del y_train, y_val, y_test
        del y_train_init, y_val_init, y_test_init
        gc.collect()

    # ---- Training ----
    args.train_data_source_path = os.path.join(destination, "train")
    args.val_data_source_path = os.path.join(destination, "val")
    args.test_data_source_path = os.path.join(destination, "test")
    args.output_dir = destination
    if TRAINING:
        print("     > [Training ...] \n")
        main(args)

    # ---- Generation ----
    if GENERATION:
        print("     > [Generation ...] \n")
        n_gen = 50
        path_data = os.path.join(INIT_PATH, 'datasets/Simu_OU/{}/simulated_dataset_MC_{}'.format(folder_input, mc_id))
        path_ckpts = destination
        T_gen = torch.linspace(0,7,200)

        gen_stat, gen_long = generate_DGBFGP(args=args, 
                                            path_data=path_data, 
                                            path_ckpts=path_ckpts, 
                                            idx_train=idx_train, 
                                            idx_val=idx_val, 
                                            x_scaler=x_scaler, 
                                            y_scaler=y_scaler, 
                                            n_gen=n_gen, 
                                            T_gen=T_gen, 
                                            device=device, 
                                            s_dim_static=5, 
                                            z_dim_static=4, 
                                            batch_size=128)
        
        # SAVING 
        save_dir = os.path.join(destination, "generated_samples_more")
        os.makedirs(save_dir, exist_ok=True)
        stat_save_path = os.path.join(save_dir, "gen_stat.pt")
        long_save_path = os.path.join(save_dir, "gen_long.pt")
        torch.save(gen_stat, stat_save_path)
        torch.save(gen_long, long_save_path)
        print(f"     > Successfully saved generated static features to: {stat_save_path}")
        print(f"     > Successfully saved generated longitudinal features to: {long_save_path}")

        del gen_stat, gen_long

        # ---- Optional: Plot the 50 trajectories for the first patient's L1 feature ----
        # import matplotlib.pyplot as plt
        # k_s = 0
        # plt.figure(figsize=(10, 6))
        # for i in range(0, gen_long.shape[1]):
        #     plt.plot(T_gen.numpy(), gen_long[k_s, i, :, 0].numpy())
        # plt.title(f"Generated GP Trajectories (Feature L1)")
        # plt.xlabel("Time")
        # plt.ylabel("L1 Value")
        # plt.show()
    

if __name__ == "__main__":
    args = parse_arguments()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    mc_group = args.seed

    for mc_id in range(mc_group*5, (mc_group+1)*5):
        args_i = copy.deepcopy(args)
        args_i.seed = mc_id
        run(args_i, mc_id)
    
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # Cleans up inter-process communication memory
        print("--- Cleared Memory ---")