import json
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('../')
sys.path.append('../../')
from data_loader.load_data import get_data, read_csv_values, weighter, normalize_long
from src.parser import base_parser
from src.utils import onehot_batch_norm
import os
import glob
import pickle
from sklearn.model_selection import train_test_split
RTSGAN_PATH = "/path/to/Documents/RTSGAN"
import sys
sys.path.append(RTSGAN_PATH)
sys.path.append(RTSGAN_PATH + "/general")
from general.missingprocessor import *

def to_canonical(gen_dict):
    """
    Convert loaded generation output to canonical format.
    """
    X = gen_dict["Long_Values"]
    W = gen_dict["Stat_Values"]

    if "Mask" in gen_dict and gen_dict["Mask"] is not None:
        M = gen_dict["Mask"]
    else:
        M = np.ones_like(X)

    T = gen_dict["Time_Grid"]

    return {
        "X": X.astype(np.float32),
        "M": M.astype(np.float32),
        "W": W.astype(np.float32),
        "T": T.astype(np.float32),
    }

def save_npz(path, data):
    np.savez_compressed(path, **data)

def save_real_mc(id_mc, orig_mc, out_dir, name="real.pt"):
    real = {
        "train": {
            "X": orig_mc["train"]["x"].numpy(),
            "M": orig_mc["train"]["mask"].numpy(),
            "W": orig_mc["train"]["W"].numpy(),
        },
        "test": {
            "X": orig_mc["val"]["x"].numpy(),
            "M": orig_mc["val"]["mask"].numpy(),
            "W": orig_mc["val"]["W"].numpy(),
        },
        "T": orig_mc["T"].numpy(),
        "static_types": orig_mc["static_types"],
        "full_reg_ode": orig_mc["full_reg_ode"].numpy(),
        "x_reg_sde": orig_mc["x_reg_sde"].numpy(),
        "idx_train": orig_mc["idx_train"], 
        "idx_test": orig_mc["idx_test"], 
        # "s_mean": orig_mc["s_mean"], 
        # "s_var": orig_mc["s_var"], 
        "absmax_params_long": orig_mc["absmax_params_long"],
    }
    torch.save(real, out_dir / name)

def save_syn_mc(name, gen, out_dir, key="posterior"):
    if gen is None or gen.get(key) is None:
        return
    canon = to_canonical(gen[key])
    save_npz(out_dir / f"{name}.npz", canon)


def load_sample(path, to_torch=False, final_T=1.0, t0_in_static=True, param_norm_long=None):
    d = torch.load(path, map_location='cpu')
    out = {}
    for k, v in d.items():
        if k == 'Gen_Time':
            k = 'Time_Grid'
        if k == 'W_train':
            k = 'Mask'
        if to_torch:
            out[k] = v
        else:
            try:
                out[k] = v.numpy()
            except Exception:
                try:
                    out[k] = v.cpu().numpy()
                except Exception:
                    out[k] = v

    if param_norm_long is not None:
        # Denormalize longitudinal values
        long_vals = out['Long_Values']
        out['Long_Values'] = long_vals * param_norm_long['max'].item()  # absmax denormalization

    # print("N our", long_vals.shape[0])

    if t0_in_static:
        # Move t0 longitudinal values to static
        long_vals = out['Long_Values']
        static_vals = out['Stat_Values']
        t0_vals = static_vals[:, -long_vals.shape[2]:]  # (n_samples, n_long_vars)
        out['Long_Values'] = np.where(~np.isnan(long_vals), long_vals + t0_vals[:, np.newaxis, :], np.nan)
        out['Stat_Values'] = static_vals[:, :-long_vals.shape[2]]  # remove t0 from static

    out['Time_Grid'] = out['Time_Grid'] * final_T 

    print("End time point load sample", out["Time_Grid"][-1])
    return out


def load_sample_MultiNODEs(path, to_torch=False, final_T=1.0, t0_in_static=True):
    config = base_parser()
    config.file_dataset = file_dataset
    config.static_data = True
    config.t_visits = t_visits
    config.n_long_var = n_long_var
    config.train_dir = train_dir
    config.long_normalization = 'none'
    config.t0_2_static = False

    config.time_normalization = False
    config, x_orig, mask_orig, T, static_onehot, static_types, static_true_miss_mask, static_vals, params_norm = get_data(config)

    t0_values = torch.full((x_orig.shape[0], x_orig.shape[2]), float('nan'), device=x_orig.device)
    for i in range(x_orig.shape[0]):  # loop over patients
        for j in range(x_orig.shape[2]):  # loop over variables
            valid_idxs = torch.where(mask_orig[i, :, j] > 0)[0]
            if len(valid_idxs) > 0:
                first_idx = valid_idxs[0]
                t0 = x_orig[i, first_idx, j]
                t0_values[i, j] = t0
                
    # compute std only over observed baselines
    baseline_std = torch.zeros(x_orig.shape[2], device=x_orig.device)
    for j in range(x_orig.shape[2]):
        vals = t0_values[:, j]
        baseline_std[j] = vals.std(unbiased=True)
    baseline_std = baseline_std.clamp(min=1e-6)

    # load gen samples
    d = torch.load(path, map_location='cpu')
    out = {}
    for k, v in d.items():
        if k == 'Gen_Time':
            k = 'Time_Grid'
        if k == 'W_train':
            k = 'Mask'
        if to_torch:
            out[k] = v
        else:
            try:
                out[k] = v.numpy()
            except Exception:
                try:
                    out[k] = v.cpu().numpy()
                except Exception:
                    out[k] = v
    
    # Denormalization
    long_vals = out['Long_Values']
    print("N multinodes", long_vals.shape[0])
    out['Long_Values'] = long_vals * baseline_std.unsqueeze(0).unsqueeze(0).numpy()

    # Inverse substraction of t0
    if t0_in_static:
        # Move t0 longitudinal values to static
        long_vals = out['Long_Values']
        static_vals = out['Stat_Values']
        t0_vals = static_vals[:, -long_vals.shape[2]:]  # (n_samples, n_long_vars)
        out['Long_Values'] = long_vals + t0_vals[:,np.newaxis,:]  # add t0 to longitudinal
        out['Stat_Values'] = static_vals[:, :-long_vals.shape[2]]  # remove t0 from static
    
    out['Time_Grid'] = out['Time_Grid'] * final_T 

    print("End time point load sample", out["Time_Grid"][-1])
    return out


def load_original_data(id_mc, file_dataset, train_dir='', t_visits=200, n_long_var=3, n_samples=1000):
    config = base_parser()
    config.file_dataset = file_dataset
    config.static_data = True
    config.t_visits = t_visits
    config.n_long_var = n_long_var
    config.train_dir = train_dir

    # Load orginal datasets
    data_regular_ode = read_csv_values(os.path.join(train_dir, file_dataset, 'data_long_regular_ode.csv'), header=None, index_col=None, method='numpy')
    data_regular_ode = torch.from_numpy(data_regular_ode).view(len(data_regular_ode), t_visits, n_long_var + 1).float()
    x_reg_ode, mask_reg_ode = weighter(data_regular_ode[:,:,1:])

    data_regular_sde = read_csv_values(os.path.join(train_dir, file_dataset, 'data_long_regular_sde.csv'), header=None, index_col=None, method='numpy')
    data_regular_sde = torch.from_numpy(data_regular_sde).view(len(data_regular_sde), t_visits, n_long_var + 1).float()
    x_reg_sde, mask_reg_sde = weighter(data_regular_sde[:,:,1:])

    config.time_normalization = False
    config.long_normalization = 'absmax'
    config.t0_2_static = True
    config.fixed_init_cond = True
    config, x_norm_t02stat, _, _, _, _, _, _, params_norm = get_data(config)

    config = base_parser()
    config.file_dataset = file_dataset
    config.static_data = True
    config.t_visits = t_visits
    config.n_long_var = n_long_var
    config.train_dir = train_dir  
    config.time_normalization = False
    config.long_normalization = 'none'
    config.t0_2_static = False 
    config.fixed_init_cond = True 
    config, x, mask, T, static_onehot, static_types, static_true_miss_mask, static_vals, _ = get_data(config)
    
    idx_train, idx_test = train_test_split(np.arange(n_samples), test_size=0.2, random_state=id_mc, shuffle=True)
    idx_train, idx_out = train_test_split(idx_train, test_size=0.75, random_state=id_mc, shuffle=True)

    orig_id_mc = {
        "train": {
            "x": x[idx_train], 
            "mask": mask[idx_train], 
            "W": static_vals[idx_train], 
            "W_onehot": static_onehot[idx_train],
            # "W_onehot_norm": static_onehot_norm[idx_train]
        },
        "val": {
            "x": x[idx_test], 
            "mask": mask[idx_test], 
            "W": static_vals[idx_test], 
            "W_onehot": static_onehot[idx_test],
            # "W_onehot_norm": static_onehot_norm[idx_test]
        },
        "full_reg_ode": x_reg_ode,
        "x_reg_sde": x_reg_sde,
        "T": T,
        "idx_train": idx_train, 
        "idx_test": idx_test, 
        "static_types": static_types,
        # "s_mean": s_mean, 
        # "s_var": s_var, 
        "absmax_params_long": params_norm,
    }
    
    return orig_id_mc

def get_unique_file(files, name):
    if len(files) == 0:
        print(f"Warning: no {name} file found")
        return None
    if len(files) > 1:
        raise RuntimeError(f"Multiple {name} files found: {files}")
    return files[0]

def load_results_multiNODEs(id_mc, final_T=1.0, folder='OU_with_best_config'): 
    """Load a saved sample .pth file and convert tensors to numpy arrays.

    Returns a dict with keys used in the saving function:
    - Long_Values: (n_traj, n_timepoints, n_dims_long)
    - Train_Time: (n_timepoints,)
    - Stat_Values: (n_traj, n_dims_static)
    - Mask: optional (n_traj, n_timepoints, n_dims_long)
    - Latent_longitudinal, Latent_residual, Latent_static, Int_lambda, Log_lambda

    """
    DIR_INIT = '/path/to/HyperNSDE/'
    MODEL = 'benchmark/multiNODEs/monte_carlo/{}/MC_{}'.format(folder, id_mc)
    SAMPLES_DIR = os.path.join(DIR_INIT, MODEL, "samples")
    folder = Path(SAMPLES_DIR)
    if folder.exists() and folder.is_dir():
        is_empty = not any(folder.iterdir())
        if is_empty:
            print(f"Warning: samples directory {SAMPLES_DIR} is empty")
            return None
        else:
            rec_files = glob.glob(os.path.join(SAMPLES_DIR, "Best_Rec_*.pth"))
            post_files = glob.glob(os.path.join(SAMPLES_DIR, "Best_Gen_Posterior_SL1_SS1_*.pth"))
            prior_files = glob.glob(os.path.join(SAMPLES_DIR, "Best_Gen_Prior_*.pth"))
            paths = {
                "reconstruction": get_unique_file(rec_files, "reconstruction"),
                "posterior": get_unique_file(post_files, "posterior"),
                "prior": get_unique_file(prior_files, "prior"),
            }
            gen = {}
            for k, p in paths.items():
                if p is None or not os.path.exists(p):
                    print(f"Warning: file for {k} not found at {p} (skipping)")
                    gen[k] = None
                else:
                    gen[k] = load_sample_MultiNODEs(p, final_T=final_T, to_torch=False)
            return gen



def load_results_RTSGAN(id_mc, n_long_var=3, final_T=1.0, t_visits=200, folder='OU_with_best_config'):
    DIR_INIT = '/path/to/HyperNSDE/'
    root_dir = DIR_INIT + 'benchmark/rtsgan/monte_carlo/{}/MC_{}/'.format(folder, id_mc)
    file_path = root_dir + "synth_50x.pkl"
    if os.path.exists(file_path):
        synth_set = pickle.load(open(root_dir + "synth_50x.pkl", "rb"))
        sta_syn_df, dyn_syn_list = synth_set["synthetic"]
        N = len(sta_syn_df)
        print("N rtsgan", N)

        real_reg_grid = np.linspace(0, final_T, t_visits)
        long_values = np.zeros((N, t_visits, n_long_var))
        mask = np.zeros((N, t_visits, n_long_var))

        # 4. Snap Each Sample to the Grid
        for i, df in enumerate(dyn_syn_list):
            # Extract time and values
            if "time" in df.columns:
                t_source = df["time"].values
            elif "Time" in df.columns:
                t_source = df["Time"].values
            else:
                continue # Should not 
            
            # Extract values (all columns except the first one)
            vals = df.iloc[:, 1:].values 

            # Filter out points where time > final_T 
            valid_time_mask = t_source <= final_T
            t_source = t_source[valid_time_mask]
            vals = vals[valid_time_mask]
        
            # Find closest index in t_target for each point in t_source: searchsorted + check neighbors
            idx = np.searchsorted(real_reg_grid, t_source)
            idx = np.clip(idx, 0, t_visits-1) # Clip to avoid out of bounds

            # Check if previous index is closer
            left = idx - 1
            mask_valid = left >= 0
            if np.any(mask_valid):
                dist_curr = np.abs(t_source - real_reg_grid[idx])
                dist_left = np.abs(t_source - real_reg_grid[left])
                # If left is closer, decrement index
                # (We only change idx where left is valid AND closer)
                better_left = (dist_left < dist_curr) & mask_valid
                idx[better_left] = left[better_left]

            long_values[i, idx, :] = vals[:, :n_long_var]
            is_nan = np.isnan(vals[:, :n_long_var])
            mask[i, idx, :] = (~is_nan).astype(float)
            
        gen = {
            "Stat_Values": sta_syn_df.values[:,:-1].astype(float),
            "Long_Values": long_values.astype(float),
            "Mask": mask.astype(float),
            "Time_Grid": real_reg_grid.astype(float)
        }
        return gen 
    else:
        return None
    

def load_results_DGBFGP(id_mc, time_grid, folder='monte_carlo'):
    DIR_INIT = '/path/to/HyperNSDE/'
    root_dir = DIR_INIT + 'benchmark/DGBFGP/results/{}/MC_{}/generated_samples_more/'.format(folder, id_mc)
    file_path_long = root_dir + "gen_long.pt"
    if os.path.exists(file_path_long):
        loaded_static = torch.load(root_dir + "gen_stat.pt")
        loaded_long = torch.load(root_dir + "gen_long.pt")
        loaded_long_np = loaded_long.cpu().numpy().astype(float)
        gen = {
            "Stat_Values": loaded_static.cpu().numpy().astype(float),
            "Long_Values": loaded_long_np,
            "Mask": np.ones(loaded_long_np.shape).astype(float),
            "Time_Grid": time_grid
            }
        return gen
    else:
        return None
    
    

def load_results_OUR(id_mc, folder_res, final_T=1.0, t0_in_static=True, param_norm_long=None):
    DIR_INIT = '/path/to/HyperNSDE/'
    MODEL = 'experiments/results/monte_carlo/{}/MC_{}/'.format(folder_res, id_mc)

    SAMPLES_DIR = os.path.join(DIR_INIT, MODEL, "samples")
    folder = Path(SAMPLES_DIR)
    if folder.exists() and folder.is_dir():
        is_empty = not any(folder.iterdir())
        if is_empty:
            print(f"Warning: samples directory {SAMPLES_DIR} is empty")
            return None
        else:
            # rec_files = glob.glob(os.path.join(SAMPLES_DIR, "reconstruction_*.pth"))
            rec_files = [
                f for f in glob.glob(os.path.join(SAMPLES_DIR, "reconstruction_*.pth"))
                if "reconstruction_drift" not in os.path.basename(f)
            ]
            rec_drift_files = glob.glob(os.path.join(SAMPLES_DIR, "reconstruction_drift_*.pth"))
            post_files = glob.glob(os.path.join(SAMPLES_DIR, "posterior_*.pth"))
            prior_files = glob.glob(os.path.join(SAMPLES_DIR, "prior_*.pth"))
            paths = {
                "reconstruction": get_unique_file(rec_files, "reconstruction"),
                "reconstruction_drift": get_unique_file(rec_drift_files, "reconstruction_drift"),
                "posterior": get_unique_file(post_files, "posterior"),
                "prior": get_unique_file(prior_files, "prior"),
            }
            gen = {}
            for k,p in paths.items():
                if p is None or not os.path.exists(p):
                    print(f'Warning: file for {k} not found at {p} (skipping)')
                    gen[k] = None
                else:
                    gen[k] = load_sample(p, to_torch=False, final_T=final_T, t0_in_static=t0_in_static, param_norm_long=param_norm_long)
            return gen



DUMP_ROOT = Path("./dumps")
# DUMP_ROOT = Path("./dumps_lbdadep")
# DUMP_ROOT = Path("./dumps_Ntrain200")
# DUMP_ROOT = Path("./dumps_100pct")
# DUMP_ROOT = Path("./dumps_15pct")
# DUMP_ROOT = Path("./dumps_N250")
# DUMP_ROOT = Path("./dumps_N5000")
DUMP_ROOT.mkdir(exist_ok=True)

if __name__ == "__main__":
    n_mc = 1
    n_samples = 1000
    t_visits = 200
    n_long_var = 3
    
    for id_mc in range(n_mc):
        print(f"\n=== Dumping MC {id_mc} ===")
        out_dir = DUMP_ROOT / f"MC_{id_mc:03d}"
        out_dir.mkdir(exist_ok=True)

        # ---- Real ----
        file_dataset = 'simulated_dataset_MC_{}'.format(id_mc)
        train_dir = "/path/to/HyperNSDE/datasets/Simu_OU/monte_carlo/"
        # train_dir = "/path/to/HyperNSDE/datasets/Simu_OU/monte_carlo_lbda_dep_50pct/"
        # train_dir = "/path/to/HyperNSDE/datasets/Simu_OU/monte_carlo_N250/"
        # train_dir = "/path/to/HyperNSDE/datasets/Simu_OU/monte_carlo_N5000/"
        # train_dir = "/path/to/HyperNSDE/datasets/Simu_OU/monte_carlo_100pct/"
        # train_dir = "/path/to/HyperNSDE/datasets/Simu_OU/monte_carlo_15pct/"
        orig = load_original_data(id_mc, file_dataset, train_dir=train_dir, t_visits=t_visits, n_long_var=n_long_var, n_samples=n_samples)
        # save_real_mc(id_mc, orig, out_dir, name="real.pt")
        absmax_params_long = orig["absmax_params_long"]
        final_T = orig["T"][-1].item()
        time_grid_np = orig["T"].numpy()
        del orig

        # # # ---- MultiNODEs ----
        # # gen = load_results_multiNODEs(id_mc, final_T=final_T, folder='OU_with_best_config_v2')
        # # gen = load_results_multiNODEs(id_mc, final_T=final_T, folder='OU_with_best_config_lbdadep')
        # # gen = load_results_multiNODEs(id_mc, final_T=final_T, folder='OU_with_best_config_N250')
        # gen = load_results_multiNODEs(id_mc, final_T=final_T, folder='OU_with_best_config_Ntrain200')
        # # gen = load_results_multiNODEs(id_mc, final_T=final_T, folder='OU_with_best_config_N5000')
        # # gen = load_results_multiNODEs(id_mc, final_T=final_T, folder='OU_with_best_config_alpha1')
        # # gen = load_results_multiNODEs(id_mc, final_T=final_T, folder='OU_with_best_config_alpha0_15')
        # save_syn_mc("MultiNODEs_posterior", gen, out_dir, key="posterior")
        # save_syn_mc("MultiNODEs_prior", gen, out_dir, key="prior")
        # save_syn_mc("MultiNODEs_reconstruction", gen, out_dir, key="reconstruction")
        # del gen

        # # ---- OUR variants ----
        # # gen = load_results_OUR(id_mc, "MC_hyperNSDE_fixloss", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # gen = load_results_OUR(id_mc, "MC_AS_lbdadep", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # gen = load_results_OUR(id_mc, "MC_hyperNSDE_15pct", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # gen = load_results_OUR(id_mc, "MC_hyperNSDE_N250", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # gen = load_results_OUR(id_mc, "MC_hyperNSDE_Ntrain200", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # save_syn_mc("OUR_Hnsde_posterior", gen, out_dir, key="posterior")
        # save_syn_mc("OUR_Hnsde_prior", gen, out_dir, key="prior")
        # save_syn_mc("OUR_Hnsde_reconstruction", gen, out_dir, key="reconstruction")
        # save_syn_mc("OUR_Hnsde_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        # del gen

        # # # gen = load_results_OUR(id_mc, "MC_AS_StatMoNDEs", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # # save_syn_mc("OUR_Snsde_posterior", gen, out_dir, key="posterior")
        # # # save_syn_mc("OUR_Snsde_prior", gen, out_dir, key="prior")
        # # # save_syn_mc("OUR_Snsde_reconstruction", gen, out_dir, key="reconstruction")
        # # # save_syn_mc("OUR_Snsde_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        # # # del gen

        # # # gen = load_results_OUR(id_mc, "MC_AS_MultiNSDEs", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # # save_syn_mc("OUR_Mnsde_posterior", gen, out_dir, key="posterior")
        # # # save_syn_mc("OUR_Mnsde_prior", gen, out_dir, key="prior")
        # # # save_syn_mc("OUR_Mnsde_reconstruction", gen, out_dir, key="reconstruction")
        # # # save_syn_mc("OUR_Mnsde_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        # # # del gen

        # # gen = load_results_OUR(id_mc, "MC_AS_training_NSDEs_fixloss", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # save_syn_mc("OUR_bis_training_NSDE_posterior", gen, out_dir, key="posterior")
        # # save_syn_mc("OUR_bis_training_NSDE_prior", gen, out_dir, key="prior")
        # # save_syn_mc("OUR_bis_training_NSDE_reconstruction", gen, out_dir, key="reconstruction")
        # # save_syn_mc("OUR_bis_training_NSDE_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        # # del gen

        # # # gen = load_results_OUR(id_mc, "MC_AS_only_ODE", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # # save_syn_mc("OUR_only_ODE_posterior", gen, out_dir, key="posterior")
        # # # save_syn_mc("OUR_only_ODE_prior", gen, out_dir, key="prior")
        # # # save_syn_mc("OUR_only_ODE_reconstruction", gen, out_dir, key="reconstruction")
        # # # save_syn_mc("OUR_only_ODE_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        # # # del gen

        # # # gen = load_results_OUR(id_mc, "MC_AS_Encoder_combined", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # # save_syn_mc("OUR_Encoder_posterior", gen, out_dir, key="posterior")
        # # # save_syn_mc("OUR_Encoder_prior", gen, out_dir, key="prior")
        # # # save_syn_mc("OUR_Encoder_reconstruction", gen, out_dir, key="reconstruction")
        # # # save_syn_mc("OUR_Encoder_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        # # # del gen

        # # # gen = load_results_OUR(id_mc, "MC_AS_without_lambda_lower_lr", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # # gen = load_results_OUR(id_mc, "MC_AS_lbdadep_without_lambda_lower_lr", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        # # # save_syn_mc("OUR_without_lambda_posterior", gen, out_dir, key="posterior")
        # # # save_syn_mc("OUR_without_lambda_prior", gen, out_dir, key="prior")
        # # # save_syn_mc("OUR_without_lambda_reconstruction", gen, out_dir, key="reconstruction")
        # # # save_syn_mc("OUR_without_lambda_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        # # # del gen

        gen = load_results_OUR(id_mc, "MC_AS_training_NSDEs_regular", final_T=final_T, t0_in_static=True, param_norm_long=absmax_params_long)
        save_syn_mc("OUR reg_training_NSDE_posterior", gen, out_dir, key="posterior")
        save_syn_mc("OUR_reg_training_NSDE_prior", gen, out_dir, key="prior")
        save_syn_mc("OUR_reg_training_NSDE_reconstruction", gen, out_dir, key="reconstruction")
        save_syn_mc("OUR_reg_training_NSDE_reconstruction_drift", gen, out_dir, key="reconstruction_drift")
        del gen

        # # ---- RTSGAN ----
        # # gen = load_results_RTSGAN(id_mc, n_long_var, final_T=final_T, t_visits=t_visits, folder='OU_with_best_config_v2')
        # # gen = load_results_RTSGAN(id_mc, n_long_var, final_T=final_T, t_visits=t_visits, folder='OU_with_best_config_lbdadep')
        # # gen = load_results_RTSGAN(id_mc, n_long_var, final_T=final_T, t_visits=t_visits, folder='OU_with_best_config_N250')
        # gen = load_results_RTSGAN(id_mc, n_long_var, final_T=final_T, t_visits=t_visits, folder='OU_with_best_config_Ntrain200')
        # # gen = load_results_RTSGAN(id_mc, n_long_var, final_T=final_T, t_visits=t_visits, folder='OU_with_best_config_alpha0_15')
        # # # gen = load_results_RTSGAN(id_mc, n_long_var, final_T=final_T, t_visits=t_visits, folder='OU_with_best_config_N5000')
        # # # gen = load_results_RTSGAN(id_mc, n_long_var, final_T=final_T, t_visits=t_visits, folder='OU_with_best_config_alpha1')
        # if gen is not None:
        #     save_npz(out_dir / "RTSGAN_posterior.npz", to_canonical(gen))
        # del gen

        # # ---- DGBFGP ----
        # # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo')
        # # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo_lbda_dep_50pct_v2')
        # # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo_N250')
        # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo_Ntrain200')
        # # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo_15pct_v2')
        # # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo_N5000')
        # # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo_100pct')
        # # gen = load_results_DGBFGP(id_mc, time_grid_np, folder='monte_carlo_v2')
        # if gen is not None:
        #     save_npz(out_dir / "DGBFGP_posterior.npz", to_canonical(gen))
        # del gen

        # ---- Metadata ----
        # meta = {
        #     "mc_id": id_mc,
        #     "n_samples": n_samples,
        #     "t_visits": t_visits,
        #     "n_long_var": n_long_var,
        #     "end_T": final_T
        # }
        # json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)

        torch.cuda.empty_cache()
