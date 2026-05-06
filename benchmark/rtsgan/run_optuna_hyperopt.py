# train_rtsgan.py
import pickle
import time
import numpy as np
import torch
import copy
import optuna
from optuna.logging import set_verbosity, WARNING, INFO
import argparse
import os
import json
import random

# absolute path to RTSGAN repo
# RTSGAN_PATH = "/path/to/Documents/RTSGAN"
RTSGAN_PATH = "/path/to/RTSGAN"
import sys
sys.path.append(RTSGAN_PATH)
sys.path.append(RTSGAN_PATH + "/general")
from general.missingprocessor import *
sys.path.append(RTSGAN_PATH + "/utils")
from utils.general import init_logger, make_sure_path_exists
from physionet2012 import Physio2012
from fastNLP import DataSetIter, SequentialSampler, RandomSampler


def compute_val_loss(syn, val_set, device):
    syn.ae.eval()
    losses = []

    batch = DataSetIter(val_set, batch_size=syn.params["ae_batch_size"], sampler=SequentialSampler())

    with torch.no_grad():
        for batch_x, _ in batch:
            sta = batch_x["sta"].to(device)
            dyn = batch_x["dyn"].to(device)
            mask = batch_x["mask"].to(device)
            lag = batch_x["lag"].to(device)
            priv = batch_x["priv"].to(device)
            nex = batch_x["nex"].to(device)
            times = batch_x["times"].to(device)
            seq_len = batch_x["seq_len"].to(device)

            out_sta, out_dyn, missing, gt = syn.ae(
                sta, dyn, lag, mask, priv, nex, times, seq_len, forcing=1
            )

            L1 = syn.sta_loss(out_sta, sta)
            L2 = syn.dyn_loss(out_dyn, dyn, seq_len, mask)
            # L3 = syn.missing_loss(missing, mask, seq_len)
            L4 = syn.time_loss(gt, times, seq_len)

            # total = float(L1 + L2 + L3 + L4)
            total = float(L1 + L2 + L4)
            losses.append(total)

    return np.mean(losses)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="./rtsgan_hyperopt")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--optuna_study", type=str, default="rtsgan_hyperopt_study")
    parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
    parser.add_argument("--devi", default="0", dest="devi", help="gpu")
    parser.add_argument("--init_path", type=str, default="/path/to/HyperNSDE/benchmark/rtsgan/")
    args = parser.parse_args()

    # --- Default RTSGAN options ---
    base_params = {
        "dataset": args.dataset,
        "log_dir": args.log_dir,
        "optuna_study": args.optuna_study,
        "devi": args.devi,

        # AE training settings
        "epochs": 600,                
        "ae_batch_size": 64,
        "gan_batch_size": 64,

        # GAN settings (kept, but GAN not trained)
        "gan_lr": 1e-4,
        "gan_alpha": 0.99,
        "d_update": 5,
        "iterations": 3000,    # reduce WGAN steps for hyperopt

        # Unused in hyperopt stage
        "fix_ae": None,
        "fix_gan": None,
        "force": "",
        "debug": False,
        "layers": 3,
    }

    def objective(trial):

        # --- Clone params so each trial is independent ---
        params = copy.deepcopy(base_params)

        # ===========================
        # Hyperparameter search space
        # ===========================
        params["ae_lr"] = trial.suggest_float("ae_lr", 1e-4, 1e-3, log=True)
        # params["ae_batch_size"] = trial.suggest_categorical("ae_batch_size", [64, 128])
        params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128])
        params["embed_dim"] = trial.suggest_categorical("embed_dim", [256, 512])
        # params["layers"] = trial.suggest_categorical("layers", [3])
        params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)

        # Keep noise_dim for consistency (GAN later)
        params["noise_dim"] = trial.suggest_categorical("noise_dim", [128, 256, 512])

        # Set deterministic seeds for this Optuna trial
        seed = trial.number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ===========================
        # Load dataset
        # ===========================
        dataset = pickle.load(open(params["dataset"], "rb"))
        train_set = dataset["train_set"]
        val_set   = dataset.get("val_set", None)

        if val_set is None:
            raise RuntimeError("Dataset must contain val_set for RTSGAN hyperopt.")

        # assign required inputs
        train_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len", "priv", "nex", "label")
        val_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len", "priv", "nex")

        static_processor  = dataset["static_processor"]
        dynamic_processor = dataset["dynamic_processor"]

        # =============================
        # Logging directory (per trial)
        # =============================
        task_name = params["optuna_study"]
        root_dir = "{}/{}".format(params["log_dir"], task_name+f"/trial_{trial.number}")
        make_sure_path_exists(root_dir)
        logger = init_logger(root_dir)


        if torch.cuda.is_available():
            devices = [int(x) for x in params["devi"]]
            device = torch.device(f"cuda:{devices[0]}")
        else:
            device = torch.device("cpu")
        
        params["static_processor"] = static_processor
        params["dynamic_processor"] = dynamic_processor
        params["logger"] = logger
        params["root_dir"] = root_dir
        params["device"] = device

        # ===========================
        # Build RTSGAN (AE-only for hyperopt)
        # ===========================
        syn = Physio2012((static_processor, dynamic_processor), params)

        # ===========================
        # Train the autoencoder only
        # ===========================
        syn.train_ae(train_set, params["epochs"])

        # Early stopping / pruning
        trial.report(0.0, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # ===========================
        # Compute validation loss
        # ===========================
        val_loss = compute_val_loss(syn, val_set, device)
        print("\n" + "="*50)
        print(f"Trial {trial.number} validation loss: {val_loss}")

        return val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    # Optuna study setup
    save_path = os.path.join(args.log_dir, args.optuna_study)
    os.makedirs(save_path, exist_ok=True)

    db_file = args.optuna_study + ".db"
    db_file = os.path.join(save_path, db_file)
    study_name = os.path.join(save_path, args.optuna_study)
    print("Optuna study will be saved to: {} with the name {}".format(db_file, study_name))
    set_verbosity(INFO)
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+db_file)
    else:
        print("This optuna study ({}) does not exist. We create a new study.".format(db_file))
        study = optuna.create_study(direction='minimize', study_name=study_name, storage='sqlite:///'+db_file)
        
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1, show_progress_bar=True)  

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
    trials_df.to_csv(os.path.join(save_path, 'study_results.csv'), index=False)

    with open(os.path.join(save_path, 'best_params.json'), "w") as f:
        json.dump(study.best_params, f)

