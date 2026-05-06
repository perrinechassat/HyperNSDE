import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import argparse

sys.path.append('../')
from src.evaluation.general import EvaluationEngine

def get_scaling_params(tensor, method="standard"):
    if tensor is None: return None
    # Flatten to (Total_Samples * Time, Features) for X 
    # or (Total_Samples, Features) for W
    flat = tensor.reshape(-1, tensor.shape[-1])
    
    if method == "maxabs":
        scale = torch.max(torch.abs(flat), dim=0)[0]
        scale[scale == 0] = 1.0
        return {"scale": scale}
    
    elif method == "standard":
        return {
            "mean": torch.mean(flat, dim=0),
            "std": torch.std(flat, dim=0) + 1e-8
        }

def apply_scaling(tensor, params, method="standard"):
    if tensor is None or params is None: return tensor
    if method == "maxabs":
        return tensor / params["scale"]
    elif method == "standard":
        return (tensor - params["mean"]) / params["std"]
    
def load_real_mc(path):
    return torch.load(path / "real.pt")

def load_syn_mc_npz(path, n_gen=50, n_per_gen=200):
    with np.load(path) as data:
        # Optimization: convert to float32 immediately
        out = {k: torch.from_numpy(data[k]).to(torch.float32) for k in data.files}
    
    # nan_mask = torch.isnan(out["X"])
    # out["M"][nan_mask] = 0.0
    # out["X"] = torch.nan_to_num(out["X"], nan=0.0)
    
    total = out["X"].shape[0]
    if len(out["X"].shape) > 3:
        assert out["X"].shape[0] == n_gen and out["X"].shape[1] == n_per_gen, (
            f"Expected {n_gen}×{n_per_gen}={n_gen*n_per_gen} samples, "
        )
        X_g = out["X"]
        M_g = out["M"]
    else:
        assert total == n_gen * n_per_gen, (
            f"Expected {n_gen}×{n_per_gen}={n_gen*n_per_gen} samples, "
            f"got {total}"
        )
        X_g = out["X"].view(n_gen, n_per_gen, *out["X"].shape[1:])
        M_g = out["M"].view(n_gen, n_per_gen, *out["M"].shape[1:])
    if out.get("W") is not None:
        if len(out["W"].shape) > 2:
            W_g = out["W"]
        else:
            W_g = out["W"].view(n_gen, n_per_gen, *out["W"].shape[1:])
    else:
        W_g = None

    return {
        "X": X_g,
        "M": M_g,
        "W": W_g,
        "T": out.get("T", None), 
    }

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def build_real_sets(real):
    real_train = {
        "X": to_tensor(real["train"]["X"]).float(),
        "M": to_tensor(real["train"]["M"]).float(),
        "W": to_tensor(real["train"]["W"]).float(),
        "T": to_tensor(real["T"]).float(),
    }
    real_test = {
        "X": to_tensor(real["test"]["X"]).float(),
        "M": to_tensor(real["test"]["M"]).float(),
        "W": to_tensor(real["test"]["W"]).float(),
        "T": to_tensor(real["T"]).float(),
    }
    real_all = {
        "X": torch.cat([real_train["X"], real_test["X"]], dim=0),
        "M": torch.cat([real_train["M"], real_test["M"]], dim=0),
        "W": torch.cat([real_train["W"], real_test["W"]], dim=0),
        "T": to_tensor(real["T"]).float(),
    }
    return real_all, real_train, real_test

def split_groups_dataset(real, extend_dim=False):
    idx_0 = torch.where(real["W"][:, 1] == 0)[0]
    idx_1 = torch.where(real["W"][:, 1] == 1)[0]
    if extend_dim:
        real_0 = {
        "X": real["X"][idx_0].unsqueeze(0),
        "M": real["M"][idx_0].unsqueeze(0),
        "W": real["W"][idx_0].unsqueeze(0),
        "T": real["T"]
        }
        real_1 = {
            "X": real["X"][idx_1].unsqueeze(0),
            "M": real["M"][idx_1].unsqueeze(0),
            "W": real["W"][idx_1].unsqueeze(0),
            "T": real["T"]
        }
    else:
        real_0 = {
            "X": real["X"][idx_0],
            "M": real["M"][idx_0],
            "W": real["W"][idx_0],
            "T": real["T"]
        }
        real_1 = {
            "X": real["X"][idx_1],
            "M": real["M"][idx_1],
            "W": real["W"][idx_1],
            "T": real["T"]
        }
    return real_0, real_1

if __name__ == "__main__":
    # 1. Argument Parsing for Parallelization
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_id", type=int, help="The Monte Carlo ID to evaluate")
    parser.add_argument("--out_dir", type=str, default="evaluation_results", help="Folder to save CSVs")
    parser.add_argument("--dump_root", type=str, default="./dumps", help="Root folder of the NPZ dumps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Paths
    DUMP_ROOT = Path(args.dump_root)
    RESULTS_DIR_0 = Path(args.out_dir + "_0")
    RESULTS_DIR_1 = Path(args.out_dir + "_1")
    RESULTS_DIR_0.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR_1.mkdir(exist_ok=True, parents=True)

    FIDELITY_METRICS = ["longi.MMD", "static.MMD", "longi.pairwise_corr", "static.pairwise_corr", "global_pairwise_corr", "longi.KL_obs_rate", "discriminative_score"]
    # FIDELITY_METRICS = ["longi.MMD", "longi.pairwise_corr", "global_pairwise_corr"]
    UTILITY_METRICS = ["prediction_score"]
    PRIVACY_METRICS = ["NNAA", "MIR", "dependence_score"]

    mc_dir = DUMP_ROOT / f"MC_{args.mc_id:03d}"
    if not mc_dir.exists():
        print(f"Error: Directory {mc_dir} does not exist.")
        sys.exit(1)

    # 2. Configuration
    metric_config = {
        "longi.MMD": {"kernel": "sig rbf", "sigmas": 0.3,"dyadic_order":1, "lead_lag":False},
        "static.MMD": {"kernel": "rbf"},
        "longi.pairwise_corr": {},
        "static.pairwise_corr": {},
        "global_pairwise_corr": {}, 
        "longi.KL_obs_rate": {"eps": 1e-8}, 
        "discriminative_score": {"classifier_type": 'S4', "filling_method": 'last', "verbose": True},
        "prediction_score": {"predictor_type": "S4", "filling_type": 'last', "future_steps": 5, "verbose": False},
        "NNAA": {"weights": (0.5, 0.5), "dist_type": "signature", "kernel": "rbf", "dyadic_order":1, "sigma":0.3, "lead_lag":False}, 
        "MIR": {"weights": (0.5, 0.5), "k": 5, "dist_type": "signature", "kernel": "rbf", "dyadic_order":1, "sigma":0.3, "lead_lag":False},
        "dependence_score": {"max_lag": None, "square": False}
    }
    
    n_gen_syn = 50
    n_times = 200
    rows_0 = []
    rows_1 = []

    print(f"\n=== Starting Evaluation: MC {args.mc_id} ===")

    # 3. Load Real Data
    try:    
        real = load_real_mc(mc_dir)
        real_all, real_train, real_test = build_real_sets(real)
    except FileNotFoundError:
        print(f"Skipping: Real file not found in {mc_dir}.")
        sys.exit(1)

    # Split group 0 and group 1
    real_all_0, real_all_1 = split_groups_dataset(real_all)
    real_train_0, real_train_1 = split_groups_dataset(real_train)
    real_test_0, real_test_1 = split_groups_dataset(real_test)

    scales_0 = {
        "X": get_scaling_params(real_all_0.get("X"), method="standard"),
        "W": get_scaling_params(real_all_0.get("W"), method="standard")
    }
    real_all_0["X"] = apply_scaling(real_all_0.get("X"), scales_0["X"], method="standard")
    real_all_0["X"][~real_all_0["M"].bool()] = 0.0
    real_all_0["W"] = apply_scaling(real_all_0.get("W"), scales_0["W"], method="standard")
    real_train_0["X"] = apply_scaling(real_train_0.get("X"), scales_0["X"], method="standard")
    real_train_0["X"][~real_train_0["M"].bool()] = 0.0
    real_train_0["W"] = apply_scaling(real_train_0.get("W"), scales_0["W"], method="standard")
    real_test_0["X"] = apply_scaling(real_test_0.get("X"), scales_0["X"], method="standard")
    real_test_0["X"][~real_test_0["M"].bool()] = 0.0
    real_test_0["W"] = apply_scaling(real_test_0.get("W"), scales_0["W"], method="standard")

    scales_1 = {
        "X": get_scaling_params(real_all_1.get("X"), method="standard"),
        "W": get_scaling_params(real_all_1.get("W"), method="standard")
    }
    real_all_1["X"] = apply_scaling(real_all_1.get("X"), scales_1["X"], method="standard")
    real_all_1["X"][~real_all_1["M"].bool()] = 0.0
    real_all_1["W"] = apply_scaling(real_all_1.get("W"), scales_1["W"], method="standard")
    real_train_1["X"] = apply_scaling(real_train_1.get("X"), scales_1["X"], method="standard")
    real_train_1["X"][~real_train_1["M"].bool()] = 0.0
    real_train_1["W"] = apply_scaling(real_train_1.get("W"), scales_1["W"], method="standard")
    real_test_1["X"] = apply_scaling(real_test_1.get("X"), scales_1["X"], method="standard")
    real_test_1["X"][~real_test_1["M"].bool()] = 0.0
    real_test_1["W"] = apply_scaling(real_test_1.get("W"), scales_1["W"], method="standard")
    
    syn_files = sorted(
        # list(mc_dir.glob("MultiNODEs_posterior.npz"))  
        # + list(mc_dir.glob("RTSGAN_posterior.npz"))  +
         list(mc_dir.glob("DGBFGP_posterior.npz"))  
        # + list(mc_dir.glob("OUR_Hnsde_prior.npz"))  
        # + list(mc_dir.glob("OUR_Hnsde_posterior.npz"))
        # + list(mc_dir.glob("OUR_Mnsde_posterior.npz"))
        # + list(mc_dir.glob("OUR_Snsde_posterior.npz"))
        # # + list(mc_dir.glob("*_prior.npz")) 
        # # + list(mc_dir.glob("RTSGAN_posterior.npz")) 
    )
    syn_files = [f for f in syn_files]


    # 4. Initialize Engine
    eval_engine_0 = EvaluationEngine(
        real_all=real_all_0,
        real_train=real_train_0,
        real_test=real_test_0,
        device=device,
        static_types=real["static_types"],
        metric_config=metric_config,
        fidelity_metrics=FIDELITY_METRICS,
        utility_metrics=UTILITY_METRICS,
        privacy_metrics=PRIVACY_METRICS,
    )

    # 5. Iterate Models (filtering for posterior)
    # Combine both patterns and then sort the final list
    for syn_file in syn_files:
        model_name = syn_file.stem #.replace("_posterior", "")
        print(f"  > Model: {model_name}")

        n_gen = n_gen_syn
        syn = load_syn_mc_npz(syn_file, n_gen=n_gen, n_per_gen=n_times)

        for g in range(n_gen):
            syn_g = {
                "X": syn["X"][g],
                "M": syn["M"][g],
                "W": syn["W"][g],
                "T": syn["T"],
            }
            
            syn_0, syn_1 = split_groups_dataset(syn_g, extend_dim=True)
            
            syn_0["X"] = apply_scaling(syn_0.get("X"), scales_0["X"], method="standard")
            syn_0["W"] = apply_scaling(syn_0.get("W"), scales_0["W"], method="standard")
            nan_mask = torch.isnan(syn_0["X"])
            syn_0["M"][nan_mask] = 0.0
            syn_0["X"] = torch.nan_to_num(syn_0["X"], nan=0.0)

            syn_1["X"] = apply_scaling(syn_1.get("X"), scales_1["X"], method="standard")
            syn_1["W"] = apply_scaling(syn_1.get("W"), scales_1["W"], method="standard")
            nan_mask = torch.isnan(syn_1["X"])
            syn_1["M"][nan_mask] = 0.0
            syn_1["X"] = torch.nan_to_num(syn_1["X"], nan=0.0)
            
            n_0 = syn_0["X"].shape[0]
            n_1 = syn_1["X"].shape[0]

            t_init = time.time()
            if n_0 > 0:
                df_0 = eval_engine_0.evaluate_several_synthetic_sets(
                    syn_0,
                    n_gen=1,
                    fidelity_metrics=FIDELITY_METRICS,
                    utility_metrics=UTILITY_METRICS,
                    privacy_metrics=PRIVACY_METRICS,
                    max_sets_training=-1 # -1 for all generations
                )
                df_0["model"] = model_name
                df_0["mc_id"] = args.mc_id
                df_0["generation_id"] = g
                rows_0.append(df_0) 

        print(f"    Done in {time.time() - t_init:.1f}s") 

    print(" > Baseline")
    df_0 = eval_engine_0.compute_baseline(
        fidelity_metrics=FIDELITY_METRICS,
        utility_metrics=UTILITY_METRICS,
        privacy_metrics=PRIVACY_METRICS,
        seed=args.mc_id
    )
    df_0["model"] = 'Baseline'
    df_0["mc_id"] = args.mc_id
    rows_0.append(df_0) 

    # 6. Save results for THIS MC id
    if rows_0:
        final_df_0 = pd.concat(rows_0, ignore_index=True)
        out_path = RESULTS_DIR_0 / f"metrics_MC_{args.mc_id:03d}.csv"
        final_df_0.to_csv(out_path, index=False)
        print(f"=== Results group 0 saved to {out_path} ===\n")

    del eval_engine_0


    eval_engine_1 = EvaluationEngine(
        real_all=real_all_1,
        real_train=real_train_1,
        real_test=real_test_1,
        device=device,
        static_types=real["static_types"],
        metric_config=metric_config,
        fidelity_metrics=FIDELITY_METRICS,
        utility_metrics=UTILITY_METRICS,
        privacy_metrics=PRIVACY_METRICS,
    )

    # 5. Iterate Models (filtering for posterior)
    # Combine both patterns and then sort the final list
    for syn_file in syn_files:
        model_name = syn_file.stem #.replace("_posterior", "")
        print(f"  > Model: {model_name}")

        n_gen = n_gen_syn
        syn = load_syn_mc_npz(syn_file, n_gen=n_gen, n_per_gen=n_times)

        for g in range(n_gen):
            syn_g = {
                "X": syn["X"][g],
                "M": syn["M"][g],
                "W": syn["W"][g],
                "T": syn["T"],
            }
            
            syn_0, syn_1 = split_groups_dataset(syn_g, extend_dim=True)

            syn_0["X"] = apply_scaling(syn_0.get("X"), scales_0["X"], method="standard")
            syn_0["W"] = apply_scaling(syn_0.get("W"), scales_0["W"], method="standard")
            nan_mask = torch.isnan(syn_0["X"])
            syn_0["M"][nan_mask] = 0.0
            syn_0["X"] = torch.nan_to_num(syn_0["X"], nan=0.0)

            syn_1["X"] = apply_scaling(syn_1.get("X"), scales_1["X"], method="standard")
            syn_1["W"] = apply_scaling(syn_1.get("W"), scales_1["W"], method="standard")
            nan_mask = torch.isnan(syn_1["X"])
            syn_1["M"][nan_mask] = 0.0
            syn_1["X"] = torch.nan_to_num(syn_1["X"], nan=0.0)

            n_0 = syn_0["X"].shape[0]
            n_1 = syn_1["X"].shape[0]

            t_init = time.time()
            if n_1 > 0:
                df_1 = eval_engine_1.evaluate_several_synthetic_sets(
                    syn_1,
                    n_gen=1,
                    fidelity_metrics=FIDELITY_METRICS,
                    utility_metrics=UTILITY_METRICS,
                    privacy_metrics=PRIVACY_METRICS,
                    max_sets_training=-1 # -1 for all generations
                )
                df_1["model"] = model_name
                df_1["mc_id"] = args.mc_id
                df_1["generation_id"] = g
                rows_1.append(df_1)  

        print(f"    Done in {time.time() - t_init:.1f}s") 

    print(" > Baseline")
    df_1 = eval_engine_1.compute_baseline(
        fidelity_metrics=FIDELITY_METRICS,
        utility_metrics=UTILITY_METRICS,
        privacy_metrics=PRIVACY_METRICS,
        seed=args.mc_id
    )
    df_1["model"] = 'Baseline'
    df_1["mc_id"] = args.mc_id
    rows_1.append(df_1)  

    if rows_1:
        final_df_1 = pd.concat(rows_1, ignore_index=True)
        out_path = RESULTS_DIR_1 / f"metrics_MC_{args.mc_id:03d}.csv"
        final_df_1.to_csv(out_path, index=False)
        print(f"=== Results group 1 saved to {out_path} ===\n")