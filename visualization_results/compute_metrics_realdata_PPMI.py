import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import argparse

sys.path.append('../')
from src.evaluation.general import EvaluationEngine

def expand_mask_to_last_obs(M):
    """
    M: (N, n_tps, n_vars) binary mask, 1 = observed
    Returns: (N, n_tps, n_vars) mask that is 1 up to and including
             the last observed time point per (patient, variable), 0 after.
    """
    N, n_tps, n_vars = M.shape
    M_flipped = M.flip(dims=[1])  # (N, n_tps, n_vars)
    last_obs_flipped = M_flipped.argmax(dim=1)  # (N, n_vars)
    last_obs_idx = n_tps - 1 - last_obs_flipped  # (N, n_vars)
    has_obs = M.sum(dim=1) > 0  # (N, n_vars)
    last_obs_idx = last_obs_idx * has_obs  # set to 0 if no obs (will be masked out)
    time_idx = torch.arange(n_tps, device=M.device).view(1, n_tps, 1)
    last_obs_idx_exp = last_obs_idx.unsqueeze(1)  # (N, 1, n_vars)
    expanded = (time_idx <= last_obs_idx_exp).float()  # (N, n_tps, n_vars)
    expanded = expanded * has_obs.unsqueeze(1)
    return expanded

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

def load_syn_mc_npz(path, n_gen=50, n_per_gen=200, scales=None, method_scaling="standard"):
    with np.load(path) as data:
        out = {k: torch.from_numpy(data[k]).to(torch.float32) for k in data.files}
    
    nan_mask = torch.isnan(out["X"])
    out["M"][nan_mask] = 0.0
    out["X"] = torch.nan_to_num(out["X"], nan=0.0)

    if scales is not None:
        out["X"] = apply_scaling(out.get("X"), scales["X"], method=method_scaling)
        out["W"] = apply_scaling(out.get("W"), scales["W"], method=method_scaling)
    out["X"][nan_mask] = 0.0

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


def load_syn_one_gen(path, gen_idx, n_per_gen, scales, method_scaling="standard"):
    """Load the full NPZ but return only the slice for `gen_idx`."""
    with np.load(path) as data:
        out = {k: torch.from_numpy(data[k]).to(torch.float32) for k in data.files}
 
    nan_mask = torch.isnan(out["X"])
    out["M"][nan_mask] = 0.0
    out["X"] = torch.nan_to_num(out["X"], nan=0.0)
 
    # Reshape to (n_gen, n_per_gen, ...)
    total = out["X"].shape[0]
    if len(out["X"].shape) > 3:
        # Already (n_gen, n_per_gen, ...)
        X_g = out["X"]
        M_g = out["M"]
    else:
        n_gen_inferred = total // n_per_gen
        X_g = out["X"].view(n_gen_inferred, n_per_gen, *out["X"].shape[1:])
        M_g = out["M"].view(n_gen_inferred, n_per_gen, *out["M"].shape[1:])
 
    W = out.get("W")
    if W is not None:
        W_g = W if len(W.shape) > 2 else W.view(X_g.shape[0], n_per_gen, *W.shape[1:])
    else:
        W_g = None
 
    # ── Slice the single generation we want ──
    # Keep shape (1, n_per_gen, ...) so evaluate_several_synthetic_sets still works
    X_slice = X_g[gen_idx : gen_idx + 1]
    M_slice = M_g[gen_idx : gen_idx + 1]
    W_slice = W_g[gen_idx : gen_idx + 1] if W_g is not None else None
 
    # Apply scaling
    if scales is not None:
        X_slice = apply_scaling(X_slice, scales["X"], method=method_scaling)
        if W_slice is not None:
            W_slice = apply_scaling(W_slice, scales["W"], method=method_scaling)
 
    # Re-zero masked positions after scaling
    X_slice[torch.isnan(X_slice)] = 0.0
    nan_slice = nan_mask.view(X_g.shape[0], n_per_gen, *nan_mask.shape[1:])[gen_idx : gen_idx + 1]
    X_slice[nan_slice] = 0.0

    W_slice = torch.nan_to_num(W_slice, nan=0.0)
 
    return {
        "X": X_slice,
        "M": M_slice,
        "W": W_slice,
        "T": out.get("T", None),
    }


def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def build_real_sets(real, method_scaling="standard"):
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

    scales = {
        "X": get_scaling_params(real_all.get("X"), method=method_scaling),
        "W": get_scaling_params(real_all.get("W"), method=method_scaling)
    }
    # Apply to Real
    real_all["X"] = apply_scaling(real_all.get("X"), scales["X"], method=method_scaling)
    real_all["X"][~real_all["M"].bool()] = 0.0
    real_all["W"] = apply_scaling(real_all.get("W"), scales["W"], method=method_scaling)

    real_train["X"] = apply_scaling(real_train.get("X"), scales["X"], method=method_scaling)
    real_train["X"][~real_train["M"].bool()] = 0.0
    real_train["W"] = apply_scaling(real_train.get("W"), scales["W"], method=method_scaling)

    real_test["X"] = apply_scaling(real_test.get("X"), scales["X"], method=method_scaling)
    real_test["X"][~real_test["M"].bool()] = 0.0
    real_test["W"] = apply_scaling(real_test.get("W"), scales["W"], method=method_scaling)

    return real_all, real_train, real_test, scales


if __name__ == "__main__":
    dataset_name = "PPMI"

    # 1. Argument Parsing for Parallelization
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_idx", type=int, required=True, help="Which synthetic generation to evaluate (0-indexed).")
    parser.add_argument("--out_dir", type=str, default="evaluation_results", help="Folder to save CSVs")
    parser.add_argument("--dump_root", type=str, default="./dumps_{}".format(dataset_name), help="Root folder of the NPZ dumps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gen_idx_block = args.gen_idx
    gen_idx = args.gen_idx

    # Setup Paths
    DUMP_ROOT = Path(args.dump_root)
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    FIDELITY_METRICS = ["longi.MMD"] #["longi.MMD", "static.MMD", "longi.pairwise_corr", "static.pairwise_corr", "global_pairwise_corr", "longi.KL_obs_rate", "discriminative_score"]
    UTILITY_METRICS = [] # ["prediction_score"]
    PRIVACY_METRICS = [] #["NNAA", "MIR"] # ["NNAA", "MIR", "dependence_score"]
    
    dir = DUMP_ROOT 
    if not dir.exists():
        print(f"Error: Directory {dir} does not exist.")
        sys.exit(1)

    # 2. Configuration
    metric_config = {
        "longi.MMD": {"kernel": "sig rbf", "sigmas": [100, 400], "dyadic_order":1, "lead_lag":False, "subtrack_init_point":False, "scale_time":True},
        "static.MMD": {"kernel": "rbf"},
        "longi.pairwise_corr": {},
        "static.pairwise_corr": {},
        "global_pairwise_corr": {}, 
        "longi.KL_obs_rate": {"eps": 1e-8}, 
        "discriminative_score": {"classifier_type": 'S4', "filling_method": 'last', "verbose": True},
        "prediction_score": {"predictor_type": "S4", "filling_type": 'last', "verbose": False, "input_len": 250}, # "future_steps": 5,
        "NNAA": {"weights": (0.5, 0.5), "dist_type": "signature", "kernel": "rbf", "dyadic_order":1, "sigma":0.3, "lead_lag":False}, 
        "MIR": {"weights": (0.5, 0.5), "k": 5, "dist_type": "signature", "kernel": "rbf", "dyadic_order":1, "sigma":0.3, "lead_lag":False},
        "dependence_score": {"max_lag": None, "square": False}
    }
    
    # for i in range(gen_idx_block*5, (gen_idx_block+1)*5):
    #     if i < gen_idx_block*5 + 4:
    #         continue
    #     else:
    #         gen_idx = i
            # n_gen_syn = 50
    rows = []

    print(f"\n=== Starting Evaluation {dataset_name} ===")
    print(f"[gen {gen_idx}] Loading real data …")
    try:    
        real = load_real_mc(dir)
        real_all, real_train, real_test, scales = build_real_sets(real)
    except FileNotFoundError:
        print(f"Skipping: Real file not found in {dir}.")
        sys.exit(1)

    n_per_gen = real_test["X"].shape[0]

    eval_engine = EvaluationEngine(
        real_all=real_all,
        real_train=real_train,
        real_test=real_test,
        device=device,
        static_types=real["static_types"],
        metric_config=metric_config,
        fidelity_metrics = FIDELITY_METRICS, 
        utility_metrics = UTILITY_METRICS, 
        privacy_metrics = PRIVACY_METRICS
    )


    # MAIN
    syn_files = sorted(
        list(dir.glob("MultiNODEs_posterior.npz"))  
        + list(dir.glob("RTSGAN_posterior.npz"))  
        + list(dir.glob("OUR_Hnsde_posterior.npz"))
        # + list(dir.glob("OUR_Hnsde_prior.npz"))
    )
    syn_files = [f for f in syn_files]
    for syn_file in syn_files:
        model_name = syn_file.stem #.replace("_posterior", "")
        print(f"  > Model: {model_name}")
        if model_name == "MultiNODEs":
            multiNODEs_mask = expand_mask_to_last_obs(real_test["M"])
        else:
            multiNODEs_mask = None

        t_init = time.time()
        # n_gen = n_gen_syn
        # syn = load_syn_mc_npz(syn_file, n_gen=n_gen, n_per_gen=n_per_gen, scales=scales, method_scaling="standard")

        syn = load_syn_one_gen(
            syn_file,
            gen_idx=gen_idx,
            n_per_gen=n_per_gen,
            scales=scales,
            method_scaling="standard",
        )

        df = eval_engine.evaluate_several_synthetic_sets(
            syn,
            n_gen=1,
            fidelity_metrics=FIDELITY_METRICS,
            utility_metrics=UTILITY_METRICS,
            privacy_metrics=PRIVACY_METRICS,
            max_sets_training=-1, # -1 for all generations
            multinodes_mask=multiNODEs_mask,
            metrics_adapt_multinodes=["longi.MMD", "longi.pairwise_corr", "global_pairwise_corr", "prediction_score"]
        )
        df["model"] = model_name
        df["gen_idx"] = gen_idx
        rows.append(df)  
        print(f"    Done in {time.time() - t_init:.1f}s") 

    if gen_idx == 0:
        print(" > Baseline")
        df = eval_engine.compute_baseline(
            fidelity_metrics=FIDELITY_METRICS,
            utility_metrics=UTILITY_METRICS,
            privacy_metrics=PRIVACY_METRICS,
            seed=0
        )
        df["model"] = 'Baseline'
        df["gen_idx"] = -1
        rows.append(df)  

    # 6. Save results for THIS MC id
    if rows:
        final_df = pd.concat(rows, ignore_index=True)
        out_path = RESULTS_DIR / f"metrics_{dataset_name}_gen{gen_idx:03d}.csv"
        final_df.to_csv(out_path, index=False)
        print(f"=== Results saved to {out_path} ===\n")