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

def load_syn_mc_npz(path, n_gen=50, n_per_gen=200, scales=None):
    with np.load(path) as data:
        out = {k: torch.from_numpy(data[k]).to(torch.float32) for k in data.files}
    
    nan_mask = torch.isnan(out["X"])
    out["M"][nan_mask] = 0.0
    out["X"] = torch.nan_to_num(out["X"], nan=0.0)

    if scales is not None:
        out["X"] = apply_scaling(out.get("X"), scales["X"], method="standard")
        out["W"] = apply_scaling(out.get("W"), scales["W"], method="standard")
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

    scales = {
        "X": get_scaling_params(real_all.get("X"), method="standard"),
        "W": get_scaling_params(real_all.get("W"), method="standard")
    }
    # Apply to Real
    real_all["X"] = apply_scaling(real_all.get("X"), scales["X"], method="standard")
    real_all["X"][~real_all["M"].bool()] = 0.0
    real_all["W"] = apply_scaling(real_all.get("W"), scales["W"], method="standard")

    real_train["X"] = apply_scaling(real_train.get("X"), scales["X"], method="standard")
    real_train["X"][~real_train["M"].bool()] = 0.0
    real_train["W"] = apply_scaling(real_train.get("W"), scales["W"], method="standard")

    real_test["X"] = apply_scaling(real_test.get("X"), scales["X"], method="standard")
    real_test["X"][~real_test["M"].bool()] = 0.0
    real_test["W"] = apply_scaling(real_test.get("W"), scales["W"], method="standard")

    return real_all, real_train, real_test, scales


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
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    FIDELITY_METRICS = ["longi.MMD", "static.MMD", "longi.pairwise_corr", "static.pairwise_corr", "global_pairwise_corr", "longi.KL_obs_rate", "discriminative_score"]
    UTILITY_METRICS = ["prediction_score"]
    PRIVACY_METRICS = ["NNAA", "MIR", "dependence_score"]
    
    mc_dir = DUMP_ROOT / f"MC_{args.mc_id:03d}"
    if not mc_dir.exists():
        print(f"Error: Directory {mc_dir} does not exist.")
        sys.exit(1)

    # 2. Configuration
    metric_config = {
        "longi.MMD": {"kernel": "sig rbf", "sigmas": 0.3, "dyadic_order":1, "lead_lag":False},
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
    rows = []

    print(f"\n=== Starting Evaluation: MC {args.mc_id} ===")

    try:    
        real = load_real_mc(mc_dir)
        real_all, real_train, real_test, scales = build_real_sets(real)
    except FileNotFoundError:
        print(f"Skipping: Real file not found in {mc_dir}.")
        sys.exit(1)

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

    ## LBDA_DEP
    # syn_files = sorted(
    #     list(mc_dir.glob("MultiNODEs_posterior.npz"))
    #     + list(mc_dir.glob("RTSGAN_posterior.npz"))
    #     + list(mc_dir.glob("DGBFGP_posterior.npz"))
    #     + list(mc_dir.glob("OUR_without_lambda_posterior.npz"))
    #     + list(mc_dir.glob("OUR_Hnsde_posterior.npz"))
    # )

    ## SENSITIVITY
    syn_files = sorted(
        list(mc_dir.glob("MultiNODEs_posterior.npz"))  
        + list(mc_dir.glob("RTSGAN_posterior.npz"))  
        + list(mc_dir.glob("DGBFGP_posterior.npz"))  
        + list(mc_dir.glob("OUR_Hnsde_posterior.npz"))
        + list(mc_dir.glob("OUR_Hnsde_prior.npz"))
    )

    ## MAIN
    # syn_files = sorted(
    #     list(mc_dir.glob("*_posterior.npz")) + 
    #     list(mc_dir.glob("OUR_Hnsde_prior.npz")) 
    # )
    syn_files = [f for f in syn_files 
                 if "Encoder" not in f.name]
    for syn_file in syn_files:
        model_name = syn_file.stem #.replace("_posterior", "")
        print(f"  > Model: {model_name}")

        t_init = time.time()
        n_gen = n_gen_syn
        syn = load_syn_mc_npz(syn_file, n_gen=n_gen, n_per_gen=n_times, scales=scales)
        df = eval_engine.evaluate_several_synthetic_sets(
            syn,
            n_gen=n_gen,
            fidelity_metrics=FIDELITY_METRICS,
            utility_metrics=UTILITY_METRICS,
            privacy_metrics=PRIVACY_METRICS,
            max_sets_training=-1 # -1 for all generations
        )
        df["model"] = model_name
        df["mc_id"] = args.mc_id
        rows.append(df)  
        print(f"    Done in {time.time() - t_init:.1f}s") 

    
    # print(" > Baseline")
    # df = eval_engine.compute_baseline(
    #     fidelity_metrics=FIDELITY_METRICS,
    #     utility_metrics=UTILITY_METRICS,
    #     privacy_metrics=PRIVACY_METRICS,
    #     seed=args.mc_id
    # )
    # df["model"] = 'Baseline'
    df["mc_id"] = args.mc_id
    rows.append(df)  

    # 6. Save results for THIS MC id
    if rows:
        final_df = pd.concat(rows, ignore_index=True)
        out_path = RESULTS_DIR / f"metrics_MC_{args.mc_id:03d}.csv"
        final_df.to_csv(out_path, index=False)
        print(f"=== Results saved to {out_path} ===\n")