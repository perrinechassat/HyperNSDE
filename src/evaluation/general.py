import torch
import time
import pandas as pd
from src.evaluation.fidelity import * 
from src.evaluation.privacy import *
from src.evaluation.utility import *
from src.evaluation.utils import *
from sklearn.model_selection import train_test_split
import warnings

# metrics_list = ["longi.MMD", "static.MMD", "longi.JS_div", "static.JS_div", "longi.pairwise_corr", "static.pairwise_corr", "longi.spatial_corr", "longi.KL_obs_rate", "discriminative_score", "prediction_score", "NNAA", "MIR", "dependence_score"]

# Metric registry with consistent naming
FIDELITY_METRICS_REG = {
    "longi.MMD": lambda real, syn, types, **kw: maximum_mean_discrepancy_longi(
        real["X"], real["M"], syn["X"], syn["M"], real["T"], **kw
    ),
    "static.MMD": lambda real, syn, types, **kw: maximum_mean_discrepancy_static(
        real["W"], syn["W"], **kw
    ),
    "longi.JS_div": lambda real, syn, types, **kw: js_divergence_longi(
        real["X"], real["M"], syn["X"], syn["M"], **kw
    ),
    "static.JS_div": lambda real, syn, types, **kw: js_divergence_static(
        real["W"], syn["W"], types, **kw
    ),
    "longi.pairwise_corr": lambda real, syn, types, **kw: pairwise_correlation_longi(
        real["X"], real["M"], syn["X"], syn["M"], **kw
    ),
    "static.pairwise_corr": lambda real, syn, types, **kw: pairwise_correlation_static(
        real["W"], syn["W"], types, **kw
    ),
    "global_pairwise_corr": lambda real, syn, types, **kw: pairwise_correlation_global(
        real["X"], real["M"], real["W"], syn["X"], syn["M"], syn["W"], **kw
    ),
    "longi.KL_obs_rate": lambda real, syn, types, **kw: kl_divergence_event_rate(
        real["X"], real["M"], real["T"], syn["X"], syn["M"], syn["T"], **kw
    ),
    "discriminative_score": lambda real, syn, types, **kw: discriminative_score(
        real["X"], real["M"], real["T"], syn["X"], syn["M"], syn["T"],
        W_real=real["W"], W_syn=syn["W"], device=None, **kw
    ),
}

PRIVACY_METRICS_REG = {
    "NNAA": nnaa,
    "MIR": membership_inference_risk,
    "dependence_score": dependence_score
}

UTILITY_METRICS_REG = {
    "prediction_score": prediction_score_TSTR 
}

class EvaluationEngine:
    def __init__(
        self, 
        real_all: dict, 
        real_train: dict, 
        real_test: dict, 
        static_types=None, 
        device='cuda',
        metric_config=None,
        fidelity_metrics = [], 
        utility_metrics = [], 
        privacy_metrics = []
    ):
        """
        Initializes the engine and PRE-COMPUTES all Real-Data statistics.
        Run this ONLY ONCE per experiment.
        """
        self.device = device
        self.static_types = static_types
        self.conf = metric_config or {}

        # Validate input data
        self._validate("real_all", real_all, ["X", "M", "T", "W"])
        self._validate("real_train", real_train, ["X", "M", "W"])
        self._validate("real_test", real_test, ["X", "M", "W", "T"])
        
        # Move data to GPU once
        self.r_all = self._to_device(real_all)
        self.r_tr = self._to_device(real_train)
        self.r_te = self._to_device(real_test)

        print("[Init] Pre-computing Real Data Statistics...")

        # [Init] Static MMD
        if "static.MMD" in fidelity_metrics:
            XX = self._precompute_static_mmd_real(self.r_te["W"])
            self.conf["static.MMD"]["cached_real_data"] = XX

        # [Init] Longitudinal Pairwise Correlation
        if "longi.pairwise_corr" in fidelity_metrics:
            corr_real = self._precompute_pairwise_corr_longi_real(self.r_te["X"], self.r_te["M"])
            self.conf["longi.pairwise_corr"]["cached_real_data"] = corr_real

        if "global_pairwise_corr" in fidelity_metrics:
            corr_real = self._precompute_pairwise_corr_global_real(self.r_te["X"], self.r_te["M"], self.r_te["W"])
            self.conf["global_pairwise_corr"]["cached_real_data"] = corr_real
           
        # [Init] Static Pairwise Correlation
        if "static.pairwise_corr" in fidelity_metrics:
            corr_real = self._precompute_pairwise_corr_stat_real(self.r_te["W"])
            self.conf["static.pairwise_corr"]["cached_real_data"] = corr_real

        # [Init] Longitudinal KL Divergence of Observation Rates
        if "longi.KL_obs_rate" in fidelity_metrics:
            events_r, sample_indices_r, counts_r, deltas_r, indices_r = self._precompute_kl_div_event_rate(self.r_te["X"], self.r_te["M"], self.r_te["T"])
            self.conf["longi.KL_obs_rate"]["cached_real_data"] = (events_r, sample_indices_r, counts_r, deltas_r, indices_r)

        # [Init] Discriminative Score
        if "discriminative_score" in fidelity_metrics:
            x_real_aug, m_real_aug = self._preprocess_real_discriminative_score(self.r_te["X"], self.r_te["M"], self.r_te["T"], self.r_te["W"])
            self.conf["discriminative_score"]["cached_real_data"] = (x_real_aug, m_real_aug)

        # [Init] Prediction Score (TSTR)
        if "prediction_score" in utility_metrics:
            x_test_real, m_test_real, y_test_real, ym_test_real = self._preprocess_real_prediction_score(self.r_te["X"], self.r_te["M"], self.r_te["T"], self.r_te["W"])
            self.conf["prediction_score"]["cached_real_data"] = (x_test_real, m_test_real, y_test_real, ym_test_real)
            self.trtr_mse = self._train_real_baseline()

        # [Init] Privacy NNAA
        if "NNAA" in privacy_metrics:
            dict_nnaa_real = self._precompute_nnaa_real(
                self.r_tr["X"], self.r_tr["M"], self.r_tr["W"],
                self.r_te["X"], self.r_te["M"], self.r_te["W"], self.r_te["T"]
            )
            self.conf["NNAA"]["cached_real_data"] = dict_nnaa_real
        
        # [Init] Dependence Score
        if "dependence_score" in privacy_metrics:
            Rr = self._precompute_real_dependence_score(self.r_all["X"], self.r_all["M"])
            self.conf["dependence_score"]["cached_real_data"] = Rr

        print("[Init] Cache build complete.")

    def _to_device(self, d):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    def _validate(self, name, data, keys):
        if not isinstance(data, dict):
            raise TypeError(f"{name} must be dict, got {type(data)}")
        missing = [k for k in keys if k not in data]
        if missing:
            raise KeyError(f"{name} missing keys {missing}. Expected keys: {keys}")
    
    def _precompute_static_mmd_real(self, w_real):
        kernel = self.conf["static.MMD"].get("kernel", "rbf")
        gamma = self.conf["static.MMD"].get("gamma", 1.0)
        coef0 = self.conf["static.MMD"].get("coef0", 0.0)
        degree = self.conf["static.MMD"].get("degree", 2)
        XX = compute_kernel_matrix(w_real, w_real, kernel, gamma, coef0, degree)
        return XX
    
    def _precompute_pairwise_corr_longi_real(self, x_real, m_real):
        N_r, T, F = x_real.shape
        xr_flat = x_real.reshape(N_r * T, F)
        mr_flat = m_real.reshape(N_r * T, F)    
        corr_real = compute_masked_correlation_matrix(xr_flat, mr_flat)
        return corr_real
    
    def _precompute_pairwise_corr_global_real(self, x_real, m_real, W_real):
        x_real_aug, m_real_aug = augment_with_statics(x_real, m_real, W_real, use_nans_for_statics=False) # Shape: (N, T, F_total)
        N_r, T, F = x_real_aug.shape
        xr_flat = x_real_aug.reshape(N_r * T, F)
        mr_flat = m_real_aug.reshape(N_r * T, F)    
        corr_real = compute_masked_correlation_matrix(xr_flat, mr_flat)
        return corr_real

    def _precompute_pairwise_corr_stat_real(self, W_real):
        W_real = W_real.clone()
        for i, (dtype, _, _) in enumerate(self.static_types):
            if dtype == 'ordinal':
                _, W_real[:, i] = torch.unique(W_real[:, i], return_inverse=True)
                W_real[:, i] = W_real[:, i].float()
            elif dtype == 'cat':
                W_real[:, i] = W_real[:, i].float()
        real_mean = W_real.mean(dim=0, keepdim=True)
        real_std = W_real.std(dim=0, keepdim=True) + 1e-8
        Z_real = (W_real - real_mean) / real_std
        N_r = W_real.size(0)
        corr_real = (Z_real.T @ Z_real) / (N_r - 1)
        return corr_real
    
    def _precompute_kl_div_event_rate(self, x_real, m_real, T_real):
        # Same logic for real data: Collapse features
        N_r = x_real.shape[0]
        mask_r = (m_real.sum(dim=-1) > 0)
        if T_real.dim() == 1:
            T_real_exp = T_real.unsqueeze(0).expand(N_r, -1)
        else:
            T_real_exp = T_real
        events_r = T_real_exp[mask_r]
        sample_indices_r = torch.arange(N_r, device=self.device).unsqueeze(1).expand_as(mask_r)[mask_r]
        counts_r = mask_r.sum(dim=1).float()

        # detla version
        mask_r_bool = (m_real.sum(dim=-1) > 0)
        deltas_r, indices_r = get_deltas_and_indices(mask_r_bool, T_real, N_r)
        return (events_r, sample_indices_r, counts_r, deltas_r, indices_r)

    def _preprocess_real_discriminative_score(self, x_real, m_real, T_real, W_real):
        filling_method = self.conf.get("discriminative_score", {}).get("filling_type", "last")
        xr_f, mr_f, _ = fill_missing_values_long(x_real, m_real, T_real, filling_type=filling_method)
        x_real_aug, m_real_aug = augment_with_statics(xr_f, mr_f, W_real, use_nans_for_statics=False)
        return (x_real_aug, m_real_aug)

    def _precompute_nnaa_real(self, x_r_train, m_r_train, W_r_train, x_r_test, m_r_test, W_r_test, T):
        dict_nnaa_real = {}
        if self.conf["NNAA"]["dist_type"] == 'signature':
            dict_nnaa_real["L_rt_rt"] = fast_signature_dist_long(x_r_test, x_r_test, m_r_test, m_r_test, T,
                                                                 kernel=self.conf["NNAA"]['kernel'], 
                                                                 dyadic_order=self.conf["NNAA"]['dyadic_order'], 
                                                                 sigma=self.conf["NNAA"]['sigma'], 
                                                                 lead_lag=self.conf["NNAA"]['lead_lag']) 
            dict_nnaa_real["L_rtr_rtr"] = fast_signature_dist_long(x_r_train, x_r_train, m_r_train, m_r_train, T,
                                                                 kernel=self.conf["NNAA"]['kernel'], 
                                                                 dyadic_order=self.conf["NNAA"]['dyadic_order'], 
                                                                 sigma=self.conf["NNAA"]['sigma'], 
                                                                 lead_lag=self.conf["NNAA"]['lead_lag']) 
        else:
            dict_nnaa_real["L_rt_rt"] = fast_pdist_long(x_r_test, x_r_test, squared=False)
            dict_nnaa_real["L_rtr_rtr"] = fast_pdist_long(x_r_train, x_r_train, squared=False)
        dict_nnaa_real["S_rt_rt"]   = fast_pdist_static(W_r_test, W_r_test)
        dict_nnaa_real["S_rtr_rtr"] = fast_pdist_static(W_r_train, W_r_train)
        return dict_nnaa_real

    def _precompute_real_dependence_score(self, x_real, m_real):
        max_lag = self.conf.get("dependence_score", {}).get("max_lag", None)
        Rr = autocorr_matrix_dataset(x_real, m_real, max_lag=max_lag) # (V,K)
        return Rr
    
    def _preprocess_real_prediction_score(self, x_real, m_real, T_real, W_real):
        xr_f, mr_f, _ = fill_missing_values_long(x_real, m_real, T_real, filling_type='last')
        if W_real is not None:
            Wr_rep = W_real.unsqueeze(1).expand(-1, xr_f.size(1), -1)
            xr_aug = torch.cat([xr_f, Wr_rep], dim=-1)
        else:
            xr_aug = xr_f
        # future_steps = self.conf.get("prediction_score", {}).get("future_steps", 5)
        input_len = self.conf.get("prediction_score", {}).get("input_len", None)
        if input_len is None:
            input_len = int(0.5*xr_aug.size(1))
        x_test_real = xr_aug[:, :input_len, :].to(self.device)
        m_test_real = mr_f[:, :input_len, :].to(self.device)
        y_test_real = xr_f[:, input_len:, :].to(self.device)
        ym_test_real = mr_f[:, input_len:, :].to(self.device)
        return (x_test_real, m_test_real, y_test_real, ym_test_real)

    def _train_real_baseline(self):
        # Run the "Train on Real, Test on Real" logic once
        print("[Init] Training Real-on-Real Utility Baseline...")
        params = self.conf.get("prediction_score", {})
        res = prediction_score_TSTR(x_real=self.r_te["X"], m_real=self.r_te["M"], T_real=self.r_te["T"], W_real=self.r_te["W"],
            x_syn=self.r_tr["X"], m_syn=self.r_tr["M"], T_syn=self.r_all["T"], W_syn=self.r_tr["W"],
            device=self.device,
            **params
        )
        return res["mse_test"]

    def compute_baseline(self, fidelity_metrics, utility_metrics, privacy_metrics, seed=0):
        """
        Compute Baseline Real/Real
        """
        # Split train dataset into "syn" and "train"
        n_train = self.r_tr["X"].shape[0]

        idx_fake_r_tr, idx_fake_syn = train_test_split(np.arange(n_train), test_size=int(self.r_te["X"].shape[0]), random_state=seed, shuffle=True)
        fake_syn = {
            "X":self.r_tr["X"][idx_fake_syn],
            "M":self.r_tr["M"][idx_fake_syn],
            "W":self.r_tr["W"][idx_fake_syn],
            "T":self.r_tr["T"],
        }
        fake_r_train = {
            "X":self.r_tr["X"][idx_fake_r_tr],
            "M":self.r_tr["M"][idx_fake_r_tr],
            "W":self.r_tr["W"][idx_fake_r_tr],
            "T":self.r_tr["T"],
        }
        fake_r_all = {
            "X": torch.cat((fake_r_train["X"],self.r_te["X"]), 0),
            "M": torch.cat((fake_r_train["M"],self.r_te["M"]), 0)
        }

        if "NNAA" in privacy_metrics:
            dict_nnaa_real_fake = self._precompute_nnaa_real(
                fake_r_train["X"], fake_r_train["M"], fake_r_train["W"],
                self.r_te["X"], self.r_te["M"], self.r_te["W"], self.r_te["T"]
            )

        if "dependence_score" in privacy_metrics:
            Rr_fake = self._precompute_real_dependence_score(fake_r_all["X"], fake_r_all["M"])

        res = {}
        # --- A. Fidelity (Fast) ---
        with torch.inference_mode():
            if fidelity_metrics:
                for name in fidelity_metrics:
                    if name == "discriminative_score": 
                        continue # Skip discriminative score for now
                    else:
                        fn = FIDELITY_METRICS_REG[name]
                        params = self.conf.get(name, {})
                        out = fn(self.r_te, fake_syn, types=self.static_types, **params)
                        for k, v in out.items():
                            res[f"FID.{name}.{k}"] = v


        # --- B. Utility & Discriminative (Slow - Training Required) ---
        with torch.enable_grad():
            if "discriminative_score" in fidelity_metrics:
                fn = FIDELITY_METRICS_REG["discriminative_score"]
                params = self.conf.get("discriminative_score", {})
                out = fn(self.r_te, fake_syn, types=self.static_types, **params)
                for k, v in out.items():
                    res[f"FID.discriminative_score.{k}"] = v
            if "prediction_score" in utility_metrics:
                params = self.conf.get("prediction_score", {})
                fn = UTILITY_METRICS_REG["prediction_score"]
                res_trts = fn(x_real=self.r_te["X"], m_real=self.r_te["M"], T_real=self.r_te["T"], W_real=self.r_te["W"],
                                x_syn=fake_syn["X"], m_syn=fake_syn["M"], T_syn=fake_syn["T"], W_syn=fake_syn["W"],
                                device=self.device, **params)
                mse_ratio = res_trts["mse_test"] / self.trtr_mse
                res["UTIL.prediction_score.ratio"] = float(mse_ratio)
                res["UTIL.prediction_score.mse_trts"] = float(res_trts["mse_test"])
                res["UTIL.prediction_score.mse_trtr"] = float(self.trtr_mse)


        # --- C. Privacy (Fast Slicing) ---
        with torch.inference_mode():
            if privacy_metrics:
                for name in privacy_metrics:
                    if name == "dependence_score": 
                        fn = PRIVACY_METRICS_REG["dependence_score"]
                        params = self.conf.get("dependence_score", {})
                        params["cached_real_data"] = Rr_fake
                        out = fn(fake_r_all["X"], fake_r_all["M"], fake_syn["X"], fake_syn["M"], **params)
                        for k, v in out.items():
                            res[f"PRIV.dependence_score.{k}"] = float(v)
                    else:
                        fn = PRIVACY_METRICS_REG[name]
                        params = self.conf.get(name, {})
                        if name == 'NNAA':
                            params["cached_real_data"] = dict_nnaa_real_fake
                        params["cached_synthetic_data"] = None
                        out = fn(fake_r_train["X"], fake_r_train["M"], fake_r_train["W"],
                                self.r_te["X"], self.r_te["M"], self.r_te["W"],
                                fake_syn["X"], fake_syn["M"], fake_syn["W"], self.r_te["T"],
                                **params)
                        for k, v in out.items():
                            res[f"PRIV.{name}.{k}"] = float(v)
        
        res["generation_id"] = 0
        return pd.DataFrame([res])


    def evaluate_several_synthetic_sets(self, syn_all, n_gen, fidelity_metrics, utility_metrics, privacy_metrics, max_sets_training=-1, multinodes_mask=None, metrics_adapt_multinodes=["longi.MMD", "longi.pairwise_corr", "global_pairwise_corr", "prediction_score"]):
        """
        Evaluate several synthetic datasets using the cached real stats.
        Returns a DataFrame with all results.

        syn_all: dict with keys:
            - "X": Tensor of shape (n_gen, n_samples, T, F)
            - "M": Tensor of shape (n_gen, n_samples, T, F)
            - "W": Tensor of shape (n_gen, n_samples, F_static) or None
            - "T": Tensor of shape (n_samples,)
        n_gen: Number of synthetic sets (generations)
        fidelity_metrics: List of fidelity metric names to compute
        utility_metrics: List of utility metric names to compute
        privacy_metrics: List of privacy metric names to compute
        max_sets_training: Maximum number of synthetic sets to use for training-based metrics (e.g. prediction score, discriminative score)
        
        - Fidelity: Computed non-training metrics per generation (Fast).
        - Privacy: Computed GLOBALLY then sliced per generation (Fastest).
        - Utility/Fidelity: Computed for first 'max_sets_training' generations only (Slow).
        """
        syn_all = self._to_device(syn_all)
        results = [] # List of dicts, one per gen
        n_per_gen = syn_all["X"].shape[1]
        if max_sets_training < 0:
            max_sets_training = n_gen

        # --- PRE-COMPUTE GLOBAL PRIVACY DISTANCES (The Optimization) ---
        privacy_cache = {}
        compute_privacy = ("NNAA" in privacy_metrics or "MIR" in privacy_metrics)
        if compute_privacy:
            print("  [Batch] Pre-computing global distance matrices for Privacy...")
            syn_X_flat = syn_all["X"].view(-1, *syn_all["X"].shape[2:])
            syn_M_flat = syn_all["M"].view(-1, *syn_all["M"].shape[2:])
            if self.conf["NNAA"]["dist_type"] == 'signature':
                privacy_cache["L_rt_s_all"] = fast_signature_dist_long(self.r_te["X"], syn_X_flat, self.r_te["M"], syn_M_flat, self.r_te["T"],
                                                                    kernel=self.conf["NNAA"]['kernel'], 
                                                                    dyadic_order=self.conf["NNAA"]['dyadic_order'], 
                                                                    sigma=self.conf["NNAA"]['sigma'], 
                                                                    lead_lag=self.conf["NNAA"]['lead_lag']) 
                privacy_cache["L_rtr_s_all"] = fast_signature_dist_long(self.r_tr["X"], syn_X_flat, self.r_tr["M"], syn_M_flat, self.r_te["T"],
                                                                    kernel=self.conf["NNAA"]['kernel'], 
                                                                    dyadic_order=self.conf["NNAA"]['dyadic_order'], 
                                                                    sigma=self.conf["NNAA"]['sigma'], 
                                                                    lead_lag=self.conf["NNAA"]['lead_lag']) 
            else:
                raise Exception("Invalid dist_type for NNAA")

            if self.r_te["W"] is not None and syn_all["W"] is not None:
                syn_W_flat = syn_all["W"].view(-1, *syn_all["W"].shape[2:])
                privacy_cache["S_rt_s_all"] = fast_pdist_static(self.r_te["W"], syn_W_flat)
                privacy_cache["S_rtr_s_all"] = fast_pdist_static(self.r_tr["W"], syn_W_flat)

        # --- 2. GENERATION LOOP ---
        print("[Eval] Starting evaluation of generation loop...")
        for g in range(n_gen):
            syn_g = {
                "X": syn_all["X"][g],
                "M": syn_all["M"][g],
                "W": syn_all["W"][g] if syn_all["W"] is not None else None,
                "T": syn_all["T"]
            }
            if multinodes_mask is not None: 
                syn_g_mndes = {
                    "X": syn_all["X"][g],
                    "M": multinodes_mask,
                    "W": syn_all["W"][g] if syn_all["W"] is not None else None,
                    "T": syn_all["T"]
                }
            res_g = {}

            # --- A. Fidelity (Fast) ---
            with torch.inference_mode():
                if fidelity_metrics:
                    for name in fidelity_metrics:
                        if name == "discriminative_score": 
                            continue # Skip discriminative score for now
                        else:
                            fn = FIDELITY_METRICS_REG[name]
                            params = self.conf.get(name, {})
                            if multinodes_mask is not None and name in metrics_adapt_multinodes:
                                out = fn(self.r_te, syn_g_mndes, types=self.static_types, **params)
                            else:
                                out = fn(self.r_te, syn_g, types=self.static_types, **params)
                            for k, v in out.items():
                                res_g[f"FID.{name}.{k}"] = v

            # --- B. Utility & Discriminative (Slow - Training Required) ---
            # We only run this for the first few sets
            if g < max_sets_training:
                with torch.enable_grad():
                    if "discriminative_score" in fidelity_metrics:
                        fn = FIDELITY_METRICS_REG["discriminative_score"]
                        params = self.conf.get("discriminative_score", {})
                        out = fn(self.r_te, syn_g, types=self.static_types, **params)
                        for k, v in out.items():
                            res_g[f"FID.discriminative_score.{k}"] = v
                    if "prediction_score" in utility_metrics:
                        params = self.conf.get("prediction_score", {})
                        fn = UTILITY_METRICS_REG["prediction_score"]
                        if multinodes_mask is not None and name in metrics_adapt_multinodes:
                            res_trts = fn(x_real=self.r_te["X"], m_real=self.r_te["M"], T_real=self.r_te["T"], W_real=self.r_te["W"],
                                        x_syn=syn_g_mndes["X"], m_syn=syn_g_mndes["M"], T_syn=syn_g_mndes["T"], W_syn=syn_g_mndes["W"],
                                        device=self.device, **params)
                        else:
                            res_trts = fn(x_real=self.r_te["X"], m_real=self.r_te["M"], T_real=self.r_te["T"], W_real=self.r_te["W"],
                                        x_syn=syn_g["X"], m_syn=syn_g["M"], T_syn=syn_g["T"], W_syn=syn_g["W"],
                                        device=self.device, **params)
                        mse_ratio = res_trts["mse_test"] / self.trtr_mse
                        res_g["UTIL.prediction_score.ratio"] = float(mse_ratio)
                        res_g["UTIL.prediction_score.mse_trts"] = float(res_trts["mse_test"])
                        res_g["UTIL.prediction_score.mse_trtr"] = float(self.trtr_mse)
            else:
                # Fill with NaN for skipped generations
                if "discriminative_score" in fidelity_metrics:
                    res_g["FID.discriminative_score.disc_score"] = float('nan')
                if "prediction_score" in utility_metrics:
                    res_g["UTIL.prediction_score.ratio"] = float('nan')

            # --- C. Privacy (Fast Slicing) ---
            with torch.inference_mode():
                if privacy_metrics:
                    if compute_privacy:
                        start_idx = g * n_per_gen
                        end_idx = (g + 1) * n_per_gen
                        cached_synthetic_data = {
                            "L_rt_s": privacy_cache["L_rt_s_all"][:, start_idx:end_idx],
                            "L_rtr_s": privacy_cache["L_rtr_s_all"][:, start_idx:end_idx],
                            "S_rt_s": privacy_cache.get("S_rt_s_all", None)[:, start_idx:end_idx] if "S_rt_s_all" in privacy_cache else None,
                            "S_rtr_s": privacy_cache.get("S_rtr_s_all", None)[:, start_idx:end_idx] if "S_rtr_s_all" in privacy_cache else None,
                        }
                        self.conf["NNAA"]["cached_synthetic_data"] = cached_synthetic_data
                        self.conf["MIR"]["cached_synthetic_data"] = cached_synthetic_data

                    for name in privacy_metrics:
                        if name == "dependence_score": 
                            fn = PRIVACY_METRICS_REG["dependence_score"]
                            params = self.conf.get("dependence_score", {})
                            out = fn(self.r_all["X"], self.r_all["M"], syn_g["X"], syn_g["M"], **params)
                            for k, v in out.items():
                                res_g[f"PRIV.dependence_score.{k}"] = float(v)
                        else:
                            fn = PRIVACY_METRICS_REG[name]
                            params = self.conf.get(name, {})
                            out = fn(self.r_tr["X"], self.r_tr["M"], self.r_tr["W"],
                                    self.r_te["X"], self.r_te["M"], self.r_te["W"],
                                    syn_g["X"], syn_g["M"], syn_g["W"], self.r_te["T"],
                                    **params)
                            for k, v in out.items():
                                res_g[f"PRIV.{name}.{k}"] = float(v)
             
            res_g["generation_id"] = g
            results.append(res_g)

        print("[Eval] Evaluation loop complete.")
        return pd.DataFrame(results)
    

    def evaluate_one_synthetic_set(self, syn_data, fidelity_metrics, utility_metrics, privacy_metrics):
        """
        Evaluate a single synthetic dataset using the cached real stats.
        Fast enough to run in a loop.
        """
        syn = self._to_device(syn_data)
        results = {}

        # ========== Fidelity Metrics ==========
        if fidelity_metrics:
            for name in fidelity_metrics:
                print(f"Computing fidelity metric: {name}")
                t_init = time.time()
                fn = FIDELITY_METRICS_REG[name]
                params = self.conf.get(name, {})
                out = fn(self.r_te, syn, types=self.static_types, **params)
                for k, v in out.items():
                    results[f"FID.{name}.{k}"] = v
                    print(v)
                print(f"Time taken: {time.time() - t_init:.2f} seconds")

        # ========== Utility Metrics ==========
        if utility_metrics:
            for name in utility_metrics:
                print(f"Computing utility metric: {name}")
                t_init = time.time()
                params = self.conf.get(name, {})
                if name == "prediction_score":
                    fn = UTILITY_METRICS_REG[name]
                    # TRTS and TRTR
                    res_trts = fn(x_real=self.r_te["X"], m_real=self.r_te["M"], T_real=self.r_te["T"], W_real=self.r_te["W"],
                                x_syn=syn["X"], m_syn=syn["M"], T_syn=syn["T"], W_syn=syn["W"],
                                device=self.device, **params)
                    mse_ratio = res_trts["mse_test"] / self.trtr_mse
                    results["UTIL.prediction_score.ratio"] = float(mse_ratio)
                    results["UTIL.prediction_score.mse_trts"] = float(res_trts["mse_test"])
                    results["UTIL.prediction_score.mse_trtr"] = float(self.trtr_mse)
                    print(f"Time taken: {time.time() - t_init:.2f} seconds")
                else:
                    raise NotImplementedError(f"Utility metric {name} not supported yet.")

        # ========== Privacy Metrics ==========
        if privacy_metrics:
            for name in privacy_metrics:
                print(f"Computing privacy metric: {name}")
                t_init = time.time()
                fn = PRIVACY_METRICS_REG[name]
                params = self.conf.get(name, {})
                if name == "dependence_score":
                    out = fn(self.r_all["X"], self.r_all["M"], 
                            syn["X"], syn["M"], 
                            **params)
                else:
                    out = fn(self.r_tr["X"], self.r_tr["M"], self.r_tr["W"],
                            self.r_te["X"], self.r_te["M"], self.r_te["W"],
                            syn["X"], syn["M"], syn["W"], self.r_te["T"],
                            **params)
                for k, v in out.items():
                    results[f"PRIV.{name}.{k}"] = float(v)
                print(f"Time taken: {time.time() - t_init:.2f} seconds")  

        return results