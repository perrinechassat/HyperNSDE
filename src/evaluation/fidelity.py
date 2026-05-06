
from __future__ import annotations
from typing import Dict, List, Literal, Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pysiglib.torch_api as pysiglib 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from src.evaluation.utils import *
from src.utils import subtract_initial_point
import sys
sys.modules['pykeops'] = None
from external.s4.models.s4.s4 import S4Block as S4D
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torchcde

# def unbiased_mmd_sq(x1_compress, x2_compress, static_kernel, dyadic_order, lead_lag, device):
#     """
#     Unbiased MMD^2 from gram matrices with diagonal normalization
#     to avoid large negative values.
#     """
#     kw = dict(dyadic_order=dyadic_order,
#                 static_kernel=static_kernel,
#                 time_aug=False,
#                 lead_lag=lead_lag,
#                 max_batch=8)

#     K11 = pysiglib.sig_kernel_gram(x1_compress, x1_compress, **kw)  # (N1, N1)
#     K22 = pysiglib.sig_kernel_gram(x2_compress, x2_compress, **kw)  # (N2, N2)
#     K12 = pysiglib.sig_kernel_gram(x1_compress, x2_compress, **kw)  # (N1, N2)

#     # Normalize by mean diagonal to stabilize kernel scale
#     scale = K11.diag().mean().clamp(min=1e-8)
#     K11 = K11 / scale
#     K22 = K22 / scale
#     K12 = K12 / scale

#     # Unbiased estimator (exclude diagonal in self-terms)
#     K11_sum = K11.sum() - K11.diag().sum()
#     K22_sum = K22.sum() - K22.diag().sum()

#     mmd_sq = (K11_sum / (N1 * (N1 - 1))
#             + K22_sum / (N2 * (N2 - 1))
#             - 2.0 * K12.mean())

#     del K11, K22, K12
#     torch.cuda.empty_cache()

#     return mmd_sq.to(device)



    

    
    
"""
-----------------------------------------------------------------------------
                                Fidelity metrics
-----------------------------------------------------------------------------
"""

# -----------------------------------------------------------------------------
# MMD (with different kernel)
# -----------------------------------------------------------------------------

def median_heuristic_sigma(paths: list[torch.Tensor]) -> float:
    """
    paths: list of tensors of shape (length_i, d)
    """
    flat = []
    for p in paths:
        flat.append(p.flatten())
    X = torch.nn.utils.rnn.pad_sequence(flat, batch_first=True)
    dists = torch.cdist(X, X, p=2)
    vals = dists[torch.triu(torch.ones_like(dists), diagonal=1) > 0]
    return float(vals.median().item())

def maximum_mean_discrepancy_longi(
        x_real: torch.Tensor, m_real: torch.Tensor, 
        x_syn: torch.Tensor, m_syn: torch.Tensor,
        T: torch.Tensor,
        kernel: Literal["sig linear", "sig rbf", "gaussian"] = "gaussian", 
        sigmas = 0.5, 
        cached_real_data = None,
        filling_type = None,
        lead_lag = False,
        subtrack_init_point = False, 
        scale_time = False,
        dyadic_order = 1
    ) -> Dict[str, float]:

    device = x_real.device


    # if kernel == "sig linear" or kernel == "sig rbf":

    #     if kernel == "sig linear":
    #         static_kernel = pysiglib.LinearKernel()
    #     else:
    #         static_kernel = pysiglib.RBFKernel(sigmas)

    #     if subtrack_init_point:
    #         x_real = subtract_initial_point(x_real).to(x_real.device)
    #         x_syn = subtract_initial_point(x_syn).to(x_syn.device)

    #     if scale_time:
    #         T_scale = ((T - T.min()) / (T.max() - T.min()))*3.5
    #         x_real_time_aug = time_aug(x_real, T_scale)
    #         x_syn_time_aug = time_aug(x_syn, T_scale)
    #     else:
    #         x_real_time_aug = time_aug(x_real, T)
    #         x_syn_time_aug = time_aug(x_syn, T)

    #     x_real_compress = compress_batch(x_real_time_aug, m_real)
    #     x_syn_compress = compress_batch(x_syn_time_aug, m_syn)

    #     x_real_compress = x_real_compress.contiguous()
    #     x_syn_compress = x_syn_compress.contiguous()
    #     mmd_score = pysiglib.sig_mmd(sample1=x_real_compress, 
    #                             sample2=x_syn_compress,
    #                             dyadic_order=dyadic_order,
    #                             static_kernel=static_kernel,
    #                             time_aug=False, # included separately because can be irregular
    #                             lead_lag=lead_lag,
    #                             max_batch=8).to(device)
    #     return {"score": float(mmd_score.item())}

    if kernel == "sig linear" or kernel == "sig rbf":

        x_real[~m_real.bool()] = torch.nan
        x_syn[~m_syn.bool()] = torch.nan

        if subtrack_init_point:
            x_real = subtract_initial_point(x_real).to(x_real.device)
            x_syn = subtract_initial_point(x_syn).to(x_syn.device)

        if scale_time:
            T_scale = (T - T.min()) / (T.max() - T.min())
            x_real_time_aug = time_aug(x_real, T_scale)
            x_syn_time_aug = time_aug(x_syn, T_scale)
        else:
            x_real_time_aug = time_aug(x_real, T)
            x_syn_time_aug = time_aug(x_syn, T)

        # fill mask until last observed value
        m_fill_real = expand_mask_to_last_obs(m_real)
        m_fill_syn = expand_mask_to_last_obs(m_syn)
        
        x_real_time_aug_interp = torchcde.linear_interpolation_coeffs(x_real_time_aug, t=T).to(x_real.device)
        x_syn_time_aug_interp = torchcde.linear_interpolation_coeffs(x_syn_time_aug, t=T).to(x_real.device)

        # x_real_compress = compress_batch(x_real_time_aug, m_real)
        # x_syn_compress = compress_batch(x_syn_time_aug, m_syn)
        x_real_compress = compress_batch(x_real_time_aug_interp, m_fill_real)
        x_syn_compress = compress_batch(x_syn_time_aug_interp, m_fill_syn)
        x_real_compress = x_real_compress.contiguous()
        x_syn_compress = x_syn_compress.contiguous()

        # N1 = x_real_compress.shape[0]
        # N2 = x_syn_compress.shape[0]

        if kernel == "sig linear":
            static_kernel = pysiglib.LinearKernel()
            mmd_score = pysiglib.sig_mmd(sample1=x_real_compress, 
                                sample2=x_syn_compress,
                                dyadic_order=dyadic_order,
                                static_kernel=static_kernel,
                                time_aug=False, # included separately because can be irregular
                                lead_lag=lead_lag,
                                max_batch=8).to(device)
            return {"score": float(mmd_score.item())}
        else:
            # sigma0 = median_heuristic_sigma(x_real_compress[:50])  # for instance
            # print("Sigma:", sigma0)
            # sigma_list = [0.5*sigma0, sigma0, 2.0*sigma0]
            sigma_list = sigmas

            mmd_values = []
            for sigma in sigma_list:
                static_kernel = pysiglib.RBFKernel(sigma)
                mmd_sigma = pysiglib.sig_mmd(
                    sample1=x_real_compress,
                    sample2=x_syn_compress,
                    dyadic_order=dyadic_order,
                    static_kernel=static_kernel,
                    time_aug=False,
                    lead_lag=lead_lag,
                    max_batch=8
                ).to(device)
                mmd_values.append(mmd_sigma)

            mmd_tensor = torch.stack(mmd_values)
            mmd_score = mmd_tensor.mean()

            out = {"score": float(mmd_score.item())}
            for sigma, val in zip(sigma_list, mmd_values):
                out[f"mmd_sigma_{sigma:g}"] = float(val.item())

            return out
        # # ---- Linear kernel ----
        # if kernel == "sig linear":
        #     static_kernel = pysiglib.LinearKernel()
        #     mmd_score = unbiased_mmd_sq(x_real_compress, x_syn_compress, static_kernel)
        #     return {"score": float(mmd_score.item())}

        # # ---- RBF kernel (loop over sigmas) ----
        # else:
        #     mmd_values = []
        #     for sigma in sigmas:
        #         static_kernel = pysiglib.RBFKernel(sigma)
        #         mmd_sigma = unbiased_mmd_sq(
        #             x_real_compress, x_syn_compress, static_kernel, dyadic_order, lead_lag, device)
        #         mmd_values.append(mmd_sigma)

        #     mmd_tensor = torch.stack(mmd_values)
        #     mmd_score  = mmd_tensor.mean()

        #     out = {"score": float(mmd_score.item())}
        #     for sigma, val in zip(sigmas, mmd_values):
        #         out[f"mmd_sigma_{sigma:g}"] = float(val.item())

        #     return out
        

    elif kernel == "gaussian":
        # Flatten Time Series
        N_real, T, V = x_real.shape
        N_syn = x_syn.shape[0]
        
        if isinstance(sigmas, (float, int)):
            sigmas = [sigmas]
        sigmas_tensor = torch.tensor(sigmas, device=x_real.device).view(-1, 1, 1) # shape: (num_sigmas, 1, 1)

        if cached_real_data is not None:
            score_rr = cached_real_data.clone()
        else:
            D_rr = fast_pdist_long(x_real, x_real, squared=True) # D_rr: (N_real, N_real)
            D_rr_exp = D_rr.unsqueeze(0)
            K_rr = torch.exp(-D_rr_exp / (2 * sigmas_tensor**2)).mean(dim=0)
            K_rr_no_diag = K_rr - torch.diag(torch.diag(K_rr))
            score_rr = K_rr_no_diag.sum() / (N_real * (N_real - 1))

        # Compute Distance Matrices (Squared L2)
        D_ss = fast_pdist_long(x_syn, x_syn, squared=True) # D_ss: (N_syn, N_syn)
        D_rs = fast_pdist_long(x_real, x_syn, squared=True) # D_rs: (N_real, N_syn)
        D_ss_exp = D_ss.unsqueeze(0)
        D_rs_exp = D_rs.unsqueeze(0)
        
        # Compute Gaussian: exp(-D^2 / (2*sigma^2)), sum over sigmas immediately to save memory
        K_ss = torch.exp(-D_ss_exp / (2 * sigmas_tensor**2)).mean(dim=0)
        K_rs = torch.exp(-D_rs_exp / (2 * sigmas_tensor**2)).mean(dim=0)

        # Compute MMD Score (Unbiased Estimate), remove diagonals from self-similarity matrices
        K_ss_no_diag = K_ss - torch.diag(torch.diag(K_ss))
        score_ss = K_ss_no_diag.sum() / (N_syn * (N_syn - 1))
        score_rs = K_rs.mean() 
        mmd_score = score_rr + score_ss - 2 * score_rs
        return {"score": float(mmd_score.item())}

    else:
        raise ValueError(f"Unsupported kernel '{kernel}'. Must be one of ['sig linear', 'sig rbf', 'gaussian'].")

    # return {"score": float(mmd_score.item())}


def maximum_mean_discrepancy_static(
    X_real: torch.Tensor,
    X_syn: torch.Tensor,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 2,
    coef0: float = 0.0,
    cached_real_data = None
) -> Dict[str, float]:
    """
    Compute the Maximum Mean Discrepancy (MMD) between real and synthetic static datasets.
    
    The lower the score, the closer the two distributions are (0 = identical).

    Args:
        X_real: (N_real, D) real samples, torch.Tensor or np.ndarray
        X_syn:  (N_syn,  D) synthetic samples, torch.Tensor or np.ndarray
        kernel: kernel type, one of {'linear', 'rbf', 'polynomial'}
        gamma:  RBF or polynomial kernel parameter (default=1.0)
        degree: polynomial kernel degree (default=2)
        coef0:  polynomial kernel constant term (default=0.0)

    Returns:
        dict with {"MMD": float}
    """
    if X_real.dim() > 2: X_real = X_real.reshape(X_real.size(0), -1)
    if X_syn.dim() > 2:  X_syn = X_syn.reshape(X_syn.size(0), -1)

    if kernel == "linear":
        delta = X_real.mean(dim=0) - X_syn.mean(dim=0)
        score = torch.dot(delta, delta)
        return {"score": score.item()}

    if cached_real_data is not None:
        XX = cached_real_data.clone()
    else:
        XX = compute_kernel_matrix(X_real, X_real, kernel, gamma, coef0, degree)
    YY = compute_kernel_matrix(X_syn, X_syn, kernel, gamma, coef0, degree)
    XY = compute_kernel_matrix(X_real, X_syn, kernel, gamma, coef0, degree)
    score = XX.mean() + YY.mean() - 2 * XY.mean()
    return {"score": score.item()}


# -----------------------------------------------------------------------------
# JS-divergence 
# -----------------------------------------------------------------------------

def js_divergence_native(p: torch.Tensor, q: torch.Tensor):
    """
    Compute JS divergence between two probability distributions (1D tensors) in pure Torch.
    """
    m = 0.5 * (p + q)
    # KL(P||M) + KL(Q||M)
    # Add epsilon for numerical stability inside log
    eps = 1e-12
    return 0.5 * (torch.sum(p * (torch.log(p + eps) - torch.log(m + eps))) + 
                  torch.sum(q * (torch.log(q + eps) - torch.log(m + eps))))


def js_divergence_longi(
        x_real: torch.Tensor, m_real: torch.Tensor,
        x_syn: torch.Tensor,  m_syn: torch.Tensor,
        n_bins: int = 20
    ) -> Dict[str, float]:
    """
    Jensen–Shannon divergence between real and synthetic longitudinal data.

    Computes the marginal JS-divergence for each variable over all timepoints,
    ignoring missing (masked) values.

    Args:
        x_real, m_real: real data and mask (N_r, T, V)
        x_syn,  m_syn:  synthetic data and mask (N_s, T, V)
        n_bins: number of bins for continuous variables
        normalize: whether to normalize histograms to probabilities

    Returns:
        dict with:
            - JS_mean: mean JS divergence across variables
            - JS_per_var: per-variable divergences
    """
    V = x_real.shape[-1]
    js_scores = []
    
    for v in range(V):
        xr = x_real[..., v][m_real[..., v].bool()]
        xs = x_syn[..., v][m_syn[..., v].bool()]

        if xr.numel() == 0 or xs.numel() == 0:
            continue
        q_low = torch.quantile(xr, 0.01)
        q_high = torch.quantile(xr, 0.99)
        
        width = q_high - q_low
        if width == 0: width = 1.0
        lo = q_low - 0.1 * width
        hi = q_high + 0.1 * width
        
        xr_clipped = torch.clamp(xr, lo, hi)
        xs_clipped = torch.clamp(xs, lo, hi)
        
        pr = torch.histc(xr_clipped, bins=n_bins, min=lo, max=hi)
        ps = torch.histc(xs_clipped, bins=n_bins, min=lo, max=hi)

        pr = pr / (pr.sum() + 1e-8)
        ps = ps / (ps.sum() + 1e-8)
        js = js_divergence_native(pr, ps)
        js_scores.append(js)

    if not js_scores:
        return {"mean": float('nan'), "per_var": []}
        
    js_stack = torch.stack(js_scores)
    return {"mean": js_stack.mean().item(), "per_var": js_stack.cpu().tolist()}


def js_divergence_static(
        W_real: torch.Tensor,
        W_syn: torch.Tensor,
        static_types: list, # List of tuples: (type, n_cats, dim)
        n_bins: int = 20
    ) -> Dict[str, float]:
    """
    Compute the Jensen–Shannon divergence (distance) between real and synthetic static variables.

    Each variable is compared marginally, using the information in static_types:
      - type == 'real'     → histogram-based JS
      - type == 'cat'      → categorical frequency-based JS
      - type == 'ordinal'  → histogram-based JS (treated as ordered numeric)

    Args:
        W_real: real static features (N_real, D)
        W_syn:  synthetic static features (N_syn, D)
        static_types: dict of [type, feature_dim, num_categories]
        n_bins: number of bins for continuous (real/ordinal) features
        normalize: whether to normalize histograms to probabilities

    Returns:
        dict with:
            - JS_mean: mean JS divergence across variables
            - JS_per_var: per-variable JS divergences
    """
    N_vars = W_real.shape[1]
    js_scores = []
    
    for j, (feat_type, num_cat, feat_dim) in enumerate(static_types):
        x_r = W_real[:, j]
        x_s = W_syn[:, j]
        
        if feat_type in ("real", "ordinal"):
            # --- Histogram-based JS (Continuous/Ordinal) ---
            combined = torch.cat([x_r, x_s])
            lo, hi = torch.min(combined), torch.max(combined)
            pr = torch.histc(x_r, bins=n_bins, min=lo, max=hi)
            ps = torch.histc(x_s, bins=n_bins, min=lo, max=hi)
            
        elif feat_type == "cat":
            # --- Categorical JS ---
            num_cat = int(num_cat)
            pr = torch.bincount(x_r.long(), minlength=num_cat).float()
            ps = torch.bincount(x_s.long(), minlength=num_cat).float()
            
        else:
            raise ValueError(f"Unsupported feature type '{feat_type}' at index {j}")

        pr = pr / (pr.sum() + 1e-8)
        ps = ps / (ps.sum() + 1e-8)
        js = js_divergence_native(pr, ps)
        js_scores.append(js)

    js_stack = torch.stack(js_scores)
    per_var_dict = {f"var_{i}": val.item() for i, val in enumerate(js_stack)}
    
    return {
        "mean": js_stack.mean().item(), 
        "per_var": per_var_dict
    }


# -----------------------------------------------------------------------------
# Correlations 
# -----------------------------------------------------------------------------

# Static-Static

def pairwise_correlation_static(W_real: torch.Tensor, 
                                W_syn: torch.Tensor, 
                                static_types: list, # List of tuples: (type, n_cats, dim), 
                                return_mat: bool = False,
                                cached_real_data = None
                                ):
    """
    Computes correlation metrics on GPU, respecting variable types:
    - 'real': Standard Pearson correlation on values.
    - 'ordinal': Spearman correlation (via ranking) matching rank(method='dense').
    - 'cat': Pearson correlation on integer codes (Label Encoding) (no onehot). 
    """
    W_real = W_real.clone()
    W_syn = W_syn.clone()

    if cached_real_data is not None:
        corr_real = cached_real_data.clone()
    else: 
        for i, (dtype, _, _) in enumerate(static_types):
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

    
    # --- Preprocessing based on types ---
    for i, (dtype, _, _) in enumerate(static_types):
        if dtype == 'ordinal':
            _, W_syn[:, i]  = torch.unique(W_syn[:, i],  return_inverse=True)
            W_syn[:, i]  = W_syn[:, i].float()
        elif dtype == 'cat':
            W_syn[:, i]  = W_syn[:, i].float()

    # --- Standardize (Z-score) ---
    # (X - mean) / std
    syn_mean = W_syn.mean(dim=0, keepdim=True)
    syn_std = W_syn.std(dim=0, keepdim=True) + 1e-8
    Z_syn = (W_syn - syn_mean) / syn_std

    # --- Compute Correlation Matrices (C = Z.T @ Z / (N-1)) ---
    N_s = W_syn.size(0)
    corr_syn = (Z_syn.T @ Z_syn) / (N_s - 1)

    # --- Metrics ---
    diff = corr_real - corr_syn
    fro_norm = torch.norm(diff, p='fro')
    rmse = torch.sqrt(torch.mean(diff**2))
    
    # Structural correlation (correlation of the correlations)
    c_real_flat = corr_real.flatten()
    c_syn_flat = corr_syn.flatten()
    v_r = c_real_flat - c_real_flat.mean()
    v_s = c_syn_flat - c_syn_flat.mean()
    denom = torch.norm(v_r) * torch.norm(v_s) + 1e-8
    struct_corr = torch.dot(v_r, v_s) / denom

    metrics = {
        "Frobenius_norm": fro_norm.item(),
        "RMSE": rmse.item(),
        "structural_corr": struct_corr.item()
    }
    
    if return_mat:
        return corr_real, corr_syn, metrics
    else:
        return metrics


# Longi-Longi
def pairwise_correlation_longi(
        x_real: torch.Tensor, m_real: torch.Tensor,
        x_syn: torch.Tensor,  m_syn: torch.Tensor, 
        return_mat: bool = False, 
        cached_real_data: Optional[torch.Tensor] = None
    ) -> Union[Dict[str, float], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
    """
    Computes Longitudinal Correlation matrices for Real and Synthetic data,
    handling missing values correctly via masking.
    
    Args:
        x_real, m_real: (N, T, F) Real data and mask (0=missing).
        x_syn, m_syn: (N, T, F) Synthetic data and mask.
        return_mat: If True, returns the matrices along with metrics.
        cached_real_data: Optional pre-computed real correlation matrix.

    Returns:
        Dictionary of metrics (Frobenius, RMSE, Structural Correlation).
    """
    N_r, T, F = x_real.shape
    N_s = x_syn.shape[0]

    xr_flat = x_real.reshape(N_r * T, F)
    mr_flat = m_real.reshape(N_r * T, F)    
    xs_flat = x_syn.reshape(N_s * T, F)
    ms_flat = m_syn.reshape(N_s * T, F)

    corr_syn = compute_masked_correlation_matrix(xs_flat, ms_flat)
    if cached_real_data is not None:
        corr_real = cached_real_data.clone()
    else:
        corr_real = compute_masked_correlation_matrix(xr_flat, mr_flat)

    diff = corr_real - corr_syn
    valid_mask = ~torch.isnan(diff)
    if valid_mask.sum() == 0:
        metrics = {
            "Frobenius_norm": float('nan'),
            "RMSE": float('nan'),
            "structural_corr": float('nan')
        }
        if return_mat: return corr_real, corr_syn, metrics
        return metrics

    valid_diff = diff[valid_mask]
    
    # Frobenius Norm (on valid entries)
    fro_norm = torch.norm(valid_diff, p='fro')
    # RMSE
    rmse = torch.sqrt(torch.mean(valid_diff ** 2))
    # Structural Correlation
    c_real_valid = corr_real[valid_mask]
    c_syn_valid = corr_syn[valid_mask]
    v_r = c_real_valid - c_real_valid.mean()
    v_s = c_syn_valid - c_syn_valid.mean()
    denom = torch.norm(v_r) * torch.norm(v_s) + 1e-8
    struct_corr = torch.dot(v_r, v_s) / denom

    metrics = {
        "Frobenius_norm": fro_norm.item(),
        "RMSE": rmse.item(),
        "structural_corr": struct_corr.item()
    }

    if return_mat:
        return corr_real, corr_syn, metrics
    else:
        return metrics


def pairwise_correlation_global(
        x_real: torch.Tensor, m_real: torch.Tensor, W_real: torch.Tensor, 
        x_syn: torch.Tensor,  m_syn: torch.Tensor, W_syn: torch.Tensor, 
        return_mat: bool = False, 
        cached_real_data: Optional[torch.Tensor] = None
    ) -> Union[Dict[str, float], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:

    x_real_aug, m_real_aug = augment_with_statics(x_real, m_real, W_real, use_nans_for_statics=False) # Shape: (N, T, F_total)
    x_syn_aug, m_syn_aug  = augment_with_statics(x_syn, m_syn, W_syn, use_nans_for_statics=False)

    return pairwise_correlation_longi(x_real_aug, m_real_aug, x_syn_aug, m_syn_aug, return_mat=return_mat, cached_real_data=cached_real_data)


# -----------------------------------------------------------------------------
# Poisson process  
# -----------------------------------------------------------------------------

def kl_divergence_event_rate(
    x_real, m_real, T_real, 
    x_syn, m_syn, T_syn, 
    bins=80, eps=1e-8, 
    cached_real_data=None
):
    device = x_real.device
    N_r = x_real.shape[0]
    N_s = x_syn.shape[0]

    if torch.isnan(T_syn).any() or torch.isinf(T_syn).any():
        T_syn = torch.nan_to_num(T_syn, nan=0.0, posinf=0.0, neginf=0.0)

    mask_s = (m_syn.sum(dim=-1) > 0) 
    if T_syn.dim() == 1:
        T_syn_exp = T_syn.unsqueeze(0).expand(N_s, -1)
    else:
        T_syn_exp = T_syn
    
    events_s = T_syn_exp[mask_s]
    sample_indices_s = torch.arange(N_s, device=device).unsqueeze(1).expand_as(mask_s)[mask_s]
    counts_s = mask_s.sum(dim=1).float()

    if cached_real_data is not None:
        events_r, sample_indices_r, counts_r, deltas_r, indices_r = [t.clone() for t in cached_real_data]
    else:
        mask_r = (m_real.sum(dim=-1) > 0)
        if T_real.dim() == 1:
            T_real_exp = T_real.unsqueeze(0).expand(N_r, -1)
        else:
            T_real_exp = T_real

        events_r = T_real_exp[mask_r]
        sample_indices_r = torch.arange(N_r, device=device).unsqueeze(1).expand_as(mask_r)[mask_r]
        counts_r = mask_r.sum(dim=1).float()

        # detla version
        mask_r_bool = (m_real.sum(dim=-1) > 0)
        deltas_r, indices_r = get_deltas_and_indices(mask_r_bool, T_real, N_r)

    # --- Compute Metrics ---
    t_min = 0.0
    max_r = events_r.max() if events_r.numel() > 0 else torch.tensor(1.0, device=device)
    max_s = events_s.max() if events_s.numel() > 0 else torch.tensor(1.0, device=device)
    t_max = max(max_r, max_s).item() * 1.0001 + 1e-6
    p = estimate_pdf_vectorized(events_r, sample_indices_r, N_r, bins, t_min, t_max, eps)
    q = estimate_pdf_vectorized(events_s, sample_indices_s, N_s, bins, t_min, t_max, eps)
    kl_score = torch.sum(p * torch.log((p + eps) / (q + eps)))

    # Detla version
    mask_s_bool = (m_syn.sum(dim=-1) > 0)
    deltas_s, indices_s = get_deltas_and_indices(mask_s_bool, T_syn, N_s)
    if deltas_r is None or deltas_s is None:
        return {"KL_Delta_T": float('nan')}
    t_max_delta = torch.quantile(deltas_r, 0.99).item()
    if t_max_delta <= t_min: 
        t_max_delta = deltas_r.max().item() + eps # Fallback if quantile fails
    deltas_r = deltas_r.clamp(max=t_max_delta)
    deltas_s = deltas_s.clamp(max=t_max_delta)
    p = estimate_pdf_vectorized(deltas_r, indices_r, N_r, bins, t_min, t_max_delta, eps)
    q = estimate_pdf_vectorized(deltas_s, indices_s, N_s, bins, t_min, t_max_delta, eps)
    kl_delta = torch.sum(p * torch.log((p + eps) / (q + eps)))

    # --- EOS: extract last observed time per patient ---
    def get_last_event_per_patient(events, sample_indices, N, device):
        """Extract the last event time per patient from flattened events."""
        T_eos = torch.zeros(N, device=device)
        for i in range(N):
            mask_i = (sample_indices == i)
            if mask_i.any():
                T_eos[i] = events[mask_i].max()
        return T_eos  # (N,)

    # EOS times = last observed time per patient
    T_real_eos = get_last_event_per_patient(events_r, sample_indices_r, N_r, device)
    T_syn_eos  = get_last_event_per_patient(events_s, sample_indices_s, N_s, device)

    idx_r_eos = torch.arange(N_r, device=device)
    idx_s_eos = torch.arange(N_s, device=device)

    t_max_eos = max(T_real_eos.max(), T_syn_eos.max()).item() * 1.0001 + 1e-6
    p_eos = estimate_pdf_vectorized(T_real_eos, idx_r_eos, N_r, bins, t_min, t_max_eos, eps)
    q_eos = estimate_pdf_vectorized(T_syn_eos,  idx_s_eos, N_s, bins, t_min, t_max_eos, eps)
    kl_eos = torch.sum(p_eos * torch.log((p_eos + eps) / (q_eos + eps)))

    return {
        "KL_event": kl_score.item(),
        "KL_delta": kl_delta.item(),
        "KL_eos": kl_eos.item(),
        "mean_N_obs_syn": counts_s.mean().item(),
        "mean_N_obs_real": counts_r.mean().item(),
        "std_N_obs_syn": counts_s.std().item(),
        "std_N_obs_real": counts_r.std().item()
    }

# -----------------------------------------------------------------------------
# The classifier 2-sample test (C2ST)
# -----------------------------------------------------------------------------


def discriminative_score(
    x_real, m_real, T_real, x_syn, m_syn, T_syn, 
    W_real=None, W_syn=None, 
    classifier_type: str = 'S4', 
    filling_method: str = 'last', 
    epochs: int = 100, 
    device=None, 
    verbose=True, 
    cached_real_data=None, # New parameter to pass pre-processed real data
    **kwargs
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if cached_real_data is not None:
        x_real_aug, m_real_aug = [t.clone() for t in cached_real_data]
    else:
        xr_f, mr_f, _ = fill_missing_values_long(x_real, m_real, T_real, filling_type=filling_method)
        x_real_aug, m_real_aug = augment_with_statics(xr_f, mr_f, W_real, use_nans_for_statics=False)

    xs_f, ms_f, _ = fill_missing_values_long(x_syn, m_syn, T_syn, filling_type=filling_method)
    x_syn_aug, m_syn_aug = augment_with_statics(xs_f, ms_f, W_syn, use_nans_for_statics=False)

    x_concat = torch.cat([x_syn_aug, x_real_aug], dim=0).to(device)
    mask_concat = torch.cat([m_syn_aug, m_real_aug], dim=0).to(device)
    y_concat = torch.cat([
        torch.zeros(x_syn_aug.size(0), device=device), 
        torch.ones(x_real_aug.size(0), device=device)
    ]).long()

    N = x_concat.size(0)
    perm = torch.randperm(N, device=device)
    x_concat, mask_concat, y_concat = x_concat[perm], mask_concat[perm], y_concat[perm]

    train_ratio, val_ratio = kwargs.get("train_ratio", 0.7), kwargs.get("val_ratio", 0.1)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    if N - n_train - n_val <= 0:
        n_val = N - n_train - 1 # force at least 1 test sample

    dataset = TensorDataset(x_concat, mask_concat, y_concat)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, range(n_train)), 
                              batch_size=kwargs.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, range(n_train, n_train + n_val)), 
                            batch_size=kwargs.get("batch_size", 32))
    test_loader = DataLoader(torch.utils.data.Subset(dataset, range(n_train + n_val, N)), 
                             batch_size=kwargs.get("batch_size", 32))

    d_input = x_concat.shape[2]
    if classifier_type == "S4":
        model = S4Classifier(d_input=d_input, d_model=kwargs.get("d_model", 64), 
                             n_layers=kwargs.get("n_layers", 2), dropout=0.2).to(device)
    elif classifier_type == "LSTM":
        model = LSTMClassifier(d_input=d_input, d_model=kwargs.get("d_model", 64)).to(device)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-3), weight_decay=kwargs.get("weight_decay", 0.001))
    criterion = torch.nn.CrossEntropyLoss()

    # --- Optimized Training Loop ---
    best_val_loss = float("inf")
    best_state = None
    patience, pat = 15, 0

    for epoch in range(epochs):
        model.train()
        for xb, mb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(xb, mb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        va_loss, va_acc = evaluate_classifier_native(model, val_loader, criterion, device, use_amp)
        # print("va_loss:", va_loss)
        # print("va_acc:", va_acc)
        
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    te_loss, te_acc, te_probs, te_labels = evaluate_classifier_native(model, test_loader, criterion, device, use_amp, return_probs=True)
    def _safe_auc(y_true, y_prob):
        try:
            return roc_auc_score(y_true, y_prob)
        except Exception:
            return float("nan")
            
    return {
        "val_acc": float(va_acc),
        "test_acc": float(te_acc),
        "test_auroc": _safe_auc(te_labels, te_probs[:, 1]) # Index 1 for 'real' class
    }


def evaluate_classifier_native(model, loader, criterion, device, use_amp, return_probs=False):
    model.eval()
    losses, corrects, total = 0, 0, 0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for xb, mb, yb in loader:
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(xb, mb)
                loss = criterion(out, yb)
            
            losses += loss.item() * xb.size(0)
            corrects += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
            
            if return_probs:
                all_probs.append(torch.softmax(out, dim=1).cpu())
                all_labels.append(yb.cpu())

    avg_loss = losses / total
    avg_acc = corrects / total
    
    if return_probs:
        return avg_loss, avg_acc, torch.cat(all_probs), torch.cat(all_labels)
    return avg_loss, avg_acc


###############################################################################
# S4Classifier for Binary Classification ("real" vs. "synthetic")
###############################################################################

# Recommended Hyperparameters for Discriminative Score:
# d_model: 64 (Instead of 128 to prevent memorization)
# n_layers: 2 or 4
# dropout: 0.2 (Higher to force the model to learn general features)

class S4Classifier(torch.nn.Module):
    def __init__(self, d_input, d_model=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = torch.nn.Linear(d_input, d_model)
        self.s4_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True))
            self.norms.append(torch.nn.LayerNorm(d_model))
        self.decoder = torch.nn.Linear(d_model, 2)

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T, D)
        x = self.encoder(x)
        x = x.transpose(-1, -2) # (B, D, T) for S4D
        
        for layer, norm in zip(self.s4_layers, self.norms):
            z = norm(x.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            x = x + z 
        x = x.transpose(-1, -2) # back to (B, T, D)
        
        # Global Average Pooling restricted to masked (valid) time steps. This is CRITICAL: it prevents the model from "cheating" using padding
        mask_val = mask[:, :, 0].unsqueeze(-1) # (B, T, 1)
        x = x * mask_val
        pooled = x.sum(dim=1) / (mask_val.sum(dim=1) + 1e-8)
        return self.decoder(pooled)
    
###############################################################################
# LSTM Classifier
###############################################################################

class LSTMClassifier(nn.Module):
    def __init__(self, d_input, d_model=64, n_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x, mask=None):
        out, (h, c) = self.lstm(x)
        pooled = h[-1]          # (B, d_model)
        logits = self.fc(pooled)
        return logits





"""
    Plot functions 

"""

# -----------------------------------------------------------------------------
# Plot events 
# -----------------------------------------------------------------------------

def plot_intervals_poisson(x_real, m_real, T_real, x_syn, m_syn, T_syn, color_real=None, color_syn=None):
    synthetic_process = []
    real_process = []
    real_intervals = []
    synthetic_intervals = []
    for n in range(x_real.shape[0]):
        real_process.append(T_real[m_real[n, :, 0].bool()])
        real_intervals.append(torch.diff(real_process[n]))
    for n in range(x_syn.shape[0]):
        synthetic_process.append(T_syn[m_syn[n, :, 0].bool()])
        synthetic_intervals.append(torch.diff(synthetic_process[n]))
    real_intervals = torch.cat(real_intervals)
    synthetic_intervals = torch.cat(synthetic_intervals)
    plt.figure(figsize=(5, 5))
    plt.hist(real_intervals, bins=30, alpha=0.5, label='Real', color=color_real)
    plt.hist(synthetic_intervals, bins=30, alpha=0.5, label='Synthetic', color=color_syn)
    plt.legend()
    plt.title('Distribution of Intervals')
    plt.show()

def _piece_wise_constant_interpolation(new_grid, init_grid, init_vals):
    idx = torch.searchsorted(init_grid, new_grid)
    new_vals = init_vals[idx]
    return new_vals

def plot_events_counts(x_real, m_real, T_real, x_syn, m_syn, T_syn, grid_plot, color_real=None, color_syn=None):
    plt.figure(figsize=(7, 5))
    synthetic_process = []
    real_process = []
    real_process_pwcst = []
    for i in range(x_real.shape[0]):
        real_process.append(T_real[m_real[i, :, 0].bool()])
        piecewise_cst_i = _piece_wise_constant_interpolation(grid_plot, real_process[i], np.arange(0, len(real_process[i]) + 1))
        plt.plot(grid_plot, piecewise_cst_i, color=color_real, alpha=0.3, linewidth=1)
        real_process_pwcst.append(piecewise_cst_i)
    synth_process_pwcst = []
    for i in range(x_syn.shape[0]):
        synthetic_process.append(T_syn[m_syn[i, :, 0].bool()])
        piecewise_cst_i = _piece_wise_constant_interpolation(grid_plot, synthetic_process[i], np.arange(0, len(synthetic_process[i]) + 1))
        plt.plot(grid_plot, piecewise_cst_i, color=color_syn, alpha=0.3, linewidth=1)
        synth_process_pwcst.append(piecewise_cst_i)
    plt.plot(grid_plot, torch.mean(torch.Tensor(real_process_pwcst), dim=0), color=color_real, label='Real Process', linewidth=2)
    plt.plot(grid_plot, torch.mean(torch.Tensor(synth_process_pwcst), dim=0), color=color_syn, label='Synthetic Process', linewidth=2)
    plt.title('Counting Processes')
    plt.xlabel('Time')
    plt.ylabel('Number of Events')
    plt.legend()
    plt.show() 


# -----------------------------------------------------------------------------
# Plot correlation matrices
# -----------------------------------------------------------------------------

def plot_corr_matrices(corr_real, corr_syn):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr_real, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Real Static Correlation")
    sns.heatmap(corr_syn, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Synthetic Static Correlation")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Plot marginal distribution static
# -----------------------------------------------------------------------------

def plot_static_marginal_distribution(W_real, W_syn, static_types, variable_names=None):
    num_vars = W_real.shape[1]
    variable_names = variable_names or [f'Var {i+1}' for i in range(num_vars)]

    n_cols = 3  # Three plots per row
    n_rows = int(np.ceil(num_vars / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing

    for i in range(num_vars):
        var_type, num_categories, _ = static_types[i]
        real_vals, synth_vals = W_real[:, i].numpy(), W_syn[:, i].numpy()
        ax = axes[i]

        if var_type == 'real':  
            sns.violinplot(data=[real_vals, synth_vals], ax=ax, scale="width", split=True, gap=.1, inner="quart")

            ax.scatter([0, 1], [np.mean(real_vals), np.mean(synth_vals)], color='black', label='Mean')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Real', 'Synthetic'])
            ax.set_ylabel("Value")
            ax.set_title(f"{variable_names[i]} (real)")
            ax.legend()

        elif var_type == 'cat':  
            categories = np.arange(num_categories)
            real_counts = np.bincount(real_vals.astype(int), minlength=num_categories)
            synth_counts = np.bincount(synth_vals.astype(int), minlength=num_categories)

            df = pd.DataFrame({'Category': categories, 'Real': real_counts, 'Synthetic': synth_counts}).melt(id_vars="Category")
            sns.barplot(x='Category', y='value', hue='variable', data=df, ax=ax)
            ax.set_ylabel("Count")
            ax.set_title(f"{variable_names[i]} (categorical)")

        elif var_type == 'ordinal':  
            real_cdf = np.sort(real_vals)
            synth_cdf = np.sort(synth_vals)
            real_cdf_vals = np.arange(1, len(real_vals)+1) / len(real_vals)
            synth_cdf_vals = np.arange(1, len(synth_vals)+1) / len(synth_vals)

            ax.plot(real_cdf, real_cdf_vals, label='Real', marker='o', linestyle='-')
            ax.plot(synth_cdf, synth_cdf_vals, label='Synthetic', marker='s', linestyle='--')
            ax.set_ylabel("Cumulative Probability")
            ax.set_title(f"{variable_names[i]} (ordinal)")
            ax.legend()

    # Hide empty subplots if num_vars isn't a multiple of 3
    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Plot marginal distribution Longi
# -----------------------------------------------------------------------------

def plot_longi_marginal_distribution_similarity_score(x_real, m_real, x_syn, m_syn, bins=10, color_real=None, color_synth=None):
    """
    Computes the marginal distribution similarity between real and synthetic data
    and plots their density estimates for each feature.
    
    Parameters:
        - bins (int): Number of histogram bins to use for density estimation.
    
    Returns:
        - marginal_score (float): The average marginal distribution discrepancy. A value of 0 indicates a perfect match.
    """
    n_features = x_real.shape[-1]
    feature_scores = []
    
    rows = (n_features + 3) // 4  # Ensure 4 features per row
    fig, axes = plt.subplots(rows, min(4, n_features), figsize=(20, 5 * rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for feature_idx in range(n_features):
        real_feature = x_real[..., feature_idx].flatten()
        real_mask_feat = m_real[..., feature_idx].flatten()
        real_feature = real_feature[real_mask_feat.bool()]
        
        synth_feature = x_syn[..., feature_idx].flatten()
        synth_mask_feat = m_syn[..., feature_idx].flatten()
        synth_feature = synth_feature[synth_mask_feat.bool()]
        
        # Find the common range for binning
        feature_min = torch.min(torch.cat([real_feature, synth_feature])).item()
        feature_max = torch.max(torch.cat([real_feature, synth_feature])).item()
        
        # Compute histograms
        real_hist = torch.histc(real_feature, bins=bins, min=feature_min, max=feature_max)
        synth_hist = torch.histc(synth_feature, bins=bins, min=feature_min, max=feature_max)
        
        # Normalize to get probability densities
        real_density = real_hist / torch.sum(real_hist)
        synth_density = synth_hist / torch.sum(synth_hist)
        
        # Compute marginal distribution difference
        diff = torch.abs(real_density - synth_density)
        feature_score = torch.mean(diff)
        feature_scores.append(feature_score)
        
        # Plot density estimates
        sns.kdeplot(real_feature.numpy(), label='Real', color=color_real, fill=True, alpha=0.5, ax=axes[feature_idx])
        sns.kdeplot(synth_feature.numpy(), label='Synthetic', color=color_synth, fill=True, alpha=0.5, ax=axes[feature_idx])
        
        axes[feature_idx].set_title(f'Feature {feature_idx + 1}')
        axes[feature_idx].legend()
        axes[feature_idx].set_xlabel("Value")
        axes[feature_idx].set_ylabel("Density")
    
    # Remove empty subplots if features < total grid slots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    marginal_score = torch.stack(feature_scores).mean().item()
    return {"Marginal score longi": marginal_score}


# -----------------------------------------------------------------------------
# Plot tSNE Longi
# -----------------------------------------------------------------------------

def plot_tsne(x_real, m_real, T_real, x_syn, m_syn, T_syn, perplexity=30, random_state=42, color_real=None, color_syn=None):
    """
    Plots t-SNE visualization of real and synthetic data.
    
    Parameters:
        - perplexity (int): t-SNE perplexity parameter.
        - random_state (int): Random seed for reproducibility.
    """

    real_data, _, _ = fill_missing_values_long(x_real, m_real, T_real, filling_type='zero', tol=1e-7)
    synthetic_data, _, _ = fill_missing_values_long(x_syn, m_syn, T_syn, filling_type='zero', tol=1e-7)

    # Reshape from (N, T, m) to (N, T*m)
    real_data_flat = real_data.reshape(real_data.shape[0], -1)
    synthetic_data_flat = synthetic_data.reshape(synthetic_data.shape[0], -1)
    combined_data = np.vstack([real_data_flat, synthetic_data_flat])

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embedded_data = tsne.fit_transform(combined_data)
    real_embedded = embedded_data[:len(real_data_flat)]
    synthetic_embedded = embedded_data[len(real_data_flat):]

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(real_embedded[:, 0], real_embedded[:, 1], color=color_real, alpha=0.7, label='Real Data')
    plt.scatter(synthetic_embedded[:, 0], synthetic_embedded[:, 1], color=color_syn, alpha=0.7, label='Synthetic Data')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Visualization of Real vs. Synthetic Data")
    plt.legend()
    plt.grid(True)
    plt.show()

