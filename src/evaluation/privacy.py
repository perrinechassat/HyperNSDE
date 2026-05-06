
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, List, Literal, Union
import torch
from src.evaluation.utils import fast_pdist_long, fast_pdist_static, fast_signature_dist_long

"""
-----------------------------------------------------------------------------
                                Privacy metrics
-----------------------------------------------------------------------------
"""

# -----------------------------------------------------------------------------
# NNAA helper
# -----------------------------------------------------------------------------

def _adversarial_accuracy(D_rs, D_rr, D_sr, D_ss):
    """Compute adversarial accuracy for a given modality or fused distances."""
    d_rs = D_rs.min(dim=1).values
    d_rr = D_rr.min(dim=1).values
    d_sr = D_sr.min(dim=1).values
    d_ss = D_ss.min(dim=1).values
    return 0.5 * ((d_rs > d_rr).float().mean() + (d_sr > d_ss).float().mean())


# -----------------------------------------------------------------------------
# Nearest Neighbor Adversarial Accuracy (NNAA) (Multimodal)
# -----------------------------------------------------------------------------

def nnaa(
        x_r_train, m_r_train, W_r_train,
        x_r_test,  m_r_test,  W_r_test,
        x_s,       m_s,       W_s,
        T,
        weights=(0.5, 0.5),
        dist_type='signature',
        cached_real_data=None, # Optional: Pass pre-computed Real-Real distances here
        cached_synthetic_data=None, # Optional: Pass pre-computed Syn-Syn and Real_Syn distances here
        filling_type: str = "last", 
        kernel = 'rbf',
        dyadic_order = 3,
        sigma = 0.5, 
        lead_lag = False
    ):

    if dist_type == 'signature':
        fn_dist = lambda x1, x2, m1, m2, T: fast_signature_dist_long(x1, x2, m1, m2, T, kernel=kernel, dyadic_order=dyadic_order, sigma=sigma, lead_lag=lead_lag, squared=False)
    else:
        fn_dist = lambda x1, x2, m1, m2, T: fast_pdist_long(x1, x2, squared=False)

    # Weights normalization
    w_long, w_stat = weights
    total_w = w_long + w_stat
    if total_w == 0: raise ValueError("Weights cannot sum to zero")
    w_long /= total_w
    w_stat /= total_w

    res = {}
    keys = ["rt_s", "rt_rt", "s_rt", "s_s", "rtr_s", "rtr_rtr", "s_rtr"]
    
    # --- Compute Longitudinal Distances ---
    D_l = {}
    if w_long > 0:
        # Syn vs Real
        if cached_synthetic_data:
            D_l["rt_s"]  = cached_synthetic_data["L_rt_s"].clone()
            D_l["rtr_s"] = cached_synthetic_data["L_rtr_s"].clone()
        else:
            D_l["rt_s"]  = fn_dist(x_r_test, x_s, m_r_test, m_s, T)
            D_l["rtr_s"] = fn_dist(x_r_train, x_s, m_r_train, m_s, T)
        D_l["s_s"]   = fn_dist(x_s, x_s, m_s, m_s, T)
        
        # Real vs Real (Potentially Cached)
        if cached_real_data:
            D_l["rt_rt"]   = cached_real_data["L_rt_rt"].clone()
            D_l["rtr_rtr"] = cached_real_data["L_rtr_rtr"].clone()
        else:
            D_l["rt_rt"]   = fn_dist(x_r_test, x_r_test, m_r_test, m_r_test, T)
            D_l["rtr_rtr"] = fn_dist(x_r_train, x_r_train, m_r_train, m_r_train, T)

        # Transposes for symmetry
        D_l["s_rt"]  = D_l["rt_s"].t()
        D_l["s_rtr"] = D_l["rtr_s"].t()

        all_L = torch.cat([D_l[k].flatten() for k in keys])

        D_l["s_s"].fill_diagonal_(float('inf'))
        D_l["rtr_rtr"].fill_diagonal_(float('inf'))
        D_l["rt_rt"].fill_diagonal_(float('inf'))
        
        # Compute Single Modality NNAA
        aa_L_test  = _adversarial_accuracy(D_l["rt_s"],  D_l["rt_rt"],  D_l["s_rt"],  D_l["s_s"])
        aa_L_train = _adversarial_accuracy(D_l["rtr_s"], D_l["rtr_rtr"], D_l["s_rtr"], D_l["s_s"])
        
        res["NNAA_longi"] = (aa_L_test - aa_L_train).item()
        res["AA_longi_test"] = aa_L_test.item()
        res["AA_longi_train"] = aa_L_train.item()

    # --- Compute Static Distances ---
    D_s = {}
    if w_stat > 0:
        # Syn vs Real
        if cached_synthetic_data:
            D_s["rt_s"]  = cached_synthetic_data["S_rt_s"].clone()
            D_s["rtr_s"] = cached_synthetic_data["S_rtr_s"].clone()
        else:
            D_s["rt_s"]  = fast_pdist_static(W_r_test, W_s)
            D_s["rtr_s"] = fast_pdist_static(W_r_train, W_s)
        D_s["s_s"] = fast_pdist_static(W_s, W_s)

        # Real vs Real (Potentially Cached)
        if cached_real_data:
            D_s["rt_rt"]   = cached_real_data["S_rt_rt"].clone()
            D_s["rtr_rtr"] = cached_real_data["S_rtr_rtr"].clone()
        else:
            D_s["rt_rt"]   = fast_pdist_static(W_r_test, W_r_test)
            D_s["rtr_rtr"] = fast_pdist_static(W_r_train, W_r_train)

        # Transposes
        D_s["s_rt"]  = D_s["rt_s"].t()
        D_s["s_rtr"] = D_s["rtr_s"].t()

        all_S = torch.cat([D_s[k].flatten() for k in keys])

        D_s["s_s"].fill_diagonal_(float('inf'))
        D_s["rtr_rtr"].fill_diagonal_(float('inf'))
        D_s["rt_rt"].fill_diagonal_(float('inf'))

        # Compute Single Modality NNAA
        aa_S_test  = _adversarial_accuracy(D_s["rt_s"],  D_s["rt_rt"],  D_s["s_rt"],  D_s["s_s"])
        aa_S_train = _adversarial_accuracy(D_s["rtr_s"], D_s["rtr_rtr"], D_s["s_rtr"], D_s["s_s"])
        
        res["NNAA_stat"] = (aa_S_test - aa_S_train).item()
        res["AA_stat_test"] = aa_S_test.item()
        res["AA_stat_train"] = aa_S_train.item()

    # --- Compute Multimodal NNAA ---
    if w_long > 0 and w_stat > 0:
        keys = ["rt_s", "rt_rt", "s_rt", "s_s", "rtr_s", "rtr_rtr", "s_rtr"]
        D_Fused = {}
        
        mean_L, std_L = all_L.mean(), all_L.std() + 1e-8
        mean_S, std_S = all_S.mean(), all_S.std() + 1e-8
        for k in keys:
            z_L = (D_l[k] - mean_L) / std_L
            z_S = (D_s[k] - mean_S) / std_S
            D_Fused[k] = (w_long * z_L) + (w_stat * z_S)
            
        aa_test  = _adversarial_accuracy(D_Fused["rt_s"],  D_Fused["rt_rt"],  D_Fused["s_rt"],  D_Fused["s_s"])
        aa_train = _adversarial_accuracy(D_Fused["rtr_s"], D_Fused["rtr_rtr"], D_Fused["s_rtr"], D_Fused["s_s"])

        res["NNAA"] = (aa_test - aa_train).item()
        res["AA_test"] = aa_test.item()
        res["AA_train"] = aa_train.item()

    return res



# ---------------------------------------------------------------------
# MIR helpers
# ---------------------------------------------------------------------

def _agg_k_optimized(dmat: torch.Tensor, k: int, agg: str) -> torch.Tensor:
    """Aggregate k nearest distances per row."""
    if k <= 1:
        return dmat.min(dim=1).values
    k = min(k, dmat.size(1))
    vals, _ = torch.topk(dmat, k=k, dim=1, largest=False)
    
    if agg == "mean":
        return vals.mean(dim=1)
    elif agg == "min":
        return vals[:, 0]
    raise ValueError("agg must be 'min' or 'mean'")

def _auc_balanced_acc_metrics(s_tr: torch.Tensor, s_te: torch.Tensor) -> Tuple[float, float, float]:
    """
    Computes AUC (via Rank-Sum) and Balanced Accuracy (via Youden's J)
    WITHOUT creating O(N^2) matrices.
    """
    # Fast AUC (Wilcoxon Rank-Sum)
    n_pos = s_tr.numel()
    n_neg = s_te.numel()
    
    combined = torch.cat([s_tr, s_te])
    # argsort twice gives ranks
    ranks = torch.argsort(torch.argsort(combined)).float() + 1 
    
    rank_sum_pos = ranks[:n_pos].sum()
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2)
    auc = u_stat / (n_pos * n_neg)

    # Fast Balanced Accuracy (Vectorized Threshold search)
    scores = combined
    labels = torch.cat([torch.ones(n_pos, device=scores.device), 
                        torch.zeros(n_neg, device=scores.device)])

    # Sort by score
    sorted_scores, idx = torch.sort(scores)
    sorted_labels = labels[idx]
    
    # Total positives and negatives
    P = n_pos
    N = n_neg
    cum_pos = torch.cumsum(sorted_labels, dim=0)
    cum_neg = torch.cumsum(1 - sorted_labels, dim=0)
    tpr = (P - cum_pos) / (P + 1e-8)
    tnr = cum_neg / (N + 1e-8)
    
    # Balanced Acc = (TPR + TNR) / 2
    bal_accs = (tpr + tnr) / 2
    
    best_idx = torch.argmax(bal_accs)
    best_bal_acc = bal_accs[best_idx].item()
    best_thr = sorted_scores[best_idx].item()

    return auc.item(), best_bal_acc, best_thr


# ---------------------------------------------------------------------
# Membership Inference Risk (MIR) (multimodal) 
# ---------------------------------------------------------------------

def membership_inference_risk(
        x_train, m_train, W_train,
        x_test,  m_test,  W_test,
        x_syn,   m_syn,   W_syn,
        T,
        weights: Tuple[float, float] = (0.5, 0.5),
        k: int = 1,
        agg: str = "min", 
        dist_type='signature',
        cached_synthetic_data=None, 
        filling_type: str = "last", # Optional: Pass pre-computed Syn-Syn and Real_Syn distances here
        kernel = 'rbf',
        dyadic_order = 3,
        sigma = 0.5, 
        lead_lag = False
    ) -> Dict[str, float]:
    """
    Membership Inference Risk via multimodal fused distances (no classifier).
        - Attack score per real sample = negative aggregated k-NN distance to synthetic set.
        - Compute AUROC for TRAIN (pos) vs TEST (neg), MIR_risk = 2*|AUC-0.5|.
    Returns fused MIR plus per-modality diagnostics.
    """
     
    if dist_type == 'signature':
        fn_dist = lambda x1, x2, m1, m2, T: fast_signature_dist_long(x1, x2, m1, m2, T, kernel=kernel, dyadic_order=dyadic_order, sigma=sigma, lead_lag=lead_lag, squared=False)
    else:
        fn_dist = lambda x1, x2, m1, m2, T: fast_pdist_long(x1, x2, squared=False)
   
    w_long, w_stat = float(weights[0]), float(weights[1])
    total_w = w_long + w_stat
    if total_w == 0: raise ValueError("Weights sum to zero")
    w_long, w_stat = w_long / total_w, w_stat / total_w
    
    res = {}

    D_L_tr_s, D_L_te_s = None, None
    D_S_tr_s, D_S_te_s = None, None

    # --- Longitudinal ---
    if w_long > 0:
        # Compute distances (Real Train -> Syn) and (Real Test -> Syn)
        if cached_synthetic_data:
            D_L_tr_s = cached_synthetic_data["L_rt_s"].clone()
            D_L_te_s = cached_synthetic_data["L_rtr_s"].clone()
        else:
            D_L_tr_s = fn_dist(x_train, x_syn, m_train, m_syn, T)
            D_L_te_s = fn_dist(x_test, x_syn, m_test, m_syn, T)
        
        # Normalize
        min_val = min(D_L_tr_s.min(), D_L_te_s.min())
        max_val = max(D_L_tr_s.max(), D_L_te_s.max())
        denom = max_val - min_val + 1e-8
        D_L_tr_s = (D_L_tr_s - min_val) / denom
        D_L_te_s = (D_L_te_s - min_val) / denom
        
        # Scores (Negative Distance)
        sL_tr = -_agg_k_optimized(D_L_tr_s, k, agg)
        sL_te = -_agg_k_optimized(D_L_te_s, k, agg)
        
        # Metrics
        auc, bal_acc, _ = _auc_balanced_acc_metrics(sL_tr, sL_te)
        res["AUROC_longi"] = auc
        res["MIR_risk_longi"] = 2.0 * abs(auc - 0.5)
        res["balanced_acc_longi"] = bal_acc

    # --- Static ---
    if w_stat > 0:
        if cached_synthetic_data:
            D_S_tr_s = cached_synthetic_data["S_rt_s"].clone()
            D_S_te_s = cached_synthetic_data["S_rtr_s"].clone()
        else:
            D_S_tr_s = fast_pdist_static(W_train, W_syn)
            D_S_te_s = fast_pdist_static(W_test, W_syn)
        
        min_val = min(D_S_tr_s.min(), D_S_te_s.min())
        max_val = max(D_S_tr_s.max(), D_S_te_s.max())
        denom = max_val - min_val + 1e-8
        D_S_tr_s = (D_S_tr_s - min_val) / denom
        D_S_te_s = (D_S_te_s - min_val) / denom
        
        sS_tr = -_agg_k_optimized(D_S_tr_s, k, agg)
        sS_te = -_agg_k_optimized(D_S_te_s, k, agg)
        
        auc, bal_acc, _ = _auc_balanced_acc_metrics(sS_tr, sS_te)
        res["AUROC_stat"] = auc
        res["MIR_risk_stat"] = 2.0 * abs(auc - 0.5)
        res["balanced_acc_stat"] = bal_acc

    # --- Fused ---
    if w_long > 0 and w_stat > 0:
        D_tr_s = w_long * D_L_tr_s + w_stat * D_S_tr_s
        D_te_s = w_long * D_L_te_s + w_stat * D_S_te_s

        s_tr = _agg_k_optimized(D_tr_s, k, agg)
        s_te = _agg_k_optimized(D_te_s, k, agg)

        auc, bal_acc, _ = _auc_balanced_acc_metrics(s_tr, s_te)
        res["AUROC"] = auc
        res["MIR_risk"] = 2.0 * abs(auc - 0.5)

    return res    


# -----------------------------------------------------------------------------
# Dependence score via autocorrelation matrices (mask-aware, irregular TS)
# -----------------------------------------------------------------------------

def autocorr_matrix_dataset(x: torch.Tensor, 
                            mask: torch.Tensor, 
                            max_lag: Optional[int] = None):
    """
    Vectorized Autocorrelation for all features and all lags.
    Replaces double Python loops with tensor operations.
    """
    """Compute dataset-level autocorrelation matrices R of shape (V, K+1).

    For each variable v and lag k ∈ {0..K}, compute per-patient Pearson autocorrelation
    using only jointly observed pairs (t, t-k), then average across patients that
    have at least one valid pair. Robust to irregular/missing observations.

    Args:
        x: (N,T,V) values
        mask: (N,T,V) binary mask (1 observed, 0 missing)
        max_lag: if None, use T//4
    Returns:
        R: (V, K+1) where R[v,k] is the average autocorrelation at lag k for variable v.
    """
    N, T, V = x.shape
    K = T // 4 if max_lag is None else max_lag
    
    # Center the data
    mu = (x * mask).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-8)
    xc = (x - mu) * mask
    
    R = torch.zeros((V, K + 1), device=x.device)
    
    # Loop only over lags (K is small, e.g., 10-20), Vectorize over Features (V)
    for k in range(K + 1):
        if k == 0:
            x0, x1 = xc, xc
            m0, m1 = mask, mask
        else:
            x0, x1 = xc[:, :-k, :], xc[:, k:, :]
            m0, m1 = mask[:, :-k, :], mask[:, k:, :]
        
        # 'both' mask: (N, T-k, V)
        both = m0 * m1
        
        num = (x0 * x1 * both).sum(dim=1)  # (N, V)
        den0 = (x0 * x0 * both).sum(dim=1) # (N, V)
        den1 = (x1 * x1 * both).sum(dim=1) # (N, V)
        
        den = torch.sqrt(den0 * den1).clamp(min=1e-12)
        corr_per_sample = num / den # (N, V)
        valid_mask = both.sum(dim=1) > 0 # (N, V)
        R[:, k] = (corr_per_sample * valid_mask).sum(0) / (valid_mask.sum(0) + 1e-8)
        
    return R

def dependence_score(x_r: torch.Tensor, 
                     mask_r: torch.Tensor, 
                     x_s: torch.Tensor, 
                     mask_s: torch.Tensor, 
                     max_lag: Optional[int] = None, 
                     square: bool = False, 
                     cached_real_data = None) -> Dict[str, torch.Tensor]:
    """Compute the Dependence score as the squared difference between the real and synthetic autocorrelation matrices up to lag T//4 (or max_lag).

    Returns a dict with:
        R_real: (V, K), R_syn: (V, K), mse: scalar mean of squared diffs (ignoring NaNs),
        frob: Frobenius norm of difference (NaNs treated as 0), K: max lag used.
    """
    if cached_real_data is not None: 
        Rr = cached_real_data.clone()
    else:
        Rr = autocorr_matrix_dataset(x_r, mask_r, max_lag=max_lag) # (V,K)

    Rs = autocorr_matrix_dataset(x_s, mask_s, max_lag=max_lag) # (V,K)

    K = min(Rr.size(1), Rs.size(1))
    Rr = Rr[:, :K]
    Rs = Rs[:, :K]

    diff = Rr - Rs
    mask_valid = ~(torch.isnan(Rr) | torch.isnan(Rs))
    se = (diff**2)
    mse = (se[mask_valid]).mean() if mask_valid.any() else torch.tensor(float('nan'), device=diff.device)

    diff_nz = torch.nan_to_num(diff, nan=0.0)
    frob = torch.norm(diff_nz, p='fro')

    return {"mse": float(mse), "frob": float(frob)}
