import torch
import math
import pysiglib.torch_api as pysiglib 
import torch.nn as nn
import torchcde

def time_aug(x, T, time_norm=True):
    if time_norm:
        time_grid_norm = (T - T.min()) / (T.max() - T.min())
    else:
        time_grid_norm = T
    T_expand = time_grid_norm.unsqueeze(0).unsqueeze(-1).expand(x.size(0), time_grid_norm.size(0), 1) 
    x_time_aug = torch.cat([T_expand, x], dim=2)
    return x_time_aug


def compress_batch(x, mask, fill='last', align='left'):
    """
    Robust vectorization for compressing time series:
    1. Collapses feature mask (any(dim=-1)) to avoid duplicates.
    2. Uses nonzero() + cumsum() to strictly preserve time order.
    
    x:    (B, T, F)
    mask: (B, T, F) with 1=observed, 0=padded
    fill:  'zero', 'nan', 'last' (if align='left'), 'first' (if align='right')
    align: 'left' (pad at the end) or 'right' (pad at the beginning)
    
    Return: (B, T_max, F) where T_max is max observed points in batch
    """
    B, T, F = x.shape
    
    mask_time = mask.any(dim=-1) 
    lengths = mask_time.sum(dim=1)
    T_max = lengths.max().item()
    
    if T_max == 0:
        return torch.zeros((B, 0, F), device=x.device, dtype=x.dtype)

    valid_b, valid_t = mask_time.nonzero(as_tuple=True)
    ranks = mask_time.cumsum(dim=1) - 1 
    valid_ranks = ranks[valid_b, valid_t] 
    
    if align == 'left':
        valid_dest_t = valid_ranks
    elif align == 'right':
        pad_lengths = T_max - lengths
        # Now both are 1D tensors of size 10890, so they add perfectly!
        valid_dest_t = valid_ranks + pad_lengths[valid_b]
    else:
        raise ValueError("align must be 'left' or 'right'")
        
    # Initialize output tensor
    xs = torch.zeros((B, T_max, F), device=x.device, dtype=x.dtype)
    if fill == 'nan':
        xs.fill_(float('nan'))
        
    # Place valid data into compressed positions
    xs[valid_b, valid_dest_t, :] = x[valid_b, valid_t, :]
    
    # Handle Dynamic Filling
    t_grid = torch.arange(T_max, device=x.device).unsqueeze(0)
    valid_seqs = lengths > 0
    
    if fill == 'last' and align == 'left':
        last_idx = (lengths - 1).clamp(min=0) 
        last_vals = xs[torch.arange(B, device=x.device), last_idx, :].unsqueeze(1)
        pad_mask = (t_grid >= lengths.unsqueeze(1)) & valid_seqs.unsqueeze(1)
        xs = torch.where(pad_mask.unsqueeze(-1), last_vals, xs)
        
    elif fill == 'first' and align == 'right':
        pad_lengths = T_max - lengths
        first_idx = pad_lengths.clamp(max=T_max-1) 
        first_vals = xs[torch.arange(B, device=x.device), first_idx, :].unsqueeze(1)
        pad_mask = (t_grid < pad_lengths.unsqueeze(1)) & valid_seqs.unsqueeze(1)
        xs = torch.where(pad_mask.unsqueeze(-1), first_vals, xs)
        
    return xs


def get_deltas_and_indices(mask, T_tensor, N_samples):
    if T_tensor.dim() == 1:
        T_tensor = T_tensor.unsqueeze(0).expand(N_samples, -1)
        
    all_deltas = []
    all_indices = []
    
    for i in range(N_samples):
        valid_times = T_tensor[i][mask[i]]
        if len(valid_times) > 1:
            deltas = valid_times[1:] - valid_times[:-1]
            all_deltas.append(deltas)
            indices = torch.full_like(deltas, i, dtype=torch.long)
            all_indices.append(indices)
    
    if len(all_deltas) == 0:
        return None, None
    return torch.cat(all_deltas), torch.cat(all_indices)


def compute_masked_correlation_matrix(x_flat: torch.Tensor, m_flat: torch.Tensor) -> torch.Tensor:
    """
    Computes the Pearson correlation matrix for flattened data (N, D),
    ignoring missing values (where mask == 0).
    
    Args:
        x_flat: Data tensor (N, D). Missing values should be 0.
        m_flat: Mask tensor (N, D). 1 for valid, 0 for missing.
    
    Returns:
        corr: Correlation matrix (D, D). Entries with insufficient overlap are NaN.
    """
    N, D = x_flat.shape
    x = x_flat.float()
    m = m_flat.float()
    n_overlap = torch.mm(m.t(), m)
    sum_x = torch.mm(x.t(), m) 
    sum_y = sum_x.t()  # Symmetric but transposed context
    x2 = x ** 2
    sum_xx = torch.mm(x2.t(), m)
    sum_yy = sum_xx.t()
    sum_xy = torch.mm(x.t(), x)
    
    # N * Sum(XY) - Sum(X)*Sum(Y)
    numerator = n_overlap * sum_xy - sum_x * sum_y
    # Denominator: sqrt( [N*Sum(X^2) - Sum(X)^2] * [N*Sum(Y^2) - Sum(Y)^2] )
    denom_x = n_overlap * sum_xx - sum_x ** 2
    denom_y = n_overlap * sum_yy - sum_y ** 2
    
    denom_x = torch.clamp(denom_x, min=0)
    denom_y = torch.clamp(denom_y, min=0)
    denominator = torch.sqrt(denom_x * denom_y)
    
    corr = numerator / (denominator + 1e-8)
    corr = torch.clamp(corr, -1.0, 1.0)
    min_samples = 10
    corr[n_overlap < min_samples] = float('nan')
    
    # Only set diagonal if we had valid data
    valid_diag = n_overlap.diag() >= min_samples
    corr.diagonal().copy_(torch.where(valid_diag, torch.tensor(1.0, device=x.device), torch.tensor(float('nan'), device=x.device)))
    
    return corr

def get_median_sigma(X):
    if X.shape[0] > 1000:
        idx = torch.randperm(X.shape[0])[:1000]
        X = X[idx]
    
    dists = torch.pdist(X).pow(2) # Returns condensed distance vector
    median_dist = dists.median()
    
    return torch.sqrt(median_dist / 2)

def get_z_score(X):
    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True) + 1e-8
    return (X - mu) / sigma



def fast_signature_dist_long(x1, x2, m1, m2, T, kernel='rbf', dyadic_order=3, sigma=0.5, lead_lag=False, chunk_size=256, squared = True):
    """
    Uses pysiglib to compute a distance matrix based on the Signature Kernel.
    x1: (N1, T, V) Longitudinal data
    """

    x1[~m1.bool()] = torch.nan
    x2[~m2.bool()] = torch.nan
    T = (T - T.min()) / (T.max() - T.min())
    x1_time_aug = time_aug(x1, T)
    x2_time_aug = time_aug(x2, T)

    m_fill1 = expand_mask_to_last_obs(m1)
    m_fill2 = expand_mask_to_last_obs(m2)

    x1_time_aug_interp = torchcde.linear_interpolation_coeffs(x1_time_aug, t=T).to(x1.device)
    x2_time_aug_interp = torchcde.linear_interpolation_coeffs(x2_time_aug, t=T).to(x1.device)

    # x1_compress = compress_batch(x1_time_aug, m1)
    # x2_compress = compress_batch(x2_time_aug, m2)
    x1_compress = compress_batch(x1_time_aug_interp, m_fill1)
    x2_compress = compress_batch(x2_time_aug_interp, m_fill2)


    # Explicitly passed linear kernel - same as default behaviour
    if kernel == 'rbf':
        static_kernel = pysiglib.RBFKernel(sigma)
    else:
        static_kernel = pysiglib.LinearKernel()

    x1_compress = x1_compress.contiguous()
    x2_compress = x2_compress.contiguous()

    # ker_11 = pysiglib.sig_kernel(x1_compress, x1_compress, dyadic_order=dyadic_order, static_kernel=static_kernel, time_aug=False, lead_lag=lead_lag)
    # ker_22 = pysiglib.sig_kernel(x2_compress, x2_compress, dyadic_order=dyadic_order, static_kernel=static_kernel, time_aug=False, lead_lag=lead_lag)

    K11 = pysiglib.sig_kernel_gram(x1_compress, x1_compress, dyadic_order=dyadic_order, static_kernel=static_kernel, time_aug=False, max_batch=4, lead_lag=lead_lag)
    K22 = pysiglib.sig_kernel_gram(x2_compress, x2_compress, dyadic_order=dyadic_order, static_kernel=static_kernel, time_aug=False, max_batch=4, lead_lag=lead_lag)
    ker_11 = torch.diagonal(K11, 0)
    ker_22 = torch.diagonal(K22, 0)
    del K11, K22
    ker_12 = pysiglib.sig_kernel_gram(x1_compress, x2_compress, dyadic_order=dyadic_order, static_kernel=static_kernel, time_aug=False, max_batch=4, lead_lag=lead_lag)

    # Convert Kernel to Distance
    # d^2(x, y) = K(x, x) + K(y, y) - 2K(x, y)
    dist_sq = ker_11.unsqueeze(1) + ker_22.unsqueeze(0) - 2.0 * ker_12
    del ker_11, ker_12, ker_22
    dist_sq = torch.clamp(dist_sq, min=0.0)

    if squared:
        return dist_sq
    else:
        return torch.sqrt(dist_sq + 1e-8)


def fast_pdist_long(
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    squared: bool = True     # True for MMD, False for RMSE/Euclidean
):
    """
    Unified function for longitudinal distance calculation.
    
    Args:
        x1, m1: First batch of data and masks (N1, T, F)
        x2, m2: Second batch of data and masks (N2, T, F)
        filling: "zero" (impute 0s) or "forward" (LOCF + Backward Fill)
        squared: If True, returns ||x-y||^2. If False, returns ||x-y||.
    """
    x1 = torch.nan_to_num(x1, 0.0)
    x2 = torch.nan_to_num(x2, 0.0)

    # Flatten (N, T*F)
    N1 = x1.size(0)
    N2 = x2.size(0)
    X1_flat = x1.reshape(N1, -1)
    X2_flat = x2.reshape(N2, -1)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    x1_norm = X1_flat.pow(2).sum(dim=1, keepdim=True)  # (N1, 1)
    x2_norm = X2_flat.pow(2).sum(dim=1, keepdim=True).t() # (1, N2)
    dist_sq = x1_norm + x2_norm - 2.0 * torch.mm(X1_flat, X2_flat.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)

    if squared:
        return dist_sq
    else:
        return torch.sqrt(dist_sq + 1e-8)

def fast_pdist_static(w1, w2):
    """
    Vectorized L2 distance for static data (N, F).
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    w1_norm = w1.pow(2).sum(dim=-1, keepdim=True)  # (N1, 1)
    w2_norm = w2.pow(2).sum(dim=-1, keepdim=True).t() # (1, N2)
    dist = w1_norm + w2_norm - 2.0 * torch.mm(w1, w2.t())
    return torch.sqrt(dist.clamp(min=1e-8))


# Helper for pairwise distance/kernels
    # XX: (Nr, Nr), YY: (Ns, Ns), XY: (Nr, Ns)
def compute_kernel_matrix(A, B, kernel, gamma, coef0, degree):
    if kernel == "rbf":
        # ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A@B.T
        # optimized for GPU memory
        A_norm = (A**2).sum(dim=1, keepdim=True)
        B_norm = (B**2).sum(dim=1, keepdim=True)
        dist_sq = A_norm + B_norm.t() - 2 * torch.mm(A, B.t())
        return torch.exp(-gamma * dist_sq)
    
    elif kernel == "polynomial":
        return (gamma * torch.mm(A, B.t()) + coef0).pow(degree)

def get_batch_correlations(x, m):
    mask = m.bool().float() # Ensure float
    counts = mask.sum(dim=1, keepdim=True) # (N, 1, F)
    masked_x = x * mask
    means = masked_x.sum(dim=1, keepdim=True) / (counts + 1e-8) # (N, 1, F)
    centered = (x - means) * mask # (N, T, F)
    numerators = torch.bmm(centered.transpose(1, 2), centered) # (N, F, F)
    sum_sq = (centered ** 2).sum(dim=1) # (N, F)
    stds = torch.sqrt(sum_sq).unsqueeze(2) # (N, F, 1)
    denominators = torch.bmm(stds, stds.transpose(1, 2))
    corrs = numerators / (denominators + 1e-8)
    corrs = torch.nan_to_num(corrs, 0.0)
    return corrs


def estimate_pdf_vectorized(events_flat, sample_indices, num_samples, bins=50, t_min=0, t_max=1, eps=1e-8):
    """
    Computes the mean of per-sample PDFs using vectorized scatter operations.
    """
    bin_indices = ((events_flat - t_min) / (t_max - t_min + 1e-7) * bins).long()
    bin_indices = torch.clamp(bin_indices, 0, bins - 1)

    max_sample_idx = sample_indices.max().item() if sample_indices.numel() > 0 else 0
    actual_num_samples = max(num_samples, max_sample_idx + 1)
    flat_indices = sample_indices * bins + bin_indices
    grid = torch.zeros(actual_num_samples * bins, device=events_flat.device)
    
    grid.scatter_add_(0, flat_indices, torch.ones_like(events_flat, dtype=torch.float))
    histograms = grid.view(actual_num_samples, bins)
    
    row_sums = histograms.sum(dim=1, keepdim=True)
    per_sample_pdfs = histograms / (row_sums + eps)
    
    return per_sample_pdfs.mean(dim=0)


def augment_with_statics(x, m, W, use_nans_for_statics=False):
    """
    Optimized augmentation using broadcasting instead of .repeat()
    """
    if W is None:
        return x, m
    N, T, F_long = x.shape
    
    # Use broadcasting: (N, 1, F_stat) expands to (N, T, F_stat) without memory copy
    W_rep = W.unsqueeze(1).expand(-1, T, -1)
    
    if use_nans_for_statics:
        mask_time = (m.sum(dim=2) > 0).unsqueeze(-1) # (N, T, 1)
        W_rep = torch.where(mask_time.bool(), W_rep, torch.tensor(float('nan'), device=x.device))
        m_stat = mask_time.expand_as(W_rep).float()
    else:
        m_stat = torch.ones_like(W_rep)
    
    x_aug = torch.cat([x, W_rep], dim=-1)
    m_aug = torch.cat([m, m_stat], dim=-1)
    return x_aug, m_aug


def fill_missing_values_long(x, mask, times, filling_type='last', tol=1e-7):
    """
    Pure Torch optimized version of missing value filling.
    Replaces Python loops with vectorized GPU kernels.
    """
    N, T, F = x.shape
    device = x.device
    
    # Check grid regularity
    diff = torch.diff(times)
    is_regular = torch.allclose(diff, diff[0:1].expand_as(diff), atol=tol)
    
    filled_data = x.clone()

    if filling_type == 'last' or filling_type == 'prec_val':
        # We assume if one feature is masked, all are (matching your logic)
        obs_mask = mask[:, :, 0].bool() 
        idx = torch.arange(T, device=device).unsqueeze(0).expand(N, T)
        
        masked_idx = torch.where(obs_mask, idx, torch.tensor(-1, device=device))
        last_valid_idx, _ = torch.cummax(masked_idx, dim=1)
        
        first_valid_val_idx = obs_mask.int().argmax(dim=1) # (N,)
        initial_values = x[torch.arange(N), first_valid_val_idx] # (N, F)
        
        gather_idx = torch.clamp(last_valid_idx, min=0).unsqueeze(-1).expand(-1, -1, F)
        filled_data = torch.gather(x, 1, gather_idx)
        
        starts_mask = (last_valid_idx == -1).unsqueeze(-1)
        filled_data = torch.where(starts_mask, initial_values.unsqueeze(1), filled_data)

    elif filling_type == 'zero':
        filled_data = torch.where(mask.bool(), x, torch.zeros_like(x))

    return filled_data, mask, times

    # --- Grid Regularization ---
    # if is_regular:
        # return filled_data, mask, times
    # else:
    #     dt_min = diff.min()
    #     time_grid_reg = torch.arange(times.min().item(), times.max().item() + 1e-8, step=dt_min.item(), device=device)
        
    #     indices = torch.searchsorted(times, time_grid_reg, right=True) - 1
    #     indices = torch.clamp(indices, 0, T - 1)
        
    #     filled_data_reg = filled_data[:, indices, :]
    #     mask_reg = mask[:, indices, :]
        
    #     return filled_data_reg, mask_reg, time_grid_reg


# ------------------------------------------------------------------ #
# 2. Helper: build input tensor (filled past + time + static)          #
# ------------------------------------------------------------------ #
def build_input(x, m, T, W, N, T_split_idx):
    """Returns (N, T_split_idx, input_dim) and m_past (N, T_split_idx, V)"""
    x_fill, m_fill, _ = fill_missing_values_long(x, m, T, filling_type='last')
    # Time channel: (N, T, 1)
    T_ch = T.unsqueeze(0).unsqueeze(-1).expand(N, -1, -1).float()
    inp = torch.cat([x_fill, T_ch], dim=-1)          # (N, T, V+1)
    if W is not None:
        W_rep = W.unsqueeze(1).expand(-1, T.shape[0], -1).float()
        inp = torch.cat([inp, W_rep], dim=-1)         # (N, T, V+1+d_W)
    # Keep only past
    x_past = inp[:, :T_split_idx, :]                 # (N, T_split_idx, D)
    m_past = m[:, :T_split_idx, :]                   # (N, T_split_idx, V)
    return x_past, m_past, x_fill

# ------------------------------------------------------------------ #
# 3. Build future observation targets (variable-length per patient)    #
# ------------------------------------------------------------------ #
def build_future_targets(x, m, T, N, min_obs_future, max_obs_future, future_mask_t):
    """
    Returns:
        valid_mask  (N,)  bool — patient has >= min_obs_future future obs
        fut_vals    (N, max_obs_future, V)  padded with 0
        fut_times   (N, max_obs_future)     padded with -1
        fut_counts  (N,)  actual number of future obs per patient (capped)
    """
    # Observed AND future: (N, T_len, V) → reduce over V dim
    obs_future = (m[:, :, :] > 0) & \
                    future_mask_t.unsqueeze(0).unsqueeze(-1).expand(N, -1, V)
    # Any variable observed at each time: (N, T_len)
    any_obs_future = obs_future.any(dim=-1)

    # Count future observed time points per patient
    counts = any_obs_future.sum(dim=1)               # (N,)
    valid_mask = counts >= min_obs_future             # (N,)

    # Cap at max_obs_future
    counts_capped = counts.clamp(max=max_obs_future)

    # Gather future values and times (padded)
    fut_vals  = torch.zeros(N, max_obs_future, V,
                            device=x.device, dtype=x.dtype)
    fut_times = torch.full((N, max_obs_future), -1.0,
                            device=x.device, dtype=T.dtype)

    for i in range(N):
        if not valid_mask[i]:
            continue
        idx = torch.where(any_obs_future[i])[0]      # future obs indices
        idx = idx[:max_obs_future]                    # cap
        n   = idx.shape[0]
        fut_vals[i, :n, :]  = x[i, idx, :]
        fut_times[i, :n]    = T[idx]

    return valid_mask, fut_vals, fut_times, counts_capped



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