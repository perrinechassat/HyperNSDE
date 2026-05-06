# from typing import Callable, Iterable, Union, List

# import sigkernel
# import torchcde
import torch
# import time
import numpy as np
# import pysiglib
import pysiglib
# from pysiglib import LinearKernel, RBFKernel

# import signatory


def masked_gaussian_log_density(x, pred_x, mask, out_logsig=None, pred_std=None):
    # out_logsig = self.L_Dec.out_logsig

    if pred_x.dim() == 3:
        pred_x = pred_x.unsqueeze(0)
    
    n_traj_samples, n_traj, n_timepoints, n_dims = pred_x.size()
    
    x_rep = x.expand(n_traj_samples, -1, -1, -1)
    mask = mask.expand(n_traj_samples, -1, -1, -1)

    if pred_std is None:
        # std = self.L_Dec.sp(self.L_Dec.out_logsig).view(1, 1, 1, -1)
        std = torch.exp(0.5*out_logsig).view(1, 1, 1, -1)
        normal = torch.distributions.Normal(pred_x, std.expand_as(pred_x))
    else:
        normal = torch.distributions.Normal(pred_x, pred_std)
    log_probs = normal.log_prob(x_rep)
    masked_log_probs = torch.where(mask.bool(), log_probs, torch.zeros_like(log_probs))
    
    mask_sum = mask.sum(dim=2, keepdim=True).clamp(min=1)
    normalized_log_probs = masked_log_probs.sum(dim=2) / mask_sum.squeeze(2)
    log_density = normalized_log_probs.mean(dim=(1, 2))

    return log_density


def log_normal_pdf(data, mean, logvar):
    #import ipdb; ipdb.set_trace()
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(data.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (data - mean) ** 2. / torch.exp(logvar))


# def mse(x, pred_x, mask, scale=True):
#     if mask is None:
#         mse = torch.mean((x - pred_x)**2)
#     else:   
#         mse = torch.sum(mask * (x - pred_x)**2)/torch.sum(mask)
#     if scale:
#         mse = mse * x.size(1) * x.size(2)
#     return mse

def mse_old(x, pred_x, mask):
    if mask is None:
        mse = torch.mean((x - pred_x)**2)
    else:
        diff = (x - pred_x)
        diff_obs = diff[mask.bool()]
        mse = torch.mean(diff_obs**2)
    return mse

def mse(x, pred_x, mask):
    if mask is None:
        mse = torch.mean((x - pred_x)**2)
    else:
        diff_sq = (x - pred_x)**2
        mse_per_patient = (diff_sq * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp_min(1)
        mse = torch.mean(mse_per_patient)
        # global_mse = (diff_sq * mask).sum() / mask.sum().clamp_min(1)
        # mse = global_mse   
    return mse


def normal_kl(mu1, lv1, mu2, lv2):
    # I am not sure, whether this implementation is correct,
    #   but it's the one from chen etal.
    # Multivariate normal_kl is sum of univariate normal KL, if Varianz is Ip
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2)**2.) / (2. * v2)) - .5
    return kl

def kl_gaussian(mean_q, logvar_q, mean_p, logvar_p):
    var_ratio = torch.exp(logvar_q - logvar_p)
    mean_diff_sq = (mean_p - mean_q).pow(2) / torch.exp(logvar_p)
    return 0.5 * torch.sum(
        var_ratio + mean_diff_sq - 1 + logvar_p - logvar_q,
        dim=1
    )


def compute_noise_sigma(x, pred_x, mask):
    residuals = mask * (x - pred_x)  # Shape: (batch_size, n_time_pts, n_features)
    var_residuals = torch.mean(residuals**2, dim=1)  # Shape: (batch_size, n_features)
    sigma_squared = torch.mean(var_residuals, dim=0)  # Shape: (n_features,)
    sigma = torch.sqrt(sigma_squared)  # Shape: (n_features,)
    return sigma


def scale_losses_MultiNODEs(long_loss, stat_loss):
    long = long_loss / (long_loss + stat_loss)
    stat = stat_loss / (long_loss + stat_loss)
    long_scaled = stat / (long + stat) * long_loss
    stat_scaled = long / (long + stat) * stat_loss
    return long_scaled, stat_scaled


def combine_losses(long_loss, stat_loss, scaling_factor=1.0):
    stat_loss = scaling_factor * stat_loss
    loss = stat_loss * long_loss / (stat_loss + long_loss)
    return loss

def harmonic_mean(loss_dict):
    """
    Computes the harmonic mean of tensor values in a dictionary.
    
    Args:
        loss_dict (dict): A dictionary where values are PyTorch tensors.
    
    Returns:
        torch.Tensor: The harmonic mean of the tensor values.
    """
    values = list(loss_dict.values())  
    values = torch.stack(values)  
    harmonic_mean_value = torch.prod(values) / torch.sum(values)
    return harmonic_mean_value


def compute_poisson_proc_likelihood(log_lambda, int_lambda, mask=None, time_grid=None, scale=False):
    """
    log_lambda : (B, T, D+1)
    int_lambda : (B, D+1)
    mask       : (B, T, D) or None

    Returns:
        total_log_l : (B,) log-likelihood per trajectory including EOS
    """
    # 1. Split Standard Features and EOS Feature
    log_lambda_eos = log_lambda[..., -1]    # (B, T)
    int_lambda_eos = int_lambda[..., -1]    # (B,)

    log_lambda_std = log_lambda[..., :-1]   # (B, T, D)
    int_lambda_std = int_lambda[..., :-1]   # (B, D)
    
    B, T, D = log_lambda_std.shape
    init_mask = mask.clone() if mask is not None else None

    # --- Standard Feature Likelihood ---
    if mask is None:
        # Unmasked version
        log_sum = torch.sum(log_lambda_std, dim=1)  # (B, D)
        log_prob_std = log_sum - int_lambda_std     # (B, D)
        
        # For EOS, if no mask is provided, assume sequence ends at the last time step
        iK = torch.full((B,), T - 1, device=log_lambda.device, dtype=torch.long)
        has_obs = torch.ones(B, dtype=torch.bool, device=log_lambda.device)
        
    else:
        # ---- Masked version ----
        if D == 1: 
            mask = mask[:,:,0] 

        log_lambda_flat = log_lambda_std.reshape(-1, T)    # (B*D, T)
        mask_flat = mask.reshape(-1, T)                    # (B*D, T)
        int_lambda_flat = int_lambda_std.reshape(-1)       # (B*D,)
        idx = mask_flat.bool()                             # (B*D, T)

        if not idx.any():
            log_prob_std = torch.zeros(B, D, device=log_lambda.device)
        else:
            row_idx = torch.nonzero(idx)[:, 0]             
            vals = log_lambda_flat[idx]                    
            log_sum = torch.zeros(B*D, device=log_lambda.device)
            log_sum.index_add_(0, row_idx, vals)
            log_prob_flat = log_sum - int_lambda_flat      # (B*D,)
            log_prob_std = log_prob_flat.view(B, D)

        # --- Find end of sequence (iK) for EOS ---
        mask_any = init_mask.any(dim=-1)                   # (B, T)
        has_obs = mask_any.any(dim=1)                      # (B,)
        
        idx_seq = torch.arange(T, device=log_lambda.device).unsqueeze(0).expand(B, T)
        iK = torch.where(mask_any, idx_seq, -1)
        iK = iK.max(dim=1).values                          # (B,)
        iK = torch.clamp(iK, 0, T - 1)                     # Force within bounds

    # --- 2. EOS Likelihood ---
    # Extract the exact log intensity at the final timestamp for each patient
    batch_indices = torch.arange(B, device=log_lambda.device)
    log_lambda_eos_at_Ti = log_lambda_eos[batch_indices, iK]  # (B,)
    
    # EOS loss: log(lambda_eos(T_i)) - integral(lambda_eos)
    log_prob_eos = log_lambda_eos_at_Ti - int_lambda_eos      # (B,)

    # --- 3. Optional Scaling (Only applied to standard features) ---
    if scale and mask is not None: 
        i0 = torch.where(mask_any, idx_seq, T)
        i0 = i0.min(dim=1).values
        i0 = torch.clamp(i0, 0, T - 1)
        
        t0 = time_grid[i0]
        tK = time_grid[iK]
        duration = (tK - t0).clamp_min(1e-8)
        
        log_prob_std = torch.where(has_obs.unsqueeze(1), log_prob_std / duration.unsqueeze(1), log_prob_std)

    # --- 4. Final Combination ---
    mean_log_prob_std = torch.mean(log_prob_std, dim=-1)  # (B,)
    total_log_l = mean_log_prob_std + torch.where(has_obs, log_prob_eos, torch.zeros_like(log_prob_eos))

    return total_log_l  # (B,)



# ==================================================================#
# ======================= SDE losses ===============================#
# ==================================================================#

def create_discriminator(discriminator_type, data_size, discr_config):
    discriminator_list = ["SigKerMMDDiscriminator", "FDDiscriminator", "PySigMMDDiscriminator"]
    discr_config["adversarial"] = False
    discr_config["path_dim"] = data_size
    if discriminator_type in discriminator_list:
        discriminator = eval(discriminator_type)(**discr_config)
    else:
        discriminator = None
        print(f"Discriminator {discriminator_type} does not exist.")
        print(f"Choose from {', '.join(discriminator_list)}")
    return discriminator



'''
----------------------------------

    PySigLib MMD Discriminator

----------------------------------
'''

class PySigMMDDiscriminator(torch.nn.Module):
    def __init__(self, kernel_type: str, dyadic_order: int, lead_lag: bool = False, max_batch: int = -1, 
                 sigma: float = 1., **kwargs):
        """
        Discriminator using pysiglib.sig_mmd to compute signature MMD.

        :param kernel_type: type of static kernel to compose with signature kernel, currently only 'rbf' and 'linear'
        :param dyadic_order: dyadic partition refinement
        :param lead_lag: whether to apply lead-lag transform
        :param max_batch: max batch size for kernel computation
        :param sigma: bandwidth for rbf kernel
        :param degree_sig: trucation level of the signature if using rbf kernel
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.dyadic_order = dyadic_order
        self.lead_lag = lead_lag
        self.max_batch = max_batch
        self.sigma = sigma # float

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor (batch, length, dim)
        :param y: tensor (batch, length, dim)
        :return: scalar tensor, MMD loss
        """
        if self.kernel_type == "linear":
            kernel = pysiglib.LinearKernel()
        else:
            kernel = pysiglib.RBFKernel(self.sigma)
        
        sig_score = pysiglib.torch_api.expected_sig_score(sample1=x,
                                                        sample2=y,
                                                        dyadic_order=self.dyadic_order, 
                                                        lam=1.0,
                                                        static_kernel=kernel,
                                                        time_aug=False, # included separately because can be irregular
                                                        lead_lag=self.lead_lag,
                                                        n_jobs=1, 
                                                        max_batch=self.max_batch).to(x.device).squeeze()
        return sig_score

