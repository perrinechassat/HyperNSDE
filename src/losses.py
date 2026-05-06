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


# def compute_masked_likelihood(log_lambda, data, mask, likelihood_func):
#     # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
#     # n_traj, n_timepoints, n_dims = data.size()
#     n_traj = log_lambda.shape[0]
#     res = []
#     for k in range(n_traj):
#         # for j in range(n_dims):
#         # Here we suppose that all the dimensions have the same missing values
#         data_masked = torch.masked_select(data, mask[k,:,0].bool())
#         log_lambda_masked = torch.masked_select(log_lambda[k,:,0], mask[k,:,0].bool())
#         log_prob = likelihood_func(log_lambda_masked, data_masked, indices = (k,0))
#         res.append(log_prob)
#     res = torch.stack(res, 0).to(data.device)
#     return res

# def compute_masked_likelihood(log_lambda, time_grid, mask, likelihood_func):
#     n_traj, n_timepoints, n_dims = log_lambda.size()
#     res = []
#     for k in range(n_traj):
#         log_prob_k = 0
#         for j in range(n_dims):
#             grid_masked = torch.masked_select(time_grid, mask[k,:,j].bool())
#             log_lambda_masked = torch.masked_select(log_lambda[k,:,j], mask[k,:,j].bool())
#             log_prob_k += likelihood_func(log_lambda_masked, grid_masked, indices = (k,j))
#         res.append(log_prob_k)
#     return torch.stack(res, 0).to(time_grid.device)


# def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
#     # masked_log_lambdas and masked_data 
#     n_data_points = masked_data.size()[-1]
#     if n_data_points > 0:
#         log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
#         # log_prob = log_prob / n_data_points
#     else:
#         log_prob = torch.zeros([1]).to(masked_data.device).squeeze()
#     return log_prob

# def compute_poisson_proc_likelihood(truth, log_lambda, int_lambda, mask = None):
#     # Compute Poisson likelihood
#     # https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
#     # Sum log lambdas across all time points
#     if mask is None:
#         # n_data_points = truth.shape[1] 
#         poisson_log_l = torch.sum(log_lambda, 1) - int_lambda
#         # Sum over data dims
#         poisson_log_l = torch.mean(poisson_log_l, -1)  #/ n_data_points
#     else:
#         # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
#         f = lambda log_lam, data, indices: poisson_log_likelihood(log_lam, data, indices, int_lambda)
#         poisson_log_l = compute_masked_likelihood(log_lambda, truth, mask, f)
    
#     return poisson_log_l


# def compute_poisson_proc_likelihood(log_lambda, int_lambda, mask=None, time_grid=None, scale=False):
#     """
#     log_lambda : (B, T, D+1)
#     int_lambda : (B, D+1)
#     mask       : (B, T, D) or None

#     Returns:
#         poisson_log_l : (B,) log-likelihood per trajectory
#     """
#     log_lambda_EOS = log_lambda[..., -1]  # (B, T)
#     log_lambda = log_lambda[..., :-1]  # (B, T, D)
#     B, T, D = log_lambda.shape
#     init_mask = mask.clone()

#     if D == 1: 
#         mask = mask[:,:,0] if mask is not None else None

#     if mask is None:
#         # log L = sum_t log λ(t) - ∫ λ dt
#         log_sum = torch.sum(log_lambda, dim=1)  # (B,D)
#         log_prob = log_sum - int_lambda    # (B,D)
        
#     else:
#         # ---- Masked version ----

#         # Flatten batch and dim (k,j) → K*J independent processes
#         log_lambda_flat = log_lambda.reshape(-1, T)    # (B*D, T)
#         mask_flat = mask.reshape(-1, T)                # (B*D, T)
#         int_lambda_flat = int_lambda.reshape(-1)       # (B*D,)

#         # Indexes of valid (observed) entries
#         idx = mask_flat.bool()                         # (B*D, T)

#         # Sécurité : Si aucune donnée n'est observée dans tout le batch
#         if not idx.any():
#             return torch.zeros(B, device=log_lambda.device)

#         # Sum masked log intensities per (k,j)
#         # To scatter per row efficiently:
#         row_idx = torch.nonzero(idx)[:, 0]             # flattened row indices
#         vals = log_lambda_flat[idx]                    # all observed vals

#         log_sum = torch.zeros(B*D, device=log_lambda.device)
#         log_sum.index_add_(0, row_idx, vals)

#         # Poisson likelihood per process (k,j)
#         log_prob_flat = log_sum - int_lambda_flat      # (B*D,)

#         # Reshape back to (B,D)
#         log_prob = log_prob_flat.view(B, D)

#     if scale: 
#         # Collapse along variable dimension to detect any observation at each time
#         mask_any = init_mask.any(dim=-1)        # (B, T)

#         # Sécurité : Identifier les trajectoires vides
#         has_obs = mask_any.any(dim=1) # (B,)

#         # Get indices of first and last observed time for each sample
#         idx = torch.arange(T, device=time_grid.device).unsqueeze(0).expand(B, T)  # (B, T)
        
#         i0 = torch.where(mask_any, idx, time_grid.shape[0])  # replace False with big number
#         i0 = i0.min(dim=1).values                    # (B,)
#         i0 = torch.clamp(i0, 0, T - 1) # Force l'index dans les bornes [0, T-1] pour eviter les erreurs d'indices
#         iK = torch.where(mask_any, idx, -1)
#         iK = iK.max(dim=1).values                    # (B,)
#         iK = torch.clamp(iK, 0, T - 1) # Force l'index dans les bornes [0, T-1] pour eviter les erreurs d'indices

#         # Gather start and end times
#         t0 = time_grid[i0]                                   # (B,)
#         tK = time_grid[iK]                                   # (B,)

#         # Duration for each sample
#         duration = (tK - t0).clamp_min(1e-8)         # (B,)

#         # log_prob = log_prob / duration
#         # # On ne scale que là où on a des observations, sinon on laisse tel quel (ou 0)
#         log_prob = torch.where(has_obs.unsqueeze(1), log_prob / duration.unsqueeze(1), log_prob)

#     # Average/sum over D dims
#     return torch.mean(log_prob, dim=-1)            # (B,)




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


# '''
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SigKer MMD Discriminator: Code from https://github.com/issaz/sigker-nsdes and https://anonymous.4open.science/r/Efficient-Training-of-Neural-SDEs-by-Matching-Finite-Dimensional-Distributions-E12B

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# '''


# class SummedKernel(object):
#     def __init__(self, kernels: List[sigkernel.SigKernel]):
#         self._kernels = kernels
#         self.n_kernels = len(kernels)

#     def compute_scoring_rule(self, X, y, max_batch=128):
#         res = 0
#         for k in self._kernels:
#             res += k.compute_scoring_rule(X, y, max_batch=max_batch)
#         return res/self.n_kernels

#     def compute_mmd(self, X, Y, max_batch=128):
#         res = 0
#         for k in self._kernels:
#             res += k.compute_mmd(X, Y, max_batch=max_batch)

#         return res/self.n_kernels


# def get_kernel(kernel_type, dyadic_order, sigma=1.):
#     if kernel_type.lower() == "rbf":
#         static_kernel = sigkernel.RBFKernel(sigma=sigma)
#     else:
#         # elif kernel_type.lower() == "linear":
#         static_kernel = sigkernel.LinearKernel()

#     return sigkernel.SigKernel(static_kernel=static_kernel, dyadic_order=dyadic_order)


# def initialise_signature_kernel(**kwargs) -> Union[sigkernel.SigKernel, SummedKernel]:
#     """
#     Helper function for classes that use the signature kernel

#     :param kwargs:  Signature kernel kwargs, must include kernel_type and dyadic_order
#     :return:        SigKernel object
#     """
#     kernel_type  = kwargs.get("kernel_type")
#     dyadic_order = kwargs.get("dyadic_order")
#     sigma        = kwargs.get("sigma")

#     if type(sigma) == float:
#         return get_kernel(kernel_type, dyadic_order, sigma=sigma)
#     elif type(sigma) == list:
#         return SummedKernel([get_kernel(kernel_type, dyadic_order, sigma=sig) for sig in sigma])


# class KernelDiscriminator(torch.nn.Module):
#     def __init__(self, kernel_kwargs: dict):
#         super().__init__()
#         self.kernel_kwargs = kernel_kwargs
#         # self._kernel = self._init_kernel()
#         self._kernel = None
#         self._metric = None

#     def _init_kernel(self):
#         raise NotImplementedError

#     def _init_metric(self) -> Callable:
#         """
#         We could set this here in abstract but methods may be implemented already within a package eg. sigkernel.

#         :return:    MMD callable object
#         """
#         raise NotImplementedError

#     def forward(self, x, y) -> torch.tensor:
#         raise NotImplementedError

# ### This class can go
# class PathMMDDiscriminator(KernelDiscriminator):
#     """
#     Path-space discrimination
#     """
#     def __init__(self, kernel_kwargs: dict, path_dim: int, adversarial: bool = True):
#         super().__init__(kernel_kwargs)

#         if adversarial:
#             inits = torch.ones(path_dim)
#             self._sigma = torch.nn.Parameter(inits, requires_grad=True)
#         else:
#             self._sigma = None

#         # self._kernel = self._init_kernel()
#         self._kernel = None
#         self._metric = None

#     def _init_kernel(self):
#         raise NotImplementedError

#     def _init_metric(self) -> Callable:
#         raise NotImplementedError

#     def forward(self, x, y):
#         """
#         Forward method for pathwise MMD discriminators, which apply a scaling as the adversarial component. They
#         also have some initial point penalty.

#         :param x:   Path data, shape (batch, stream, channel). Must require grad for training
#         :param y:   Path data, shape (batch, stream, channel). Should not require grad
#         :return:    Mixture MMD + initial point loss
#         """
#         mu = torch.clone(x.type(torch.float64))
#         nu = torch.clone(y.type(torch.float64))

#         if self._sigma is not None:
#             mu[..., 1:] *= self._sigma

#             # with torch.no_grad():
#             #    nu[..., 1:] *= self._sigma
            
#         out = self._metric(mu, nu)
#         return out


# class SigKerMMDDiscriminator(PathMMDDiscriminator):
#     def __init__(self, kernel_type: str, dyadic_order: int, path_dim: int, sigma: float = 1., adversarial: bool = False,
#                  max_batch: int = 128, use_phi_kernel = False, n_scalings = 0):
#         """
#         Init method for MMD discriminator using the signature kernel in the MMD calculation.

#         :param kernel_type:     Type of static kernel to compose with the signature kernel. Current choices are "rbf"
#                                 and "linear".
#         :param dyadic_order:    Dyadic partitioning of PDE solver used to estimate the lifted signature kernel.
#         :param path_dim:        Dimension of path outputs, to specify number of scalings in each path dimension.
#                                 Should not include time
#         :param sigma:           Optional fixed scaling parameter for use in the RBF kernel.
#         :param adversarial:     Whether to adversarialise the discriminator or not
#         :param max_batch:       Maximum batch size used in computation
#         :param use_phi_kernel:  Optional. Whether to implement an approximation for the generalized signature kernel
#                                 <., .>_\phi where \phi(k) = (k/2)!
#         :param n_scalings:      Number of scalings to use in the phi-kernel.
#         """
#         kernel_kwargs = {
#             "kernel_type": kernel_type, "dyadic_order": dyadic_order, "sigma": sigma
#         }

#         self.max_batch = max_batch

#         super().__init__(kernel_kwargs, path_dim, adversarial)

#         if use_phi_kernel:
#             self._phi_kernel = True
#             self._scalings   = torch.zeros(n_scalings).exponential_()
#         else:
#             self._phi_kernel = False
#             self._scalings = None

#         self._kernel = self._init_kernel()
#         self._metric = self._init_metric()

#     def _init_kernel(self) -> sigkernel.SigKernel:
#         """
#         Inits kernel for SigKerMMDDiscriminator object, using the SigKer package
#         :return:
#         """

#         return initialise_signature_kernel(**self.kernel_kwargs)

#     def _init_metric(self):
#         """
#         Initialises the MMD calculation for the Signature Kernel MMD Discriminator
#         :return:
#         """

#         def metric(X, Y, pi=None):
#             if pi is None:
#                 return self._kernel.compute_mmd(X, Y, max_batch=self.max_batch)
#             else:
#                 piX = X.clone()*pi
#                 piY = X.clone()*pi

#                 K_XX = self._kernel.compute_Gram(piX, X, sym=False, max_batch=self.max_batch)
#                 # Because k_\phi(\piX, Y) = k_\phi(X, \piY), we only need to calculate wrt one scaling
#                 K_XY = self._kernel.compute_Gram(piX, Y, sym=False, max_batch=self.max_batch)
#                 K_YY = self._kernel.compute_Gram(piY, Y, sym=False, max_batch=self.max_batch)

#                 mK_XX = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
#                 mK_YY = (torch.sum(K_YY) - torch.sum(torch.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

#                 return mK_XX + mK_YY - 2.*torch.mean(K_XY)

#         if self._phi_kernel:
#             def _weighted_metric(x, y):
#                 loss = 0
#                 n_scalings = len(self._scalings)
#                 for scale in self._scalings:
#                     mu = x.clone()
#                     nu = y.clone()

#                     mu *= scale

#                     loss += metric(mu, nu)

#                 return loss/n_scalings

#             return _weighted_metric
#         else:
#             return metric

        

# '''
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FDD Discrimiator: Code from https://anonymous.4open.science/r/Efficient-Training-of-Neural-SDEs-by-Matching-Finite-Dimensional-Distributions-E12B/src/gan/discriminators.py

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# '''

# class FDDiscriminator(torch.nn.Module):
#     def __init__(self, sigma=1.0, **kwargs):
#         super().__init__()
#         self.sigma = sigma

#     def forward(self, x, y):
#         """
#         Forward method for pathwise MMD discriminators, which apply a scaling as the adversarial component. They
#         also have some initial point penalty.

#         :param x:   Path data, shape (batch, stream, channel). Must require grad for training
#         :param y:   Path data, shape (batch, stream, channel). Should not require grad
#         :return:    Mixture MMD + initial point loss
#         """
#         T = x.shape[1]
#         Nx = x.shape[0]
#         Ny = y.shape[0]

#         xclone = torch.clone(x.type(torch.float64))
#         yclone = torch.clone(y.type(torch.float64))

#         x_fds = torch.cat((xclone[:, :-1, :], xclone[:, 1:, :]), axis=1) # (batch, stream-1, 2*channel)
#         y_fds = torch.cat((yclone[:, :-1, :], yclone[:, 1:, :]), axis=1) # (batch, stream-1, 2*channel)


#         # x_fds = torch.transpose(x_fds, 0, 1) # (stream-1, batch, 2*channel)
#         # y_fds = torch.transpose(y_fds, 0, 1) # (stream-1, batch, 2*channel)
        
#         # Compute E[k(x1, x2)]
#         #E_k_x1_x2 = expected_rbf_kernel_batched(x1, x2, self.sigma).mean()
#         E_k_x1_x2 = expected_rbf_kernel_batched(x_fds, x_fds, self.sigma, True).mean()

#         # Compute E[k(x, y)]
#         E_k_x_y = expected_rbf_kernel_batched(x_fds, y_fds, self.sigma).mean()

#         # E_k_y1_y2 = expected_rbf_kernel_batched(y_fds, y_fds, self.sigma, True).mean()
#         # return torch.abs(E_k_y1_y2 + E_k_x1_x2 - 2.* E_k_x_y)
#         return E_k_x1_x2 - 2.* E_k_x_y


# def finite_dimensional_matching_loss(x, y, sigma=1.0):
#     """
#     Forward method for pathwise MMD discriminators, which apply a scaling as the adversarial component. They
#     also have some initial point penalty.

#     :param x:   Path data, shape (batch, stream, channel). Must require grad for training
#     :param y:   Path data, shape (batch, stream, channel). Should not require grad
#     :return:    Mixture MMD + initial point loss
#     """
#     x_fds = torch.cat((x[:, :-1, :], x[:, 1:, :]), axis=1) # (batch, stream-1, 2*channel)
#     y_fds = torch.cat((y[:, :-1, :], y[:, 1:, :]), axis=1) # (batch, stream-1, 2*channel)

#     x_fds = torch.transpose(x_fds, 0, 1) # (stream-1, batch, 2*channel)
#     y_fds = torch.transpose(y_fds, 0, 1) # (stream-1, batch, 2*channel)
#     # print(x_fds.shape, y_fds.shape)

#     # Compute E[k(x1, x2)]
#     #E_k_x1_x2 = expected_rbf_kernel_batched(x1, x2, self.sigma).mean()
#     E_k_x1_x2 = expected_rbf_kernel_batched(x_fds, x_fds, sigma, True).mean()

#     # Compute E[k(x, y)]
#     E_k_x_y = expected_rbf_kernel_batched(x_fds, y_fds, sigma).mean()
#     return E_k_x1_x2 - 2* E_k_x_y


# def expected_rbf_kernel_batched(X, Y, sigma, XisY=False):
#     """
#     Compute the expected RBF kernel E[k(x, y)] where x and y are samples
#     drawn from distributions represented by X and Y respectively in batches.
    
#     Args:
#     - X (torch.Tensor): Samples from the first distribution, shape (batch_size, n_samples_X, n_features).
#     - Y (torch.Tensor): Samples from the second distribution, shape (batch_size, n_samples_Y, n_features).
#     - sigma (float): The bandwidth parameter for the RBF kernel.
    
#     Returns:
#     - torch.Tensor: The expected RBF kernel value for each batch, shape (batch_size,).
#     """
#     kernel_matrix = rbf_kernel_matrix_batched(X, Y, sigma)
#     if XisY:
#         # Create a mask to ignore the diagonal elements
#         batch_size, n_samples, _ = X.shape
#         mask = ~torch.eye(n_samples, dtype=bool).unsqueeze(0).expand(batch_size, -1, -1).to(X.device)
#         kernel_matrix = kernel_matrix.masked_select(mask).view(batch_size, -1)
#         expected_value = kernel_matrix.mean(dim=1)  # Mean over the non-diagonal elements
#     else:
#         expected_value = kernel_matrix.mean(dim=(1, 2))  # Mean over the sample dimensions
#     return expected_value
    
    
# def rbf_kernel_matrix_batched(X, Y, sigma, num_random_sigmas=10):
#     """
#     Compute the RBF kernel matrix between two sets of samples X and Y in batches.
    
#     Args:
#     - X (torch.Tensor): Samples from the first distribution, shape (batch_size, n_samples_X, n_features).
#     - Y (torch.Tensor): Samples from the second distribution, shape (batch_size, n_samples_Y, n_features).
#     - sigma (float): The bandwidth parameter for the RBF kernel.
    
#     Returns:
#     - torch.Tensor: The RBF kernel matrix, shape (batch_size, n_samples_X, n_samples_Y).
#     """
#     XX = torch.sum(X ** 2, dim=2, keepdim=True)  # Shape: (batch_size, n_samples_X, 1)
#     YY = torch.sum(Y ** 2, dim=2, keepdim=True)  # Shape: (batch_size, n_samples_Y, 1)
#     distances = XX + YY.transpose(1, 2) - 2 * torch.bmm(X, Y.transpose(1, 2))  # Shape: (batch_size, n_samples_X, n_samples_Y)
#     if (sigma == 'random') or isinstance(sigma, list):
#         # Generate random positive real numbers for sigma
#         if sigma == 'random':
#             sigmas = torch.exp(torch.abs(torch.randn(num_random_sigmas)))  # Shape: (num_random_sigmas,)
#         else:
#             sigmas = torch.tensor(sigma)
#         sigmas = sigmas.to(X.device).view(-1, 1, 1, 1)  # Shape: (num_random_sigmas, 1, 1, 1)
        
#         # Compute the kernel matrices for all random sigmas
#         kernel_matrices = torch.exp(-distances.unsqueeze(0) / (2 * sigmas ** 2))  # Shape: (num_random_sigmas, batch_size, n_samples_X, n_samples_Y)
        
#         # Compute the weighted average of the kernel matrices
#         kernel_matrix = kernel_matrices.mean(dim=0)  # Shape: (batch_size, n_samples_X, n_samples_Y)
#     else:
#         kernel_matrix = torch.exp(-distances / (2 * sigma ** 2))
#     return kernel_matrix