import torch
import numpy as np
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import interpolate

from sklearn.mixture import GaussianMixture


def time_normalisation_transform(bank):
    batch, stream, channel = bank.size()

    res = torch.zeros((batch, stream, channel), device=bank.device)
    res[:, :, 0] = torch.linspace(0, 1, stream, device=bank.device)
    res[:, :, 1:] = bank[:, :, 1:]

    return res

# ==================================================================#
# ======== Function for ODE-RNN Encoder of Latent ODE Paper ========#
# ==================================================================#
def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim//2

	if len(data.size()) == 3:
		res = data[:,:,:last_dim], data[:,:,last_dim:]

	if len(data.size()) == 2:
		res = data[:,:last_dim], data[:,last_dim:]
	return res
               
def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, val=0)
        
def check_mask(data, mask):
    #check that "mask" argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()

    # mask should contain only zeros and ones
    assert((n_zeros + n_ones) == np.prod(list(mask.size())))

    # all masked out elements should be zeros
    assert(torch.sum(data[mask == 0.] != 0.) == 0)

def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

# ==================================================================#
# ==================================================================#
def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


# ==================================================================#
# ==================================================================#
class RunningAverageMeter(object):
    # Computes and stores the average and current values
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def set_from_values(self, avg, val=None):
        if val is None: 
            self.val = avg
        else:
            self.val = val
        self.avg = avg

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

# ==================================================================#
# ==================================================================#
def define_logs(config):
    import os
    import time
    typ = 'a' if os.path.exists(config.save_path_losses) else 'wt'
    with open(config.save_path_losses, typ) as opt_file:
        now = time.strftime("%c")
        opt_file.write('================ Training Loss (%s) ================\n' % now)
    
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    opt_name = os.path.join(config.save_path, 'train_opt.txt')
    with open(opt_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


# ==================================================================#
# ==================================================================#
def print_current_losses(epoch, iters, train_losses, val_losses, t_epoch, t_comp, log_name, s_excel=True, save_losses=True):
    """Print current losses on console; also save the losses to the disk.
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        train_losses (OrderedDict) -- training losses stored as (name, float) pairs
        val_losses (OrderedDict) -- validation losses stored as (name, float) pairs
        t_epoch (float or str) -- total time spent per epoch (formatted if needed)
        t_comp (float or str) -- computational time per data point (formatted if needed)
        log_name (str) -- path to the log file
        s_excel (bool) -- whether to save to an Excel sheet and plot losses (default True)
        print_losses (bool) -- 
    """
    t_epoch = human_format(t_epoch)
    t_comp = human_format(t_comp)

    # Create the initial message with epoch and timing information
    message = f'[epoch: {epoch}], [iters: {iters}], [epoch time: {t_epoch}], [total time: {t_comp}] '

    # Append train losses to the message
    message += ' | '.join([f'{k}={v:.6f}' for k, v in train_losses.items()])
    
    # Append val losses to the message
    message += ' | ' + ' | '.join([f'{k}={v:.6f}' for k, v in val_losses.items()])

    # Print and log the message
    print(message)

    if save_losses:
        with open(log_name, 'a') as log_file:
            log_file.write(f'{message}\n')

        # If Excel saving is enabled, update Excel and generate plot
        if s_excel:
            excel_name = log_name[:-3] + 'xlsx'
            data = {}

            # Load existing data if the Excel file exists
            if os.path.exists(excel_name):
                loss_df = pd.read_excel(excel_name, index_col=0)
                for k in loss_df.keys():
                    vect = loss_df[k].values.tolist()
                    if k == 'Epoch':
                        vect.append(epoch)
                    elif k in train_losses:
                        vect.append(train_losses[k])
                    elif k in val_losses:
                        vect.append(val_losses[k])
                    data[k] = vect
            else:
                # Initialize the Excel columns
                data['Epoch'] = [epoch]
                for k, v in train_losses.items():
                    data[k] = [v]
                for k, v in val_losses.items():
                    data[k] = [v]

            # Save data to Excel
            df = pd.DataFrame(data)
            df.to_excel(excel_name)

            # Plot the losses if there’s more than one epoch
            if epoch > 1:
                time_vect = loss_df['Epoch'].values.tolist()
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                # Plot Loss (Primary Y-axis)
                for k in loss_df.keys():
                    if k != 'Epoch':
                        if 'Loss' in k:
                            ax1.plot(time_vect, loss_df[k].values.tolist(), label=k, linewidth=2, linestyle='-')
                        if 'MSE' in k:
                            ax2.plot(time_vect, loss_df[k].values.tolist(), label=k, linewidth=2, linestyle='--')
                ax1.set_xlabel("Epoch")
                # ax1.set_ylabel("Loss (log scale)")
                ax1.set_ylabel("Loss")
                # ax1.set_yscale("log")
                ax1.tick_params(axis='y')
                # ax2.set_ylabel("MSE (log scale)")
                ax2.set_ylabel("MSE")
                # ax2.set_yscale("log")
                ax2.tick_params(axis='y')
                ax1.legend(loc='upper left', bbox_to_anchor=(0.05, 1))
                ax2.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
                plt.title('Training and Validation Smooth Total Losses and MSEs')
                img_name = log_name[:-3] + 'png'
                plt.savefig(img_name)
                plt.close()

                for k in loss_df.keys():
                    if k != 'Epoch' and ('val' not in k and 'Loss' not in k and 'MSE' not in k):
                        plt.plot(time_vect, loss_df[k].values.tolist(), label=k, linewidth=2)
                img_name = log_name[:-4] + '_train.png'
                plt.legend()
                plt.title('All Raw Losses Train Set')
                # plt.yscale('log')
                plt.xlabel('Epoch')
                # plt.ylabel('Loss (log scale)')
                plt.ylabel('Loss')
                plt.savefig(img_name)
                plt.close()

                for k in loss_df.keys():
                    if k != 'Epoch' and ('train' not in k and 'Loss' not in k and 'MSE' not in k and 'smooth' not in k):
                        plt.plot(time_vect, loss_df[k].values.tolist(), label=k, linewidth=2)
                img_name = log_name[:-4] + '_validation.png'
                plt.legend()
                plt.title('All Raw Losses Validation Set')
                # plt.yscale('log')
                plt.xlabel('Epoch')
                # plt.ylabel('Loss (log scale)')
                plt.ylabel('Loss')
                plt.savefig(img_name)
                plt.close()

                for k in loss_df.keys():
                    if k != 'Epoch' and ('train' not in k and 'Loss' not in k and 'MSE' not in k and 'smooth' in k):
                        plt.plot(time_vect, loss_df[k].values.tolist(), label=k, linewidth=2)
                img_name = log_name[:-4] + '_smooth_validation.png'
                plt.legend()
                plt.title('All Smooth Losses Validation Set')
                # plt.yscale('log')
                plt.xlabel('Epoch')
                # plt.ylabel('Loss (log scale)')
                plt.ylabel('Loss')
                plt.savefig(img_name)
                plt.close()


def plot_final_losses(train_losses_list, val_losses_list, val_mses_list, log_name):
    plt.plot(range(len(train_losses_list)), train_losses_list, label='Training Loss')
    plt.plot(range(len(val_losses_list)), val_losses_list, label='Validation Loss')
    plt.plot(range(len(val_mses_list)), val_mses_list, label='Validation MSE')
    img_name = log_name[:-4] + '_avg.png'
    plt.legend()
    plt.title('Training and Validation Losses/MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(img_name)
    plt.close()


def generate_poisson_process_thinning(lambdas, grid, T):
    """
    Génère un processus de Poisson non homogènes sur [0, T] à partir des intensités en utilisant la méthode de thinning.

    Args:
        lambdas (torch.Tensor): Tenseur de forme (M,) représentant les intensités sur la grille fine.
        grid (torch.Tensor): Tenseur de forme (M,) représentant la grille fine.
        T (float): La durée totale de l'intervalle [0, T].

    Returns:
        torch.Tensor: Tenseur représentant les temps d'événements générés.
    """
    M = len(lambdas)

    # Trouver l'intensité maximale sur toute la grille
    lambda_max = lambdas.max().item()

    # Générer des temps d'événements pour le i-ème processus de Poisson
    events = []
    t = 0
    while t < T:
        # Générer un temps d'événement candidat selon un processus de Poisson homogène avec intensité lambda_max
        u = torch.rand(1)
        t_candidate = t - (1 / lambda_max) * torch.log(u)

        if t_candidate >= T:
            break

        # Trouver l'indice correspondant sur la grille
        idx = torch.searchsorted(grid, t_candidate, right=True).item() - 1
        idx = max(0, min(idx, M - 1))

        # Accepter ou rejeter l'événement candidat en fonction de l'intensité variable
        acceptance_prob = lambdas[idx].item() / lambda_max
        if torch.rand(1).item() < acceptance_prob:
            events.append(t_candidate)

        t = t_candidate

    return torch.tensor(events)



def generate_poisson_processes_thinning(lambdas, grid, T):
    """
    Génère N processus de Poisson non homogènes sur [0, T] à partir des intensités en utilisant la méthode de thinning.

    Args:
        lambdas (torch.Tensor): Tenseur de forme (N, M) représentant les intensités sur la grille fine.
        grid (torch.Tensor): Tenseur de forme (M,) représentant la grille fine.
        T (float): La durée totale de l'intervalle [0, T].

    Returns:
        List[torch.Tensor]: Liste de tenseurs représentant les temps d'événements générés pour chaque processus de Poisson.
    """
    N, M = lambdas.shape
    poisson_processes = []

    # Trouver l'intensité maximale sur toute la grille
    lambda_max = lambdas.max().item()

    for i in range(N):
        # Générer des temps d'événements pour le i-ème processus de Poisson
        events = []
        t = 0
        while t < T:
            # Générer un temps d'événement candidat selon un processus de Poisson homogène avec intensité lambda_max
            u = torch.rand(1)
            t_candidate = t - (1 / lambda_max) * torch.log(u)

            if t_candidate >= T:
                break

            # Trouver l'indice correspondant sur la grille
            idx = torch.searchsorted(grid, t_candidate, right=True).item() - 1
            idx = max(0, min(idx, M - 1))

            # Accepter ou rejeter l'événement candidat en fonction de l'intensité variable
            acceptance_prob = lambdas[i, idx].item() / lambda_max
            if torch.rand(1).item() < acceptance_prob:
                events.append(t_candidate)

            t = t_candidate

        poisson_processes.append(torch.tensor(events))

    return poisson_processes


def union_grids_with_tolerance(grids, tolerance):
    """
    Fait l'union de plusieurs grilles avec une certaine tolérance.

    Args:
        grids (List[torch.Tensor]): Liste de tenseurs représentant les grilles.
        tolerance (float): Tolérance pour fusionner les valeurs proches.

    Returns:
        torch.Tensor: Tenseur représentant l'union des grilles avec la tolérance spécifiée.
    """
    combined_grid = torch.cat(grids)
    combined_grid = torch.unique(combined_grid, sorted=True)
    
    diff =  combined_grid[1:] - combined_grid[:-1]

    mask = diff >= tolerance 
    mask = torch.cat((mask, torch.Tensor([True]))).bool()
    combined_grid = combined_grid[mask]

    return combined_grid


def generate_mask_grid_from_inhomogeneous_poisson_old(intensities, reg_grid):
    n_samples = intensities.shape[0]
    mask = torch.zeros(intensities.shape)
    T = reg_grid[-1].item() if isinstance(reg_grid[-1], torch.Tensor) else float(reg_grid[-1])
    for n in range(n_samples):
        for k in range(intensities.shape[-1]):
            lambda_nk = intensities[n, :, k]
            K = len(reg_grid)
            lambda_max = torch.max(lambda_nk)
            N = int(torch.distributions.Poisson(T * lambda_max).sample().item())
            if N == 0:
                continue
            I = torch.randint(0, K, (N,))
            M = torch.zeros(I.shape)
            for l in range(N):
                num_interval = I[l].item()
                u = torch.rand(1).item() * lambda_max
                # u = self.rng.random() * lambda_max.item()
                t_left = num_interval * T / K
                t_right = (num_interval + 1) * T / K
                # find idx of t_left and t_right in reg_grid
                idx_left = torch.searchsorted(reg_grid, t_left, right=True).item() - 1
                idx_right = torch.searchsorted(reg_grid, t_right, right=True).item() - 1
                if u <= ((lambda_nk[idx_left] + lambda_nk[idx_right]) / 2).item():
                    M[l] = 1.
            if M.sum() == 0:
                continue
            else:
                idx_new_grid_nk, _ = torch.sort(torch.unique(I[M.bool()]))
                mask[n, idx_new_grid_nk, k] = 1.
    return mask


def generate_mask_grid_from_inhomogeneous_poisson(intensities, reg_grid):
    """
    intensities: (n_samples, K, d_X + 1) - The last dimension is the EOS intensity.
    reg_grid: (K,) - The time grid.
    
    Returns:
        mask: (n_samples, K, d_X) - The binary mask for standard features.
        final_times: (n_samples,) - The generated T_i for each patient.
    """
    n_samples, K, n_features_total = intensities.shape
    D = n_features_total - 1  # Number of standard features
    mask = torch.zeros((n_samples, K, D), device=intensities.device)
    final_times = torch.zeros(n_samples, device=intensities.device)
    T = reg_grid[-1].item() if isinstance(reg_grid[-1], torch.Tensor) else float(reg_grid[-1])
    
    for n in range(n_samples):
        # ---------------------------------------------------------
        # Generate End-Of-Sequence (EOS) Event First
        # ---------------------------------------------------------
        lambda_eos = intensities[n, :, D] # The last dimension
        lambda_max_eos = torch.max(lambda_eos)
        N_eos = int(torch.distributions.Poisson(T * lambda_max_eos).sample().item())
        eos_idx = K - 1  # Default to max time if no EOS triggers
        if N_eos > 0:
            I_eos = torch.randint(0, K, (N_eos,))
            M_eos = torch.zeros(I_eos.shape, dtype=torch.bool)
            for l in range(N_eos):
                num_interval = I_eos[l].item()
                u = torch.rand(1).item() * lambda_max_eos.item()
                t_left = num_interval * T / K
                t_right = (num_interval + 1) * T / K
                idx_left = torch.searchsorted(reg_grid, t_left, right=True).item() - 1
                idx_right = torch.searchsorted(reg_grid, t_right, right=True).item() - 1
                if u <= ((lambda_eos[idx_left] + lambda_eos[idx_right]) / 2).item():
                    M_eos[l] = True
            if M_eos.sum() > 0:
                # The sequence ends at the FIRST generated EOS event
                valid_eos_indices, _ = torch.sort(torch.unique(I_eos[M_eos]))
                eos_idx = valid_eos_indices[0].item()
        # Record the final time for patient n
        final_times[n] = reg_grid[eos_idx]

        # ---------------------------------------------------------
        # Generate Standard Events and Censor them
        # ---------------------------------------------------------
        for k in range(D):
            lambda_nk = intensities[n, :, k]
            lambda_max = torch.max(lambda_nk)
            N = int(torch.distributions.Poisson(T * lambda_max).sample().item())
            if N == 0:
                continue    
            I = torch.randint(0, K, (N,))
            M = torch.zeros(I.shape, dtype=torch.bool)
            for l in range(N):
                num_interval = I[l].item()
                u = torch.rand(1).item() * lambda_max.item()
                t_left = num_interval * T / K
                t_right = (num_interval + 1) * T / K
                idx_left = torch.searchsorted(reg_grid, t_left, right=True).item() - 1
                idx_right = torch.searchsorted(reg_grid, t_right, right=True).item() - 1
                if u <= ((lambda_nk[idx_left] + lambda_nk[idx_right]) / 2).item():
                    M[l] = True
            if M.sum() > 0:
                idx_new_grid_nk, _ = torch.sort(torch.unique(I[M]))
                
                # !! CENSORING STEP !!
                # We only keep events that occur BEFORE or AT the EOS event
                valid_indices = idx_new_grid_nk[idx_new_grid_nk <= eos_idx]
                if len(valid_indices) > 0:
                    mask[n, valid_indices, k] = 1.
    
    return mask, final_times

# def generate_mask_grid_from_inhomogeneous_poisson_old(lambdas, grid):
#     """
#     Génère N processus de Poisson non homogènes sur [0, T] à partir des intensités en utilisant la méthode de thinning.
#     Compute l'union de ces N processus de Poisson et la matrice mask correspondante. 

#     Args:
#         lambdas (torch.Tensor): Tenseur de forme (N, M, d) représentant les intensités sur la grille fine.
#         grid (torch.Tensor): Tenseur de forme (M,) représentant la grille fine.

#     Returns:
#         combined_grid (torch.Tensor): Tenseur représentant l'union des grilles.
#         mask (torch.Tensor): Tenseur de forme (N, len(combined_grid), d) représentant la matrice mask.
#     """
#     T = grid[-1]
#     N, M, d = lambdas.shape
#     tolerance = 0.0001

#     all_events = []
#     combined_grids = []
#     for n in range(N):
#         events_n = []
#         for i in range(d):
#             events = generate_poisson_process_thinning(lambdas[n, :, i], grid, T)
#             events_n.append(events)
#         combined_grid_n = union_grids_with_tolerance(events_n, tolerance)
#         all_events.append(events_n)
#         combined_grids.append(combined_grid_n)

#     full_combined_grid = union_grids_with_tolerance(combined_grids, tolerance)
#     mask = torch.zeros(N, len(full_combined_grid), d)
#     for n in range(N):
#         for i in range(d):
#             idx = torch.searchsorted(full_combined_grid, all_events[n][i]) # mettre un round ou une tolérance ici peut être
#             mask[n, idx, i] = 1.0
#     return full_combined_grid, mask 

    # poisson_processes = generate_poisson_processes_thinning(lambdas, grid, T)
    # # tolerance = torch.min(grid[1:]-grid[:-1])/10
    # # print('Tolerance:', tolerance)
    # tolerance = 0.0001
    # combined_grid = union_grids_with_tolerance(poisson_processes, tolerance)

    # mask = torch.zeros(lambdas.shape[0], len(combined_grid))
    # N_poiss = []
    # for n in range(lambdas.shape[0]):
    #     N_poiss.append(len(poisson_processes[n]))
    #     idx = torch.searchsorted(combined_grid, poisson_processes[n])
    #     mask[n, idx] = 1.0
    # mask = mask.unsqueeze(2)
    # print('Nb points in generated grids: mean ', torch.mean(torch.Tensor(N_poiss)), ' std ', torch.std(torch.Tensor(N_poiss)))
    # return combined_grid, mask
    

def compute_marginal_distrib(arr_mu, arr_logvar):
    """
    Computes the marginal distribution of the latent variables.

    Parameters:
        - arr_mu (torch.Tensor): Mean of the latent variables.
        - arr_logvar (torch.Tensor): Log variance of the latent variables.

    Returns:
        - mean_mu (torch.Tensor): Mean of the marginal distribution.
        - mean_cov (torch.Tensor): Covariance of the marginal distribution.
    """
    mean_mu = torch.mean(arr_mu, axis=0)
    arr_var = torch.exp(arr_logvar)
    # mean_std = torch.sqrt(torch.mean(arr_var, axis=0))
    # return mean_mu, mean_std
    eps = 1e-8
    mu_diff = arr_mu - mean_mu
    mean_cov = torch.mean(arr_var, axis=0) @ torch.eye(arr_var.shape[1]) + mu_diff.T @ mu_diff / len(arr_mu) #+ eps * torch.eye(arr_var.shape[1])
    return mean_mu, mean_cov



def sample_from_large_gmm(weights, means, covs, num_samples):
    """
    Sample from a GMM with many components.
    
    Args:
        weights: Tensor of shape [N*K]
        means: Tensor of shape [N*K, D]
        covs: Tensor of shape [N*K, D, D] (full) or [N*K, D] (diagonal)
        num_samples: int, number of samples

    Returns:
        samples: Tensor of shape [num_samples, D]
    """
    print(weights.shape, means.shape, covs.shape)
    N_K, D = means.shape
    weights_np = weights.cpu().numpy()
    component_indices = np.random.choice(N_K, size=num_samples, p=weights_np)
    
    samples = []
    for idx in component_indices:
        mean = means[idx]
        cov = covs[idx]
        if cov.ndim == 1:
            # Diagonal covariance
            sample = torch.normal(mean, torch.sqrt(cov))
        else:
            # Full covariance
            sample = torch.distributions.MultivariateNormal(mean, cov).sample()
        samples.append(sample)

    return torch.stack(samples, dim=0)  # [num_samples, D]


def fit_gmm_with_k_components(samples, K):
    """
    Fit a K-component GMM to samples using sklearn.

    Args:
        samples: Tensor [num_samples, D]
        K: int, desired number of components

    Returns:
        pi_k: [K] torch tensor of component weights
        mu_k: [K, D] tensor of means
        sigma_k: [K, D, D] tensor of covariances
    """
    samples_np = samples.cpu().numpy()
    gmm = GaussianMixture(n_components=K, covariance_type='full')
    gmm.fit(samples_np)

    pi_k = torch.tensor(gmm.weights_, dtype=torch.float32)
    mu_k = torch.tensor(gmm.means_, dtype=torch.float32)
    sigma_k = torch.tensor(gmm.covariances_, dtype=torch.float32)

    return pi_k, mu_k, sigma_k


def subtract_initial_point(paths):
    _, length, dim = paths.size()
    res = paths.clone()
    start_points = torch.transpose(res[:, 0, 1:].unsqueeze(-1), -1, 1)
    res[..., 1:] -= torch.tile(start_points, (1, length, 1))
    return res

# def simulate_inhomogeneous_poisson(t_min, t_max, lambda_values, grid):
#     """
#     Simulate an inhomogeneous Poisson process using the thinning method.
    
#     Parameters:
#         t_min (float): Start time.
#         t_max (float): End time.
#         lambda_values (torch.Tensor): Intensity values on the grid.
#         grid (torch.Tensor): Time grid corresponding to lambda_values.

#     Returns:
#         torch.Tensor: Simulated event times.
#     """
    
#     # Step 1: Compute the maximum intensity
#     lambda_max = torch.max(lambda_values)
#     print(lambda_max, (t_max - t_min) * lambda_max)
    
#     # Step 2: Simulate a homogeneous Poisson process with rate lambda_max
#     N_max = torch.poisson((t_max - t_min) * lambda_max * 1.5).item()  # Number of candidate events
#     uniform_times = torch.rand(int(N_max)) * (t_max - t_min) + t_min  # Uniform proposal times
#     uniform_times, _ = torch.sort(uniform_times)  # Sort times
    
#     # Step 3: Compute lambda(t) at the proposed times via interpolation
#     lambda_t = torch.from_numpy(np.interp(uniform_times.numpy(), grid.numpy(), lambda_values.numpy()))
    
#     # Step 4: Accept or reject using thinning
#     acceptance_probs = lambda_t / lambda_max
#     accepted = torch.rand(int(N_max)) < acceptance_probs
#     event_times = uniform_times[accepted]
    
#     return event_times



# def simulate_inhomogeneous_poisson(t_min, t_max, lambda_values, grid):
#     """
#     Simulate an inhomogeneous Poisson process.
    
#     Args:
#     lambda_t (callable): Intensity function λ(t)
#     T (float): End time of the process
    
#     Returns:
#     torch.Tensor: Array of event times
#     """
#     # Initialize variables
#     n = m = 0
#     t0 = s0 = 0.0
#     lambda_bar = torch.max(lambda_values)
#     lambda_t = interpolate.interp1d(grid.numpy(), lambda_values.numpy())

#     s = torch.tensor([s0])
#     t = torch.tensor([])
    
#     while s[-1] < t_max:
#         # Generate uniform random number
#         u = torch.rand(1)
        
#         # Generate exponential random number
#         w = -torch.log(u) / lambda_bar
        
#         # Update s
#         s_new = s[-1] + w
#         s = torch.cat([s, s_new])
        
#         # Generate uniform random number for acceptance
#         D = torch.rand(1)
        
#         # Check acceptance condition
#         if s[-1] < t_max and D <= lambda_t(s[-1])/lambda_bar:
#             t = torch.cat([t, s[-1].unsqueeze(0)])
#             n += 1
        
#         m += 1
#     print(m, n)
    
#     # Return the appropriate subset of t
#     if t[-1] <= t_max:
#         return t
#     else:
#         return t[:-1]
    


def invert_cumulative_intensity(cum_intensity, t_grid, value):
    """
    Given an increasing cumulative intensity vector (cum_intensity) computed on t_grid,
    return an approximate time t such that cum_intensity(t) = value using linear interpolation.
    """
    # Find the index where the cumulative intensity first exceeds the value.
    idx = torch.searchsorted(cum_intensity, torch.tensor(value))
    
    # If the value is smaller than the first element, return the first time point.
    if idx == 0:
        return t_grid[0].item()
    # If the value is larger than the maximum, return the last time.
    elif idx >= len(t_grid):
        return t_grid[-1].item()
    else:
        t1, t2 = t_grid[idx - 1], t_grid[idx]
        L1, L2 = cum_intensity[idx - 1], cum_intensity[idx]
        # Linear interpolation to approximate the inversion.
        t_event = t1 + (value - L1) / (L2 - L1) * (t2 - t1)
        return t_event.item()

def simulate_inhomogeneous_poisson(cum_intensity, t_grid):
    """
    Sample event times from a non-homogeneous Poisson process on [0, T] using
    inverse transform sampling.

    Args:
        lambda_func: A function mapping a tensor of times to intensities.
                     It should be vectorized (i.e. work on a tensor of t values).
        T: The time horizon.
        dt: Time step for approximating the cumulative intensity.
        
    Returns:
        A 1D tensor containing the sampled event times.
    """
    
    events = []
    s = 0.0  # This will track the cumulative sum of exponential jumps.
    exponential = torch.distributions.Exponential(torch.tensor(1.0))
    
    while True:
        # Sample the next exponential jump.
        u = exponential.sample().item()
        s += u
        
        # If the total exceeds the cumulative intensity at T, we stop.
        if s > cum_intensity[-1]:
            break
        
        # Invert the cumulative intensity function to get the event time.
        t_event = invert_cumulative_intensity(cum_intensity, t_grid, s)
        events.append(t_event)
    
    return torch.tensor(events)


# def print_current_losses(epoch, iters, losses, t_epoch, t_comp,
#                          log_name, s_excel=True):
#     """print current losses on console; also save the losses to the disk
#     Parameters:
#         epoch (int) -- current epoch
#         iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
#         losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
#         t_comp (float) -- computational time per data point (normalized by batch_size)
#         t_data (float) -- data loading time per data point (normalized by batch_size)
#     """
#     t_epoch = human_format(t_epoch)
#     t_comp = human_format(t_comp)
#     # 
#     if isinstance(t_epoch, float)  and isinstance(t_comp, float):
#         message = '[epoch: %d], [iters: %d], [epoch time: %.3f], [total time: %.3f] ' % (epoch, iters, t_epoch, t_comp)
#     elif isinstance(t_epoch, float) and isinstance(t_comp, str):
#         message = '[epoch: %d], [iters: %d], [epoch time: %.3f], [total time: %s] ' % (epoch, iters, t_epoch, t_comp)
#     elif isinstance(t_epoch, str) and isinstance(t_comp, float):
#         message = '[epoch: %d], [iters: %d], [epoch time: %s], [total time: %.3f] ' % (epoch, iters, t_epoch, t_comp)
#     else:
#         message = '[epoch: %d], [iters: %d], [epoch time: %s], [total time: %s] ' % (epoch, iters, t_epoch, t_comp)

#     for k, v in losses.items():
#         message += '%s=%.6f, ' % (k, v)
#     message = message[:-2]
#     print(message)  # print the message
#     with open(log_name, 'a') as log_file:
#         log_file.write('%s\n' % message)  # save the message

#     if s_excel:
#         # Save the losses in an excel sheet and plot in an image
#         excel_name = log_name[:-3] + 'xlsx'
#         data = {}

#         # It is necessary to do 2 times this for
#         # to get the complete vectors for the graphs
#         if os.path.exists(excel_name):
#             loss_df = pd.read_excel(excel_name, index_col=0)
#             for k in loss_df.keys():
#                 vect = loss_df[k].values.tolist()
#                 if k == 'Epoch':
#                     vect.append(epoch)
#                 else:
#                     vect.append(losses[k])
#                 data[k] = vect
#         else:
#             data['Epoch'] = epoch
#             for k, v in losses.items():
#                 data[k] = [v]

#         df = pd.DataFrame(data)
#         df.to_excel(excel_name)

#         if epoch > 1:
#             for k in loss_df.keys():
#                 if k == 'Epoch':
#                     time_vect = loss_df[k].values.tolist()
#                 else:
#                     vect = loss_df[k].values.tolist()           
#                     plt.plot(time_vect, vect, label=k, linewidth=2)

#             img_name = log_name[:-3] + 'png'
#             plt.legend()
#             plt.title('Losses')
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss')
#             plt.savefig(img_name)
#             plt.close()

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if magnitude == 0:
        # return str(num)
        return num
    else:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])




##################################################################################

def concatenate_datasets(X_obs, X_gen, mask_obs, mask_gen, T_obs, T_gen):
    # Compute the unified time grid
    T_concat = torch.unique(torch.cat([T_obs, T_gen]))  # Sorted unique time points
    n_time_concat = T_concat.shape[0]
    
    # Initialize output tensors
    N, n_time_obs, n_features = X_obs.shape
    N_gen, n_time_gen, _ = X_gen.shape
    device = X_obs.device  # Use the same device as the input tensors
    
    X_concat = torch.full((N + N_gen, n_time_concat, n_features), float('nan'), device=device, dtype=X_obs.dtype)
    mask_concat = torch.zeros((N + N_gen, n_time_concat, n_features), dtype=torch.int, device=device)

    # Fill the new tensors based on the original time grids
    def insert_values(X_source, mask_source, T_source, X_target, mask_target, index_offset=0):
        indices = torch.searchsorted(T_concat, T_source)  # Find corresponding indices in T_concat
        X_target[int(index_offset):int(index_offset+X_source.shape[0]), indices, :] = X_source
        mask_target[int(index_offset):int(index_offset+X_source.shape[0]), indices, :] = mask_source

    insert_values(X_obs, mask_obs, T_obs, X_concat, mask_concat)
    insert_values(X_gen, mask_gen, T_gen, X_concat, mask_concat, index_offset=N)

    return X_concat, mask_concat, T_concat


def onehot_batch_norm(s_onehot, s_types, s_miss):
    # Batch Normalization for the Onehot encoded static data
    s_data_norm = s_onehot.clone()
    b_mean, b_var = [], []
    onehot_id = 0
    n_stat_var_init = s_types.shape[0]

    for i in range(n_stat_var_init):
        if s_types[i, 0] == 'real':
            n_vec = []
            for j in range(s_miss.shape[0]):
                if s_miss[j, i] == 1:
                    n_vec.append(s_onehot[j, onehot_id])

            n_vec = torch.stack(n_vec)
            mean = torch.mean(n_vec)
            var = torch.var(n_vec)
            var = torch.clamp(var, min=1e-6, max=1e20)  # Prevent division by zero
            # s_data_norm[:, i] = (s_data_norm[:, i] - mean) / torch.sqrt(var)

            normalized_s_data_i = (s_data_norm[:, onehot_id] - mean) / torch.sqrt(var)
            normalized_s_data_i[s_miss[:, i] == 0.0] = 0  # Missing values set to 0
            s_data_norm[:, onehot_id] = normalized_s_data_i

            b_mean.append(mean)
            b_var.append(var)
            onehot_id += 1
        elif s_types[i, 0] == 'pos':
            n_vec = []
            for j in range(s_miss.shape[0]):
                if s_miss[j, i] == 1:
                    s_onehot_log = torch.log1p(s_onehot[j, onehot_id])
                    n_vec.append(s_onehot_log)

            n_vec = torch.stack(n_vec)
            mean = torch.mean(n_vec)
            var = torch.var(n_vec)
            var = torch.clamp(var, min=1e-6, max=1e20)  # Prevent division by zero
            # s_data_norm[:, i] = (torch.log1p(s_data_norm[:, i]) - mean) / torch.sqrt(var)

            normalized_s_data_i = (torch.log1p(s_data_norm[:, onehot_id]) - mean) / torch.sqrt(var)
            normalized_s_data_i[s_miss[:, i] == 0.0] = 0  # Missing values set to 0
            s_data_norm[:, i] = normalized_s_data_i

            b_mean.append(mean)
            b_var.append(var)
            onehot_id += 1
        else:
            onehot_id += s_types[i, 1]
            
    b_mean = torch.stack(b_mean).to(s_onehot.device)
    b_var = torch.stack(b_var).to(s_onehot.device)
    s_data_norm = s_data_norm.to(s_onehot.device)

    return s_data_norm, b_mean, b_var


def onehot_batch_norm_bis(s_onehot, s_types, s_miss):
    # Batch Normalization for the Onehot encoded static data
    n_stat_var_init = s_types.shape[0]
    s_data_norm = s_onehot.clone()
    b_mean, b_var = torch.zeros(n_stat_var_init), torch.ones(n_stat_var_init)
    onehot_id = 0

    for i in range(n_stat_var_init):
        if s_types[i, 0] == 'real':
            n_vec = []
            for j in range(s_miss.shape[0]):
                if s_miss[j, i] == 1:
                    n_vec.append(s_onehot[j, onehot_id])

            if len(n_vec) < 2:
                print(f"[WARNING] Feature {i} has only {len(n_vec)} observed sample(s), skipping normalization.")
                onehot_id += 1
                continue
            n_vec = torch.stack(n_vec)
            mean = torch.mean(n_vec)
            var = torch.var(n_vec)
            var = torch.clamp(var, min=1e-6, max=1e20)  # Prevent division by zero
            # s_data_norm[:, i] = (s_data_norm[:, i] - mean) / torch.sqrt(var)

            normalized_s_data_i = (s_data_norm[:, onehot_id] - mean) / torch.sqrt(var)
            normalized_s_data_i[s_miss[:, i] == 0.0] = 0  # Missing values set to 0
            s_data_norm[:, onehot_id] = normalized_s_data_i

            b_mean[i] = mean 
            b_var[i] = var
            onehot_id += 1
        elif s_types[i, 0] == 'pos':
            n_vec = []
            for j in range(s_miss.shape[0]):
                if s_miss[j, i] == 1:
                    s_onehot_log = torch.log1p(s_onehot[j, onehot_id])
                    n_vec.append(s_onehot_log)

            if len(n_vec) < 2:
                print(f"[WARNING] Feature {i} has only {len(n_vec)} observed sample(s), skipping normalization.")
                onehot_id += 1
                continue
            n_vec = torch.stack(n_vec)
            mean = torch.mean(n_vec)
            var = torch.var(n_vec)
            var = torch.clamp(var, min=1e-6, max=1e20)  # Prevent division by zero
            # s_data_norm[:, i] = (torch.log1p(s_data_norm[:, i]) - mean) / torch.sqrt(var)

            normalized_s_data_i = (torch.log1p(s_data_norm[:, onehot_id]) - mean) / torch.sqrt(var)
            normalized_s_data_i[s_miss[:, i] == 0.0] = 0  # Missing values set to 0
            s_data_norm[:, i] = normalized_s_data_i

            b_mean[i] = mean 
            b_var[i] = var
            onehot_id += 1
        else:
            onehot_id += s_types[i, 1]
            
    b_mean = b_mean.to(s_onehot.device)
    b_var = b_var.to(s_onehot.device)
    s_data_norm = s_data_norm.to(s_onehot.device)

    return s_data_norm, b_mean, b_var
    

    # def onehot_batch_norm(self, batch_data_list, feat_types_list, miss_list):
    #     """
    #     Normalizes real-valued data while leaving categorical/ordinal variables unchanged.

    #     Parameters:
    #     -----------
    #         batch_data_list : list of torch.Tensor
    #             List of input data tensors, each corresponding to a feature.
            
    #         feat_types_list : list of dict
    #             List specifying the type of each feature.
            
    #         miss_list : torch.Tensor
    #             Binary mask indicating observed (1) and missing (0) values.

    #     Returns:
    #     --------
    #         normalized_data : list of torch.Tensor
    #             List of normalized feature tensors.
            
    #         normalization_parameters : list of tuples
    #             Normalization parameters for each feature.
    #     """

    #     normalized_data = []
    #     normalization_mean = []
    #     normalization_var = []

    #     print(miss_list.shape, miss_list.type)
    #     print(batch_data_list.shape, batch_data_list.type)
    #     print(feat_types_list.shape, feat_types_list.type)


    #     for i, d in enumerate(batch_data_list):
    #         missing_mask = miss_list[:, i] == 0  # True for missing values, False for observed values
    #         observed_data = d[~missing_mask]  # Extract observed values

    #         feature_type = feat_types_list[i]['type']

    #         if feature_type == 'real':
    #             # Standard normalization (mean 0, std 1)
    #             data_var, data_mean = torch.var_mean(observed_data, unbiased=False)
    #             data_var = torch.clamp(data_var, min=1e-6, max=1e20)  # Prevent division by zero
                
    #             normalized_observed = (observed_data - data_mean) / torch.sqrt(data_var)
    #             normalized_d = torch.zeros_like(d)
    #             normalized_d[~missing_mask] = normalized_observed  # Assign transformed values
    #             normalized_d[missing_mask] = 0  # Missing values set to 0
                
    #             # normalization_parameters.append((data_mean, data_var))
    #             normalization_mean.append(data_mean)
    #             normalization_var.append(data_var)

            # elif feature_type == 'pos':
            #     # Log-normal transformation and normalization
            #     observed_data_log = torch.log1p(observed_data)
            #     data_var_log, data_mean_log = torch.var_mean(observed_data_log, unbiased=False)
            #     data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)

            #     normalized_observed = (observed_data_log - data_mean_log) / torch.sqrt(data_var_log)
            #     normalized_d = torch.zeros_like(d)
            #     normalized_d[~missing_mask] = normalized_observed
            #     normalized_d[missing_mask] = 0

            #     normalization_parameters.append((data_mean_log, data_var_log))

    #         else:
    #             # Keep categorical and ordinal values unchanged
    #             normalized_d = d.clone()
    #             # # normalization_parameters.append((0.0, 1.0))
    #             # normalization_mean.append(0.0)
    #             # normalization_var.append(1.0)

    #         normalized_data.append(normalized_d)

    #     normalized_data = torch.Tensor(normalized_data)

    #     return normalized_data, normalization_mean, normalization_var