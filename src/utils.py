import torch
import numpy as np
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


               
def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, val=0)


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
                ax1.set_ylabel("Loss")
                ax1.tick_params(axis='y')
                ax2.set_ylabel("MSE")
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
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(img_name)
                plt.close()

                for k in loss_df.keys():
                    if k != 'Epoch' and ('train' not in k and 'Loss' not in k and 'MSE' not in k and 'smooth' not in k):
                        plt.plot(time_vect, loss_df[k].values.tolist(), label=k, linewidth=2)
                img_name = log_name[:-4] + '_validation.png'
                plt.legend()
                plt.title('All Raw Losses Validation Set')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(img_name)
                plt.close()

                for k in loss_df.keys():
                    if k != 'Epoch' and ('train' not in k and 'Loss' not in k and 'MSE' not in k and 'smooth' in k):
                        plt.plot(time_vect, loss_df[k].values.tolist(), label=k, linewidth=2)
                img_name = log_name[:-4] + '_smooth_validation.png'
                plt.legend()
                plt.title('All Smooth Losses Validation Set')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(img_name)
                plt.close()



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


def subtract_initial_point(paths):
    _, length, dim = paths.size()
    res = paths.clone()
    start_points = torch.transpose(res[:, 0, 1:].unsqueeze(-1), -1, 1)
    res[..., 1:] -= torch.tile(start_points, (1, length, 1))
    return res


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
    
