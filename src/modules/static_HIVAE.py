import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import init_network_weights

# ========================================
# ============= STATIC MODELS ============
# ========================================

class HIVAE_Encoder(nn.Module):
    def __init__(self, s_vals_dim, onehot_dim, s_dim, z_dim):
        super(HIVAE_Encoder, self).__init__()

        self.s_layer = nn.Linear(onehot_dim, s_dim)  # q(s|x^o)
        self.z_mean_layer = nn.Linear(onehot_dim + s_dim, z_dim)  # q(z|s,x^o)
        self.z_logvar_layer = nn.Linear(onehot_dim + s_dim, z_dim)  # q(z|s,x^o)
        self.y_layer = nn.Linear(z_dim, s_vals_dim)

    def forward(self, static_data, tau=1):
        # Softmax over s (categorical distribution)
        logits_s = self.s_layer(static_data)
        samples_s = F.gumbel_softmax(logits_s, tau, hard=False)
        # Compute q(z|s,x^o)
        mean_qz = self.z_mean_layer(torch.cat([static_data, samples_s], dim=1))
        log_var_qz = self.z_logvar_layer(torch.cat([static_data, samples_s], dim=1))
        log_var_qz = torch.clamp(log_var_qz, -15.0, 15.0)
        # Reparametrization trick
        eps = torch.randn_like(mean_qz, device=static_data.device)
        samples_z = mean_qz + torch.exp(log_var_qz / 2) * eps
        # Compute deterministic y layer
        samples_y = self.y_layer(samples_z) 

        q_params = {"s": logits_s, "z": (mean_qz, log_var_qz)} 
        samples = {"s": samples_s, "z": samples_z, "y": samples_y}
        return q_params, samples


class HIVAE_Decoder(nn.Module):
    def __init__(self, s_vals_dim, static_types, s_dim, z_dim):
        """
        :param s_vals_dim: number of static variables
        :param static_types: types of the static variables (before one hot encoding), e.g. ['real', 1], ['cat', 3]
        :param s_dim: dimension of the static latent space
        :param z_dim: dimension of the static latent space
        """
        super(HIVAE_Decoder, self).__init__()

        self.z_dim = z_dim
        self.s_vals_dim = s_vals_dim
        self.z_distribution_layer = nn.Linear(s_dim, z_dim)  # p(z|s)
        self.h_layers = nn.ModuleList()
        for i in range(s_vals_dim):
            if static_types[i, 0] == 'real' or static_types[i, 0] == 'pos':
                self.h_layers.append(nn.Linear(s_vals_dim + s_dim, static_types[i, 1]*2))
            elif static_types[i, 0] == 'cat':
                self.h_layers.append(nn.Linear(s_vals_dim + s_dim, static_types[i, 1]-1))
        init_network_weights(self.h_layers)


    def forward(self, samples, static_onehot, static_types, static_true_miss_mask, tau=1, normalization_params=None, compute_loss=True):

        # Compute p(z|s)
        p_params = {}
        mean_pz = self.z_distribution_layer(samples["s"])
        log_var_pz = torch.zeros_like(mean_pz).to(static_onehot.device)
        p_params["z"] = (mean_pz, log_var_pz)

        # Computing the output data
        samples_out = torch.zeros(samples["y"].shape).to(static_onehot.device)
        log_p_x = torch.zeros(samples["y"].shape).to(static_onehot.device)
        onehot_id = 0
        for i in range(self.s_vals_dim):
            if static_types[i, 0] == 'real' or static_types[i, 0] == 'pos':
                data = static_onehot[:, onehot_id].clone()
                data[torch.isnan(data)] = 0

                # params = self.h_layers[i](samples["y"])
                params = self.h_layers[i](torch.cat([samples["y"], samples["s"]], dim=1))
                est_mean = params[:, 0]
                est_logvar = params[:, 1]

                if normalization_params is not None:
                    batch_mean = normalization_params['mean'][i]
                    batch_var = normalization_params['var'][i]

                    # Diagnostic prints
                    if batch_var < 1e-6:
                        print(f"[WARNING] Feature {i} ({static_types[i]}) has near-zero batch_var={batch_var.item():.8f}")
                    if torch.isnan(est_logvar).any():
                        print(f"[WARNING] Feature {i} ({static_types[i]}) has NaN in est_logvar BEFORE normalization")
                        print(f"  raw params[:, 1]: {params[:, 1].detach()}")

                    # Fix: guard log(0) with clamp
                    est_mean   = torch.sqrt(batch_var.clamp(min=1e-6)) * est_mean + batch_mean
                    log_batch_var = torch.log(batch_var.clamp(min=1e-6))
                    est_logvar = log_batch_var + est_logvar

                    if torch.isnan(est_logvar).any():
                        print(f"[WARNING] Feature {i} ({static_types[i]}) has NaN in est_logvar AFTER normalization")
                        print(f"  log_batch_var={log_batch_var.item():.4f}, batch_var={batch_var.item():.8f}")
                        print(f"  est_logvar: {est_logvar.detach()}")

                # Fix: nan_to_num before clamp so NaNs don't pass through
                est_logvar = torch.nan_to_num(est_logvar, nan=0.0, posinf=8.0, neginf=-8.0)
                est_logvar = torch.clamp(est_logvar, -10.0, 10.0)

                if compute_loss:
                    # Compute log-likelihood using the Gaussian log-likelihood formula
                    log_normalization = -0.5 * torch.log(torch.tensor(2 * torch.pi))
                    log_variance_term = -0.5 * est_logvar
                    log_exponent = -0.5 * ((data - est_mean) ** 2 / torch.exp(est_logvar))
                    miss = static_true_miss_mask[:, i].float()
                    log_p_x[:,i] = (log_exponent + log_variance_term + log_normalization) * miss

                # Fix: clamp std to always be strictly positive
                std = torch.exp(0.5 * est_logvar).clamp(min=1e-6)
                samples_out[:, i] = torch.normal(est_mean, std)
                onehot_id += 1

            else:
                data = static_onehot[:, onehot_id:onehot_id + static_types[i, 1]].clone()
                data = data.float()
                
                # logits = self.h_layers[i](samples["y"])
                logits = self.h_layers[i](torch.cat([samples["y"], samples["s"]], dim=1))
                log_pi = torch.cat([torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device), logits], dim=1)

                if compute_loss:
                    # Differentiable reconstruction log-prob (categorical log-likelihood)
                    log_probs = F.log_softmax(log_pi, dim=1)
                    log_p_x[:, i] = (data * log_probs).sum(dim=1)
                    miss = static_true_miss_mask[:, i].float()
                    log_p_x[:, i] = log_p_x[:, i] * miss

                samples_soft = F.gumbel_softmax(log_pi, tau, hard=False)
                samples_out[:,i] = torch.argmax(samples_soft.detach(), 1).float()
                
                onehot_id += static_types[i, 1]
                
        if compute_loss:
            return samples_out, p_params, log_p_x.sum(dim=1)
        else:
            return samples_out, p_params