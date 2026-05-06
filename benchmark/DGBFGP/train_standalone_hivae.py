import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import *

def train_standalone_hivae(S_Enc, S_Dec, train_dataloader, val_dataloader, static_types, s_dim_static, device, 
                           epochs=100, lr=1e-3, initial_tau=1.0, min_tau=0.1, tau_decay_rate=0.95):
    """
    Trains the HI-VAE Encoder and Decoder.
    
    Args:
        S_Enc, S_Dec: The initialized HI-VAE encoder and decoder models.
        dataloader: PyTorch DataLoader yielding batches of your formatted static data 
                    (W_onehot, W_true_miss_mask, W_onehot_mask).
        static_types: The types of the static features.
        s_dim_static: The dimension of the categorical latent space.
        device: 'cuda' or 'cpu'.
        epochs: Number of training epochs.
        lr: Learning rate.
        initial_tau, min_tau, tau_decay_rate: Parameters for temperature annealing.
    """
    optimizer = optim.Adam(list(S_Enc.parameters()) + list(S_Dec.parameters()), lr=lr)
    S_Enc.train()
    S_Dec.train()
    
    tau = initial_tau
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_loss_val = 0.0
        
        S_Enc.train()
        S_Dec.train()
        # train_progress_bar = tqdm(enumerate(train_dataloader), unit_scale=True, total=len(train_dataloader), desc="Training")
        for batch_data_stat in train_dataloader: 
            batch_data_stat = [d.to(device) for d in batch_data_stat]
            optimizer.zero_grad()
            _, ELBO = static_forward_pass(
                S_Enc, S_Dec, batch_data_stat, tau, static_types, s_dim_static, device
            )
            ELBO.backward()
            optimizer.step()
            epoch_loss += ELBO.item()

        S_Enc.eval()
        S_Dec.eval()
        # val_progress_bar = tqdm(enumerate(val_dataloader), desc="Validation")   
        with torch.no_grad():
            for val_data_stat in val_dataloader:
                val_data_stat = [d.to(device) for d in val_data_stat]
                _, ELBO_val = static_forward_pass(
                    S_Enc, S_Dec, val_data_stat, tau, static_types, s_dim_static, device
                )    
                epoch_loss_val += ELBO_val.item()

        # tau = max(min_tau, tau * tau_decay_rate)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_dataloader)
            avg_loss_val = epoch_loss_val / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] | Average ELBO Loss Train: {avg_loss:.4f} | Average ELBO Loss Val: {avg_loss_val:.4f} | Tau: {tau:.4f}")
            
    print("HI-VAE Training Complete!")
    return S_Enc, S_Dec


def static_forward_pass(S_Enc, S_Dec, data, tau, static_types, s_dim_static, device, batch_norm_static=True, return_pred_stat=False):
    """
    Your provided function, adapted to take S_Enc and S_Dec as arguments.
    """
    W_onehot = data[0].to(device)
    W_types = static_types
    W_true_miss_mask = data[1].to(device)
    
    if batch_norm_static:
        W_onehot_mask = data[2].to(device)
        # Assumes onehot_batch_norm_bis is imported and available in your script
        W_onehot_batch_norm, W_batch_mean, W_batch_var = onehot_batch_norm_bis(W_onehot, W_types, W_onehot_mask)
        batch_normalization_params = {'mean': W_batch_mean, 'var': W_batch_var}
    else:
        W_onehot_batch_norm = W_onehot.clone()
        batch_normalization_params = None

    # Encode 
    q_params, samples = S_Enc(W_onehot_batch_norm, tau)     

    # Decode 
    pred_W, p_params, log_p_x = S_Dec(samples, W_onehot, W_types, W_true_miss_mask, tau, batch_normalization_params)
    
    # Loss
    eps = 1e-20
    log_pi = q_params['s']
    pi_param = F.softmax(log_pi, dim=-1)
    
    # Ensure the s_dim_static tensor is on the correct device to avoid crash
    KL_s = torch.sum(pi_param * torch.log(pi_param + eps), dim=1) + torch.log(torch.tensor(float(s_dim_static)).to(device))
    
    mean_qz, log_var_qz = q_params['z']
    mean_pz, log_var_pz = p_params['z']
    
    # Assumes kl_gaussian is imported and available in your script
    KL_z = kl_gaussian(mean_qz, log_var_qz, mean_pz, log_var_pz)
    
    num_obs = W_true_miss_mask.sum(dim=1).clamp_min(1.0)
    loss_reconstruction = log_p_x / num_obs
    
    ELBO = -torch.mean(loss_reconstruction - KL_z - KL_s, dim=0) 
    z_stat = torch.cat([samples['z'], samples['s']], dim=1)

    if return_pred_stat:
        return z_stat, pred_W, ELBO
    else:
        return z_stat, ELBO
    


def kl_gaussian(mean_q, logvar_q, mean_p, logvar_p):
    var_ratio = torch.exp(logvar_q - logvar_p)
    mean_diff_sq = (mean_p - mean_q).pow(2) / torch.exp(logvar_p)
    return 0.5 * torch.sum(
        var_ratio + mean_diff_sq - 1 + logvar_p - logvar_q,
        dim=1
    )