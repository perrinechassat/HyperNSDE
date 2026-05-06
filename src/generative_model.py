import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from collections import OrderedDict
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_nonadjoint
import torchsde
# import torchcde
from opacus import PrivacyEngine
from src.losses import *
from src.utils import print_current_losses, plot_final_losses, time_normalisation_transform, onehot_batch_norm, onehot_batch_norm_bis, subtract_initial_point, RunningAverageMeter, generate_mask_grid_from_inhomogeneous_poisson
from src.build_module import Module
warnings.filterwarnings('ignore')


class Generative_Model_Longi_Static:
    def __init__(self, config, static_types, hyperopt_mode=False, init_x=None, init_mask=None):

        self.config = config
        self.static_types = static_types
        self.module = Module(config, self.static_types, init_x, init_mask)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_optimizer()
        self.train_loss = RunningAverageMeter(momentum=0.97)
        self.train_mse = RunningAverageMeter(momentum=0.97)
        self.val_loss = RunningAverageMeter(momentum=0.97)
        self.val_mse = RunningAverageMeter(momentum=0.97)
        self.val_losses_dict = {task: RunningAverageMeter(momentum=0.97) for task in self.module.tasks}
        self.load_models(hyperopt_mode=hyperopt_mode) # Load models from checkpoint if existsa
        self.hyperopt_mode = hyperopt_mode
        print(f"Epoch init: {self.config.epoch_init}, Best loss: {self.best_loss}")


    def set_optimizer(self):
        params_opt = []
        params_opt.append({'params': self.module.L_Dec.parameters(), 'weight_decay': 0.0})
        if self.config.sde:
            if self.config.sde_split_training:
                if self.config.latent_model == 'HyperNDEs':
                    params_opt += [{'params': self.module.L_Latent._hypernetwork_drift.parameters(), 'weight_decay': self.config.weight_decay_drift}, # 0.0},
                                    {'params': self.module.L_Latent._residual.parameters(), 'weight_decay': self.config.weight_decay_diff}]
                else:
                    params_opt += [{'params': self.module.L_Latent._drift.parameters(), 'weight_decay': self.config.weight_decay_drift},
                                    {'params': self.module.L_Latent._residual.parameters(), 'weight_decay': self.config.weight_decay_diff}]
            else:
                if self.config.latent_model == 'HyperNDEs':
                    params_opt += [{'params': self.module.L_Latent._hypernetwork_drift.parameters(), 'weight_decay': self.config.weight_decay_drift}, 
                                   {'params': self.module.L_Latent._latent_sde.parameters(), 'weight_decay': self.config.weight_decay_diff}]
                else:
                    params_opt += [{'params': self.module.L_Latent._latent_sde.parameters(), 'weight_decay': self.config.weight_decay_diff}]
        else:
            if self.config.latent_model == 'HyperNDEs':
                params_opt += [{'params': self.module.L_Latent._hypernetwork_drift.parameters(), 'weight_decay': self.config.weight_decay_drift}]
            else:
                params_opt += [{'params': self.module.L_Latent._drift.parameters(), 'weight_decay': self.config.weight_decay_drift}]
        if self.config.static_data:
            params_opt += [{'params': self.module.S_Enc.parameters(), 'weight_decay': 0.0}, 
                            {'params': self.module.S_Dec.parameters(), 'weight_decay': 0.0}]
        if self.config.type_enc != 'none':
            params_opt += [{'params': self.module.L_Enc.parameters(), 'weight_decay': 0.0},
                            {'params': self.module.Imp_Layer.parameters(), 'weight_decay': 0.0}]
        else:
            if self.config.fixed_init_cond == False:
                params_opt += [{'params': self.module.L_Latent._initial.parameters(), 'weight_decay': 0.0, 'lr': self.config.lr * 10}]
        if self.config.estim_event_rate:
            params_opt += [{'params': self.module.L_Latent._event_rate.parameters(), 'weight_decay': 0.0}]
        
        self.optimizer = torch.optim.AdamW(params_opt, lr=self.config.lr)


    def load_models(self, hyperopt_mode=False):
        """Load model and optimizer state."""
        if hyperopt_mode:
            self.best_loss = float('inf')
            return 
        else:
            ckpt_path = None
            if self.config.from_best:
                ckpt_path = os.path.join(self.config.save_path_models, 'ckpt_best.pth')
            elif self.config.epoch_init > 1:
                ckpt_path = os.path.join(self.config.save_path_models, f"ckpt_{self.config.epoch_init}.pth")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(self.config.save_path_models, "ckpt_latest.pth")
            else:
                ckpt_path = os.path.join(self.config.save_path_models, "ckpt_latest.pth")

            if ckpt_path and os.path.exists(ckpt_path):
                map_location = torch.device('cuda') if self.config.GPU else torch.device('cpu')
                ckpt = torch.load(ckpt_path, map_location=map_location)
                print(f"Loaded checkpoint from {ckpt_path}, epoch: {ckpt['Epoch']}")

                self.module.load_state_dict(ckpt['model_state'])
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
                self.config.epoch_init = ckpt['Epoch']
                self.best_loss = ckpt['Val Loss']
                self.train_loss.set_from_values(ckpt['Train Loss'])
                self.train_mse.set_from_values(ckpt['Train MSE'])
                self.val_loss.set_from_values(ckpt['Val Loss'])
                self.val_mse.set_from_values(ckpt['Val MSE'])
                for task in self.module.tasks:
                    self.val_losses_dict[task].set_from_values(ckpt['Val Losses Dict'][task])
            else:
                print("No checkpoint found. Starting from scratch.")
                self.best_loss = float('inf')

        
    def save_models(self, epoch, train_loss, val_loss, best=False):
        
        ckpt = {}
        ckpt['model_state'] = self.module.state_dict()
        ckpt['optimizer_state'] = self.optimizer.state_dict()
        ckpt['Train Loss'] = train_loss
        ckpt['Val Loss'] = val_loss
        ckpt['Epoch'] = epoch
        ckpt['Train MSE'] = self.train_mse.avg
        ckpt['Val MSE'] = self.val_mse.avg
        ckpt['Val Losses Dict'] = {}
        for task in self.module.tasks:
            ckpt['Val Losses Dict'][task] = self.val_losses_dict[task].avg
        
        if best:
            torch.save(ckpt, os.path.join(self.config.save_path_models, 'ckpt_best.pth'))
        else:
            torch.save(ckpt, os.path.join(self.config.save_path_models, 'ckpt_%d.pth'%(epoch)))
            torch.save(ckpt, os.path.join(self.config.save_path_models, 'ckpt_latest.pth'))

        print('Models have been saved.')

    # =============================== Static encoder / decoder ===============================

    def static_encoder_decoder(self, data, tau, sigma=1.0, return_pred_stat=False):
        W_onehot = data[0].to(self.device)
        W_types = self.static_types
        W_true_miss_mask = data[1].to(self.device)
        if self.config.batch_norm_static:
            W_onehot_mask = data[2].to(self.device)
            W_onehot_batch_norm, W_batch_mean, W_batch_var = onehot_batch_norm_bis(W_onehot, W_types, W_onehot_mask)
            batch_normalization_params = {'mean': W_batch_mean, 'var': W_batch_var}
        else:
            W_onehot_batch_norm = W_onehot.clone()
            batch_normalization_params = None

        # Encode 
        q_params, samples = self.module.S_Enc(W_onehot_batch_norm, tau)     

        # Decode 
        pred_W, p_params, log_p_x = self.module.S_Dec(samples, W_onehot, W_types, W_true_miss_mask, tau, batch_normalization_params)
        # Loss
        # -- KL(q(s|x) || p(s))
        eps = 1e-20
        log_pi = q_params['s']
        pi_param = F.softmax(log_pi, dim=-1)
        KL_s = torch.sum(pi_param * torch.log(pi_param + eps), dim=1) + torch.log(torch.tensor(float(self.config.s_dim_static)))
        # -- KL(q(z|s,x) || p(z|s))
        mean_qz, log_var_qz = q_params['z']
        mean_pz, log_var_pz = p_params['z']
        KL_z = kl_gaussian(mean_qz, log_var_qz, mean_pz, log_var_pz)
        # -- Expectation of log p(x|y)
        num_obs = W_true_miss_mask.sum(dim=1).clamp_min(1.0)
        loss_reconstruction = log_p_x / num_obs
        # -- Complete ELBO
        ELBO = - torch.mean(loss_reconstruction - KL_z - KL_s, dim=0) 
        z_stat = torch.cat([samples['z'], samples['s']], dim=1)

        if return_pred_stat:
            return z_stat, pred_W, ELBO
        else:
            return z_stat, ELBO

    # =============================== Long encoder ===============================

    def long_encoder(self, x, mask, sigma=1.0):
        if ~torch.all(mask == 1.0):
            x_mask = self.module.Imp_Layer(x, mask)
        else:
            x_mask = x.clone()

        out = self.module.L_Enc(x_mask)
        qz0_mean, qz0_logvar = out[:, :self.module.latent_dim_long], out[:, self.module.latent_dim_long:]
        eps = sigma * torch.randn(qz0_mean.size()).to(self.device)
        z0_long = eps * torch.exp(.5 * qz0_logvar) + qz0_mean
            
        pz0_mean = pz0_logvar = torch.zeros(z0_long.size()).to(self.device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
        KL_avg = torch.mean(analytic_kl)
        
        return z0_long, KL_avg

    # =============================== Long latent model ===============================
    
    def long_latent_model(self, z0_long, time_grid, z_stat=None):
        if self.config.solver == 'Adjoint':
            odeint = odeint_adjoint 
        else:
            odeint = odeint_nonadjoint

        z0_ode = z0_long.to(self.device)

        # Manage static variables dependence as function of the latent model type
        if self.config.latent_model == 'MultiNDEs':
            z0_ode = torch.cat((z0_long, z_stat), dim=1).to(self.device)
            z0_long = z0_ode
        elif self.config.latent_model == 'StatMoNDEs':
            self.module.L_Latent.augment(z_stat)
        elif self.config.latent_model == 'HyperNDEs':
            weights_ode, biases_ode = self.module.L_Latent._hypernetwork_drift(z_stat.to(self.device))
            self.module.L_Latent._drift.set_params_model(weights_ode, biases_ode)
        else: 
            raise ValueError(f"Unknown latent_model_type: {self.config.latent_model}")

        if self.config.sde: 
            if self.config.sde_split_training:
                # solve deterministic ODE part first
                pred_z = odeint(self.module.L_Latent._drift, z0_ode, time_grid, rtol=self.config.rtol, atol=self.config.atol, method=self.config.method_solver) # Use regular integration
                pred_z = pred_z.permute(1,0,2) # n_traj, n_timepoints, n_dims
                # solve stochastic SDE part
                pred_r = torchsde.sdeint(self.module.L_Latent._residual, z0_ode, time_grid, dt=1e-2, logqp=False, method='reversible_heun')
                pred_r = pred_r.permute(1,0,2)
            else: 
                pred_z = torchsde.sdeint(self.module.L_Latent._latent_sde, z0_ode, time_grid, dt=1e-2, logqp=False, method='reversible_heun')
                pred_z = pred_z.permute(1,0,2)
                pred_r = torch.zeros_like(pred_z, device=self.device)
        else: 
            pred_z = odeint(self.module.L_Latent._drift, z0_ode, time_grid, rtol=self.config.rtol, atol=self.config.atol, method=self.config.method_solver) # Use regular integration
            pred_z = pred_z.permute(1,0,2) # n_traj, n_timepoints, n_dims
            pred_r = torch.zeros_like(pred_z, device=self.device)

        if self.config.latent_model != 'MultiNDEs':
            pred_z = pred_z + z0_long.unsqueeze(1)

        return pred_z, pred_r

    # =============================== Long decoder ===============================

    def long_decoder(self, pred_z, z_stat=None, time_grid=None):
        pred_x = self.module.L_Dec(pred_z)
        return pred_x
    
    # =============================== Compute only static loss =========================

    def compute_static_loss(self, data, tau):
        static_data = data[3:]
        losses_dict = OrderedDict()
        if self.config.static_data:
            z_stat, loss_stat = self.static_encoder_decoder(static_data, tau)
            losses_dict['Static'] = loss_stat # * self.config.loss_scaling_static
        return loss_stat, losses_dict
        
    # =============================== Compute all losses ===============================
    
    def _compress_batch(self, x, mask, fill='last', align='left'):
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
    

    def _compute_sde_loss(self, pred, target, time_grid, mask):
        time_grid_norm = (time_grid - time_grid.min()) / (time_grid.max() - time_grid.min())
        T_expand = time_grid_norm.unsqueeze(0).unsqueeze(-1).expand(pred.size(0), time_grid_norm.size(0), 1) 
        if self.config.sde_training == 'MMD_PySig':
            pred_xT = torch.cat([T_expand, pred], dim=2).to(self.device)
            target_xT = torch.cat([T_expand, target], dim=2).to(self.device)    
        else:
            raise ValueError(f"Unknown sde_training method: {self.config.sde_training}")          
        # else:
        #     pred_xT = torchcde.linear_interpolation_coeffs(torch.cat([T_expand, pred], dim=2)).to(self.device)
        #     target_xT = torchcde.linear_interpolation_coeffs(torch.cat([T_expand, target], dim=2)).to(self.device)

        if self.config.subtract_initial_point:
            pred_xT = subtract_initial_point(pred_xT).to(self.device)
            target_xT = subtract_initial_point(target_xT).to(self.device)

        pred_xT_c = self._compress_batch(pred_xT, mask)
        target_xT_c = self._compress_batch(target_xT, mask)

        loss_sde = self.module.discriminator(pred_xT_c, target_xT_c.detach()).to(self.device)
        return loss_sde
    

    def _reshape_obs(self, x_obs, mask):
        mask_any = mask.any(dim=-1)  # (B,T)
        x_reshape = torch.zeros_like(mask)
        x_reshape[mask_any] = x_obs
        x_reshape[mask == 0] = 0.0
        return x_reshape


    def compute_all_loss(self, data, tau, scale_mse=False):        
        time_grid = data[0][0].float() # Because same time grid for all in the batch
        x, mask = data[1].float() , data[2].float()
        static_data = data[3:]

        losses_dict = OrderedDict()

        # Static 
        if self.config.static_data:
            z_stat, loss_stat = self.static_encoder_decoder(static_data, tau)
            losses_dict['Static'] = loss_stat 

        # Initial condition
        if self.config.type_enc != 'none':
            z0_long, KL_z0 = self.long_encoder(x, mask)
        else:
            z0_long = torch.zeros(x.shape[0], self.module.latent_dim_long, device=self.device).float()
            KL_z0 = torch.tensor(0.).to(self.device)

        # Latent dynamics
        pred_z, pred_r = self.long_latent_model(z0_long, time_grid, z_stat)

        if self.config.estim_event_rate:
            int_lambda, log_lambda = self.module.L_Latent._event_rate.compute_event_rate_complete(pred_z, time_grid.to(self.device), mask, method="trapezoidal")
            poisson_log_l = compute_poisson_proc_likelihood(log_lambda, int_lambda, mask = mask, time_grid=time_grid, scale=True)
            loss_poisson = torch.mean(-poisson_log_l, dim=0)
            losses_dict['Poisson'] = loss_poisson.to(self.device) 

        # Decoder
        if self.config.sde:
            if self.config.sde_split_training:
                mask_any = mask.any(dim=-1)  # (B,T)
                pred_z_obs = pred_z[mask_any]
                pred_x_mean_obs = self.long_decoder(pred_z_obs, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))
                pred_x_mean = self._reshape_obs(pred_x_mean_obs, mask)
                loss_ode = mse(x, pred_x_mean, mask=mask)
                losses_dict['Longi ODE'] = loss_ode + KL_z0 
                pred_r_obs = pred_r[mask_any]
                pred_x_residu_obs = self.long_decoder(pred_r_obs, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))
                target_x_residu_obs = x[mask_any] - pred_x_mean_obs.detach()
                pred_x_residu = self._reshape_obs(pred_x_residu_obs, mask)
                target_x_residu = self._reshape_obs(target_x_residu_obs, mask)
                loss_sde = self._compute_sde_loss(pred_x_residu, target_x_residu, time_grid, mask)
                losses_dict['Longi SDE'] = loss_sde    
            else:
                mask_any = mask.any(dim=-1)  # (B,T)
                pred_z_obs = pred_z[mask_any]
                pred_x_obs = self.long_decoder(pred_z_obs, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))                
                pred_x = self._reshape_obs(pred_x_obs, mask)
                loss_ode = mse(x, pred_x, mask=mask) 
                loss_sde = self._compute_sde_loss(pred_x, x, time_grid, mask)
                losses_dict['Longi SDE'] = loss_sde

        else: 
            pred_x = self.long_decoder(pred_z, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))
            loss_ode = mse(x, pred_x, mask=mask) 

            if scale_mse:
                loss_ode_scale = loss_ode * x.size(1) * x.size(2)
                losses_dict['Longi ODE'] = loss_ode_scale + KL_z0
            else:
                losses_dict['Longi ODE'] = loss_ode + KL_z0

        mse_longi = loss_ode.detach() 
        return mse_longi, losses_dict
    
    # =============================== Train model ===============================

    def train(self, train_dataloader, val_dataloader, save_model=True, differential_privacy=False):

        global_steps = 0
        total_time = time.time()
        self.best_losses = {task: self.best_loss for task in self.module.tasks}
        self.counter = {task: 0 for task in self.module.tasks}

        if differential_privacy:
            privacy_engine = PrivacyEngine()
            self.module, self.optimizer, train_dataloader = privacy_engine.make_private(
                module=self.module,
                optimizer=self.optimizer,
                data=train_dataloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
        
        nb_batches_train = len(train_dataloader)
        nb_batches_val = len(val_dataloader)

        ## --- Scheduler ---
        n_epochs_scheduler = 600 # ~ expected early-stop max length
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=self.config.lr, 
                                                             steps_per_epoch=len(train_dataloader),
                                                             epochs=n_epochs_scheduler,
                                                             pct_start=0.3, 
                                                             final_div_factor=1e3)

        for epoch in range(self.config.epoch_init, self.config.num_epochs + 1):
            tau = max(1.0 - (0.999/(self.config.num_epochs - 50 + 1e-6)) * (epoch), 1e-3)
            epoch_time_init = time.time()

            ## --- Training ---
            epoch_loss_train = torch.tensor(0.0, device=self.device)
            epoch_mse_train = torch.tensor(0.0, device=self.device)
            epoch_dict_train_losses = OrderedDict()
            for key in self.module.tasks:
                epoch_dict_train_losses[key + ' train'] = torch.tensor(0.0, device=self.device)
            epoch_dict_train_losses["Global train"] = torch.tensor(0.0, device=self.device)

            self.module.train()  
            train_progress_bar = tqdm(enumerate(train_dataloader), unit_scale=True, total=len(train_dataloader), desc="Training")
            for iter, train_data in train_progress_bar:    
                train_data = [d.to(self.device) for d in train_data]
                global_steps += 1

                self.optimizer.zero_grad()
                mse_longi_train, losses_dict_train = self.compute_all_loss(train_data, tau, scale_mse=False)

                ## --- Loss aggregation ---
                agg_loss_train = self.module.agg_func.forward(losses_dict_train)

                ## --- Backpropagation ---
                agg_loss_train.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.config.max_norm)
                self.optimizer.step()
                if epoch <= n_epochs_scheduler:
                    self.scheduler.step()

                ## --- Tracking losses ---
                epoch_loss_train += agg_loss_train.detach()
                epoch_mse_train += mse_longi_train
                losses_dict_train_weighted = self.module.agg_func.compute_dict_weighted(losses_dict_train)
                for key in self.module.tasks:
                    epoch_dict_train_losses[key + ' train'] += losses_dict_train_weighted[key].detach()
                    # epoch_dict_train_losses[key + ' train'] += losses_dict_train[key].item()
                epoch_dict_train_losses['Global train'] += agg_loss_train.detach()

            self.train_loss.update(epoch_loss_train.item()/nb_batches_train)
            self.train_mse.update(epoch_mse_train.item()/nb_batches_train)
            for key in self.module.tasks:
                    epoch_dict_train_losses[key + ' train'] = epoch_dict_train_losses[key + ' train'].item()/nb_batches_train
            epoch_dict_train_losses['Global train'] = epoch_dict_train_losses['Global train'].item()/nb_batches_train

            ## --- Validation ---
            epoch_loss_val = torch.tensor(0.0, device=self.device)
            epoch_mse_val = torch.tensor(0.0, device=self.device)
            epoch_dict_val_losses = OrderedDict()
            for key in self.module.tasks:
                epoch_dict_val_losses[key + ' val'] = torch.tensor(0.0, device=self.device)
            epoch_dict_val_losses["Global val"] = torch.tensor(0.0, device=self.device)

            self.module.eval()
            val_progress_bar = tqdm(enumerate(val_dataloader), desc="Validation")
            with torch.no_grad():
                for iter, val_data in val_progress_bar:
                    val_data = [d.to(self.device) for d in val_data]
                    mse_longi_val, losses_dict_val = self.compute_all_loss(val_data, tau, scale_mse=False)
                    ## --- Loss aggregation ---
                    agg_loss_val = self.module.agg_func.forward(losses_dict_val)

                    epoch_loss_val += agg_loss_val.detach()
                    epoch_mse_val += mse_longi_val
                    epoch_dict_val_losses['Global val'] += agg_loss_val.detach()

                    for key in self.module.tasks:
                        epoch_dict_val_losses[key + ' val'] += losses_dict_val[key].detach() 

            self.val_loss.update(epoch_loss_val.item()/nb_batches_val)
            self.val_mse.update(epoch_mse_val.item()/nb_batches_val)
            epoch_dict_val_losses['Global val'] = epoch_dict_val_losses['Global val'].item()/nb_batches_val
            for key in self.module.tasks:
                epoch_dict_val_losses[key + ' val'] = epoch_dict_val_losses[key + ' val'].item()/nb_batches_val
                self.val_losses_dict[key].update(epoch_dict_val_losses[key + ' val'])

            ## --- Print losses --- 
            if (epoch) % self.config.print_freq == 0: 
                # Train losses 
                epoch_dict_train_losses['Loss train'] = self.train_loss.avg 
                epoch_dict_train_losses['MSE train'] = self.train_mse.avg
                # Val losses 
                epoch_dict_val_losses['Loss val'] = self.val_loss.avg
                epoch_dict_val_losses['MSE val'] = self.val_mse.avg
                for key in self.module.tasks:
                    epoch_dict_val_losses[key + ' smooth val'] = self.val_losses_dict[key].avg
               
                t_epoch = time.time() - epoch_time_init
                t_total = time.time() - total_time
                
                print_current_losses(epoch, global_steps, epoch_dict_train_losses, epoch_dict_val_losses, t_epoch, t_total, self.config.save_path_losses, s_excel=True, save_losses=save_model)

            if (epoch) % self.config.save_freq == 0 and save_model:
                self.save_models(epoch, self.train_loss.avg, self.val_loss.avg)
                self.config.epoch_init = epoch
            
            ## --- Early stopping ---
            if epoch > 50:
                stop_flags = []
                any_improved = False # track if at least one task improved
                for task in self.module.tasks:
                    if self.val_losses_dict[task].avg < self.best_losses[task]:
                        self.best_losses[task] = self.val_losses_dict[task].avg
                        self.counter[task] = 0
                        self.best_loss = self.val_loss.avg
                        any_improved = True 
                    else:
                        self.counter[task] += 1
                    stop_flags.append(self.counter[task] >= self.config.patience)
                if save_model and any_improved:
                    self.save_models(epoch, self.train_loss.avg, self.val_loss.avg, best=True)
                if all(stop_flags):
                    print("Early stopping triggered.")
                    break  
            else:
                for task in self.module.tasks:
                    if self.val_losses_dict[task].avg < self.best_losses[task]:
                        self.best_losses[task] = self.val_losses_dict[task].avg
                        self.best_loss = self.val_loss.avg
        
        if save_model:
            self.save_models(epoch, self.train_loss.avg, self.val_loss.avg)
        self.config.epoch_init = epoch

        if self.hyperopt_mode:
            return self.best_losses
    

    # =============================== Generate samples ===============================

    @torch.no_grad()
    def generate(self, dataloader, n_generated_samples=None, type_gen='prior', from_drift_only=False, sigma_stat=1.0, sigma_long=1.0, save=True, name_save=None):
        """
        Generates data from the model.
        """
        self.module.eval()
        if type_gen not in ['prior', 'posterior', 'reconstruction']:
            raise ValueError(f"Unknown generation type: {type_gen}. Must be one of: 'prior', 'posterior', 'reconstruction'")

        time_grid = dataloader.dataset.get_T().to(self.device)
        x, mask = dataloader.dataset.get_x_mask()
        x, mask = x.to(self.device), mask.to(self.device)
        static_data = dataloader.dataset.get_onehot_static()
        var_names_save, var_names_static = dataloader.dataset.get_var_names()
        tau = 1e-3
        
        if n_generated_samples is None:
            n_generated_samples = x.shape[0]

        if type_gen == 'reconstruction':
            n_generated_samples = x.shape[0]
        elif type_gen == 'posterior':
            n_init_samples = x.shape[0]
            n_population_to_gen = n_generated_samples // n_init_samples + int(n_generated_samples%n_init_samples != 0.0)
            set_idx = torch.cat([torch.arange(0, n_init_samples, 1)]*n_population_to_gen, dim=0)
            idx_gen, _ = torch.sort(torch.randperm(len(set_idx))[:n_generated_samples])

        # Static
        if self.config.static_data:
            if type_gen == 'prior':
                W_onehot = static_data[0]
                W_types = self.static_types
                W_true_miss_mask = static_data[1]
                if self.config.batch_norm_static:
                    W_onehot_mask = static_data[2]
                    _, W_batch_mean, W_batch_var = onehot_batch_norm_bis(W_onehot, W_types, W_onehot_mask)
                    batch_normalization_params = {'mean': W_batch_mean, 'var': W_batch_var}
                else:
                    batch_normalization_params = None
                s_samples = torch.randint(0, self.config.s_dim_static, (n_generated_samples,))
                samples_s = torch.nn.functional.one_hot(s_samples, num_classes=self.config.s_dim_static).float().to(self.device)
                
                mean_pz = self.module.S_Dec.z_distribution_layer(samples_s)
                log_var_pz = torch.zeros_like(mean_pz).clamp(min=-15.0, max=15.0)
                eps = torch.randn_like(mean_pz)
                samples_z = (mean_pz + torch.exp(log_var_pz / 2) * eps).to(self.device)  # mean_pz + eps
                samples_y = self.module.S_Enc.y_layer(samples_z)
                samples = {"s": samples_s, "z": samples_z, "y": samples_y.to(self.device)}

                z_stat = torch.cat([samples['z'], samples['s']], dim=1).to(self.device)
                static_types = dataloader.dataset.get_static_types()
                pred_stat, _ = self.module.S_Dec(samples, W_onehot, static_types, W_true_miss_mask, tau, batch_normalization_params, compute_loss=False)

            elif type_gen == 'reconstruction':
                z_stat, pred_stat, _ = self.static_encoder_decoder(static_data, tau, sigma=1.0, return_pred_stat=True) 

            else: # 'posterior' 
                z_stat = []
                pred_stat = []
                for i in range(n_population_to_gen): 
                    z_stat_i, pred_stat_i, _ = self.static_encoder_decoder(static_data, tau, sigma=sigma_stat, return_pred_stat=True)
                    z_stat.append(z_stat_i)
                    pred_stat.append(pred_stat_i)
                z_stat = torch.cat(z_stat, dim=0)
                z_stat = z_stat[idx_gen].to(self.device)
                pred_stat = torch.cat(pred_stat, dim=0)
                pred_stat = pred_stat[idx_gen].to(self.device)
        else:
            z_stat = None
        
        # Initial condition for longitudinal
        if self.config.type_enc != 'none':
            if type_gen == 'prior':
                z0_long = np.random.normal(size=(n_generated_samples, self.module.latent_dim_long))
                z0_long = torch.from_numpy(z0_long).float().to(self.device)
            elif type_gen == 'reconstruction':
                z0_long, _ = self.long_encoder(x, mask, sigma=1.0)
            else: # 'posterior'
                z0_long = []
                for i in range(n_population_to_gen):
                    z0_long_i, _ = self.long_encoder(x, mask, sigma=sigma_long)
                    z0_long.append(z0_long_i)
                z0_long = torch.cat(z0_long, dim=0)[idx_gen]
        else:
            if self.config.fixed_init_cond == True:
                z0_long = torch.zeros(n_generated_samples, self.module.latent_dim_long, device=self.device)
            else:
                init_noise = torch.randn(n_generated_samples, self.config.init_noise_size, device=self.device)
                if z_stat is not None:
                    z0_long = self.module.L_Latent._initial(torch.cat([init_noise, z_stat], dim=1))
                else:
                    z0_long = self.module.L_Latent._initial(init_noise)

        # Latent dynamics
        if (from_drift_only == True) and (self.config.sde == True) and (self.config.sde_split_training == False):
            # raise ValueError("from_drift_only cannot be True when sde=True and sde_split_training=False")
            if self.config.solver == 'Adjoint':
                odeint = odeint_adjoint 
            else:
                odeint = odeint_nonadjoint
            z0_ode = torch.zeros(z0_long.shape, device=self.device)
            if self.config.latent_model == 'MultiNDEs':
                z0_ode = torch.cat((z0_ode, z_stat), dim=1).to(self.device)
            elif self.config.latent_model == 'StatMoNDEs':
                self.module.L_Latent.augment(z_stat)
            elif self.config.latent_model == 'HyperNDEs':
                weights_ode, biases_ode = self.module.L_Latent._hypernetwork_drift(z_stat.to(self.device))
                self.module.L_Latent._drift.set_params_model(weights_ode, biases_ode)
            else: 
                raise ValueError(f"Unknown latent_model_type: {self.config.latent_model}")
            pred_z = odeint(self.module.L_Latent._latent_sde.f, z0_ode, time_grid, rtol=self.config.rtol, atol=self.config.atol, method=self.config.method_solver) # Use regular integration
            pred_z = pred_z.permute(1,0,2) + z0_long.unsqueeze(1)
            pred_r = torch.zeros_like(pred_z, device=self.device)
        else:
            pred_z, pred_r = self.long_latent_model(z0_long, time_grid=time_grid, z_stat=z_stat)
        
        int_lambda = None
        if self.config.estim_event_rate:
            T_expanded = time_grid.view(1, -1, 1).expand(pred_z.shape[0], -1, 1).to(self.device)
            pred_z_with_t = torch.cat([pred_z, T_expanded], dim=-1)
            log_lambda = self.module.L_Latent._event_rate.log_lambda_net(pred_z_with_t)
           
            lambda_T = torch.exp(log_lambda) # n_traj, n_timepoints, n_features_long
            if type_gen == 'reconstruction': 
                mask_gen = mask
            else:
                mask_gen, final_times = generate_mask_grid_from_inhomogeneous_poisson(lambda_T, time_grid)
                mask_gen = mask_gen.expand((mask_gen.shape[0], mask_gen.shape[1], x.shape[-1])) 
        else:   
            log_lambda = None
            if type_gen == 'reconstruction':
                mask_gen = mask
            else:
                mask_gen = torch.ones((pred_z.shape[0], mask.shape[1], mask.shape[-1]), device=self.device)
        
        if self.config.sde and from_drift_only == False:
            if self.config.sde_split_training:
                pred_x_mean = self.long_decoder(pred_z, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))
                pred_x_residu = self.long_decoder(pred_r, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))
                pred_x = pred_x_mean + pred_x_residu
                pred_x[~mask_gen.bool()] = torch.nan
            else:
                pred_x = self.long_decoder(pred_z, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))
                pred_x[~mask_gen.bool()] = torch.nan
        else:
            pred_x = self.long_decoder(pred_z, z_stat=z_stat.to(self.device), time_grid=time_grid.to(self.device))
            pred_x[~mask_gen.bool()] = torch.nan

        if name_save is not None:
            name = name_save
        else:
            if from_drift_only:
                name = type_gen + '_drift'
            else:
                name = type_gen

        res_dict = self._save_gen(pred_x, pred_stat, time_grid, mask_gen, name=name, pred_z=pred_z, pred_r=pred_r, z_stat=z_stat, int_lambda=int_lambda, log_lambda=log_lambda, var_names_save=var_names_save, var_names_static=var_names_static, save=save)
        return res_dict
    

    def _save_gen(self, pred_x, pred_s, time_grid, mask=None, name='Rec', pred_z=None, pred_r=None, z_stat=None, int_lambda=None, log_lambda=None, var_names_save=None, var_names_static=None, save=True):   
        """
        Saves the generated data.

        Parameters:
            - pred_x (torch.Tensor): Predicted longitudinal data. (n_traj, n_timepoints, n_dims_long)
            - pred_s (torch.Tensor): Predicted static data. (n_traj, n_dims_static)
            - time_grid (torch.Tensor): Time grid. (n_timepoints)
            - mask (torch.Tensor, optional): Mask of the generated data. (n_traj, n_timepoints, n_dims_long) Default is None.
            - name (str): Name of the saved file. Default is 'Rec'.
            - pred_z (torch.Tensor, optional): Latent longitudinal variables. (n_traj, n_timepoints, n_dims_latent_long) Default is None. 
            - pred_r (torch.Tensor, optional): Latent residual variables. (n_traj, n_timepoints, n_dims_latent_long) Default is None.
            - z_stat (torch.Tensor, optional): Latent static variables. (n_traj, n_dims_latent_static) Default is None.    
            - int_lambda (torch.Tensor, optional): Integrated event rate. (n_traj, n_timepoints, n_dims_long) Default is None.
            - log_lambda (torch.Tensor, optional): Log event rate. (n_traj, n_timepoints, n_dims_long) Default is None. 
        """
        if self.config.from_best:
            name = 'Best_' + name
        name = '%s_%d.pth'%(name, self.config.epoch_init)

        s_dict = {}
        s_dict['Long_Values'] = pred_x.cpu()
        s_dict['Time_Grid'] = time_grid.cpu()
        s_dict['Var_Names'] = var_names_save
        s_dict['Var_Names_Static'] = var_names_static

        if pred_s is not None:
            pred_s = pred_s.cpu()
        s_dict['Stat_Values'] = pred_s
        if mask is not None:
            s_dict['Mask'] = mask.cpu()
        if pred_z is not None:
            s_dict['Latent_longitudinal'] = pred_z.cpu()
        if pred_r is not None:
            s_dict['Latent_residual'] = pred_r.cpu()
        if z_stat is not None:      
            s_dict['Latent_static'] = z_stat.cpu()
        if int_lambda is not None:
            s_dict['Int_lambda'] = int_lambda.cpu()
        if log_lambda is not None:
            s_dict['Log_lambda'] = log_lambda.cpu()
        
        if save:    
            s_path = os.path.join(self.config.save_path_samples, name)
            torch.save(s_dict, s_path)
        return s_dict
    
