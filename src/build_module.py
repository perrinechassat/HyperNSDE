import warnings
import numpy as np
import torch
import torch.nn as nn
from src.modules.long_decoder import *
from src.modules.long_encoder import *
from src.modules.long_latent_model import *
from src.modules.static_HIVAE import *
from src.modules.loss_aggregation import *
from src.losses import create_discriminator
from src.utils import init_network_weights
warnings.filterwarnings('ignore')

# Params config: [GPU, latent_dim, n_long_var, nhidden, static_data, s_vals_dim, s_onehot_dim, s_dim_static, z_dim_static, 
#                   batch_norm_static, sde, sigma_kernel, kernel_type, n_scalings_kernel, max_batch_kernel]

class Module(nn.Module):    
    def __init__(self, config, static_types, init_x=None, init_mask=None):

        super(Module, self).__init__()

        self.config = config
        self.device = torch.device('cuda') if config.GPU else torch.device('cpu')
        # Build model submodules
        self.build_model(static_types, init_x, init_mask)
        # Move model to device
        self.to(self.device)


    def build_model(self, static_types, init_x=None, init_mask=None):
        """Build all submodules of the Model."""
        
        # Latent dimensions
        self.latent_dim_long = self.config.latent_dim 
        if self.config.static_data:
            self.latent_dim_static = self.config.z_dim_static + self.config.s_dim_static # modification here to include the s part of the static data
        else:
            self.latent_dim_static = 0

        self.weights_loss = {}
        # Longitudinal Latent Model
        if self.config.sde and not self.config.sde_split_training:
            self.tasks = []
        else:
            self.tasks = ['Longi ODE']
            self.weights_loss['Longi ODE'] = self.config.loss_scaling_ode
        self.L_Latent = Longitudinal_Latent_Model(config=self.config, 
                                                    lat_size_long=self.latent_dim_long,
                                                    lat_size_stat=self.latent_dim_static, 
                                                    nhidden_number=self.config.nhidden).to(self.device)
        if self.config.sde:
            self.tasks.append('Longi SDE')
            self.weights_loss['Longi SDE'] = self.config.loss_scaling_sde

        # Static Encoder & Decoder
        if self.config.static_data:
            self.S_Enc = HIVAE_Encoder(self.config.s_vals_dim, 
                                      self.config.s_onehot_dim, 
                                      self.config.s_dim_static, 
                                      self.config.z_dim_static).to(self.device)
            self.S_Dec = HIVAE_Decoder(self.config.s_vals_dim, 
                                     static_types, 
                                     self.config.s_dim_static, 
                                     self.config.z_dim_static).to(self.device)

            self.static_types = static_types
            self.tasks.append('Static')
            self.weights_loss['Static'] = self.config.loss_scaling_static

        if self.config.estim_event_rate:
            self.tasks.append('Poisson')
            self.weights_loss['Poisson'] = self.config.loss_scaling_poisson

        # Longitudinal Encoder & Decoder
        # Encoder: 'RNN', 'LSTM', 'none' 
        if self.config.type_enc == 'LSTM':
            A_init = self.initialize_imputation(X=init_x, W=init_mask)
            self.Imp_Layer = VaderLayer(A_init).to(self.device)
            self.L_Enc = LSTMEncoder(i_size=self.config.n_long_var,
                                     h_size=self.config.nhidden_enc,  
                                     target_size=2 * self.latent_dim_long).to(self.device)
            init_network_weights(self.L_Enc)
        elif self.config.type_enc == 'RNN':
            A_init = self.initialize_imputation(X=init_x, W=init_mask)
            self.Imp_Layer = VaderLayer(A_init).to(self.device)
            self.L_Enc = RecognitionRNN(latent_dim=2 * self.latent_dim_long, 
                                        obs_dim=self.config.n_long_var, 
                                        nhidden=self.config.nhidden_enc, 
                                        act=self.config.act_init).to(self.device)
            init_network_weights(self.L_Enc)
       
        if self.config.type_dec == 'multiNODEs':
            self.L_Dec = Decoder_MultiNODEs(config=self.config, 
                                            latent_dim=self.L_Latent.out_size, 
                                            nhidden_number=self.config.nhidden_dec).to(self.device)
        elif self.config.type_dec == 'LSTM':
            self.L_Dec = LSTMDecoder(i_size=self.L_Latent.out_size, 
                                     h_size=self.config.nhidden_dec, 
                                     target_size=self.config.n_long_var).to(self.device)
        else:
            self.L_Dec = Decoder(config=self.config, 
                                 latent_dim=self.L_Latent.out_size).to(self.device)

        # Discriminators 
        if self.config.sde == True:
            if self.config.sde_training == 'MMD_FDM':
                discriminator_type = 'FDDiscriminator'
                discriminator_config = {"sigma" : self.config.sigma_kernel}             # Sigma in RBF kernel
            elif self.config.sde_training == 'MMD_SigKer':
                discriminator_type = 'SigKerMMDDiscriminator'
                discriminator_config = {"dyadic_order" : 1,                             # Mesh size of PDE solver used in loss function
                                        "kernel_type" : self.config.kernel_type,        # Type of kernel to use in the discriminator
                                        "sigma" : self.config.sigma_kernel,             # Sigma in RBF kernel
                                        "use_phi_kernel" : False,                       # Whether to use the the phi(k) = (k/2)! scaling. Set "kernel_type" to "linear".
                                        "n_scalings" : self.config.n_scalings_kernel,   # Number of samples to draw from Exp(1). ~8 tends to be a good choice.
                                        "max_batch" : self.config.max_batch_kernel}     # Maximum batch size to pass through the discriminator. 
            elif self.config.sde_training == 'MMD_PySig':
                discriminator_type = 'PySigMMDDiscriminator'
                discriminator_config = {"dyadic_order" : 1,                             # dyadic partition refinement
                                        "lead_lag" : self.config.sde_lead_lag,                             # whether to apply lead-lag transform
                                        "max_batch" : self.config.max_batch_kernel, 
                                        "sigma": self.config.sigma_kernel, 
                                        "kernel_type" : self.config.kernel_type}     # max batch size for kernel computation
            else:
                raise ValueError(f"Unknown neural sde training method: {self.config.sde_training}")
            self.discriminator = create_discriminator(discriminator_type, self.config.n_long_var, discriminator_config).to(self.device)

        # Aggregation
        self.agg_func = LinearScalarization(tasks=self.tasks, weights=self.weights_loss, device=self.device)

        print('Models were built')

    
    def initialize_imputation(self, X, W):
        # MultiNODES additional functions 
        # Initialize of the A-Variable for VADER
        W_A = torch.sum(W, 0)
        A = torch.sum(X * W, 0)
        A[W_A>0] = A[W_A>0] / W_A[W_A>0]
        # A[W_A>0] = A[W_A>0] / W_A[W_A>0]
        # if not available, then average across entire variable
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if W_A[i, j] == 0:
                    A[i, j] = torch.sum(X[:, :, j]) / torch.sum(W[:, :, j])
                    W_A[i, j] = 1
        # if not available, then average across all variables
        A[W_A==0] = torch.mean(X[W==1])
        return A


    def forward(self, *args, **kwargs):
        # Just a dummy forward; we won't actually use this in training
        # Opacus only needs the module to hook into parameters
        return None
        

    # def load_models(self):

    #     if self.config.from_best:
    #         ckpt_path_best = os.path.join(self.config.save_path_models, 'Best.pth')
    #         if os.path.exists(ckpt_path_best):
    #             print(f"Loading best checkpoint from {ckpt_path_best}")
    #             if self.config.GPU:
    #                 weights = torch.load(ckpt_path_best)
    #             else:
    #                 weights = torch.load(ckpt_path_best, map_location=torch.device('cpu'))
    #         else:
    #             raise FileNotFoundError(f"Best checkpoint not found at {ckpt_path_best}")
    #     else: 
    #         if self.config.epoch_init == 1:
    #             ckpt_path = os.path.join(self.config.save_path_models, "Ckpt_latest.pth")
    #             if os.path.exists(ckpt_path):
    #                 print(f"Loading checkpoint from {ckpt_path}")
    #                 if self.config.GPU:
    #                     weights = torch.load(ckpt_path)
    #                 else:
    #                     weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    #             else:
    #                 print("No checkpoint found. Starting from scratch.")
    #                 weights = None
    #                 self.best_loss = float('inf')
    #         else: 
    #             epoch = self.config.epoch_init
    #             ckpt_path = os.path.join(self.config.save_path_models, 'Ckpt_%d.pth'%(epoch))
    #             if not os.path.exists(ckpt_path):
    #                 print(f"Checkpoint for epoch {epoch} not found. Starting from latest.")
    #                 ckpt_path = os.path.join(self.config.save_path_models, "Ckpt_latest.pth")
    #             if self.config.GPU:
    #                 weights = torch.load(ckpt_path)
    #             else:
    #                 weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
        
    #     if weights is not None:        

    #         self.config.epoch_init = weights['Epoch']
    #         self.best_loss = weights['Val Loss']

    #         # Restore running average meters state
    #         if 'Train Loss' in weights:
    #             self.train_loss.set_from_values(weights['Train Loss'])
    #         if 'Train MSE' in weights:
    #             self.train_mse.set_from_values(weights['Train MSE'])
    #         if 'Val Loss' in weights:
    #             self.val_loss.set_from_values(weights['Val Loss'])
    #         if 'Val MSE' in weights:
    #             self.val_mse.set_from_values(weights['Val MSE'])
    #         if 'Val Losses Dict' in weights:
    #             for task in self.tasks:
    #                 if task in weights['Val Losses Dict']:
    #                     self.val_losses_dict[task].set_from_values(weights['Val Losses Dict'][task])
            
    #         # Load model weights
    #         self.L_Latent.load_state_dict(weights['L_Latent'])
    #         self.L_Dec.load_state_dict(weights['L_Dec'])
    #         self.optimizer.load_state_dict(weights['Opt'])
    #         if self.config.type_enc != 'none':
    #             self.L_Enc.load_state_dict(weights['L_Enc'])
    #             self.Imp_Layer.load_state_dict(weights['IL'])
    #         if self.config.static_data:
    #             self.S_Enc.load_state_dict(weights['S_Enc'])
    #             self.S_Dec.load_state_dict(weights['S_Dec'])
                
    #         print('Models have loaded from epoch:', self.config.epoch_init)


    # def save(self, epoch, train_loss, val_loss, best=False):

    #     weights = {}
    #     weights['L_Latent'] = self.L_Latent.state_dict()
    #     weights['L_Dec'] = self.L_Dec.state_dict()
    #     weights['Opt'] = self.optimizer.state_dict()

    #     if self.config.type_enc != 'none':
    #         weights['L_Enc'] = self.L_Enc.state_dict()
    #         weights['IL'] = self.Imp_Layer.state_dict()
    #     if self.config.static_data:
    #         weights['S_Enc'] = self.S_Enc.state_dict()
    #         weights['S_Dec'] = self.S_Dec.state_dict()

    #     weights['Train Loss'] = train_loss
    #     weights['Val Loss'] = val_loss
    #     weights['Epoch'] = epoch
    #     weights['Train MSE'] = self.train_mse.avg
    #     weights['Val MSE'] = self.val_mse.avg
    #     weights['Val Losses Dict'] = {}
    #     for task in self.tasks:
    #         weights['Val Losses Dict'][task] = self.val_losses_dict[task].avg
        
    #     if best:
    #         torch.save(weights, 
    #             os.path.join(self.config.save_path_models, 'Best.pth'))
    #     else:
    #         torch.save(weights, 
    #             os.path.join(self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))
    #         torch.save(weights, 
    #             os.path.join(self.config.save_path_models, 'Ckpt_latest.pth'))

    #     print('Models have been saved')



    