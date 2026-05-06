import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import init_network_weights, split_last_dim, check_mask, linspace_vector, reverse


# params in config: [num_ode_layers, act_ode, num_hypernet_layers, degree, static_data, 
#                       sde_type, noise_type, GPU, diff_shape, latent_model, use_context, 
#                       diff_mpl_size, diff_mlp_num_layers, act_diff, init_noise_size, init_mlp_size, 
#                       init_mlp_num_layers, act_init]


class LipSwish(torch.nn.Module):
    """
    LipSwish activation to control Lipschitz constant of MLP output
    """
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class ScaledTanh(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        return self.scale * torch.nn.Tanh()(x)



""" ______________________________________________________________________  

        MLP -- Standard multi-layer perceptron 
    ______________________________________________________________________"""

class MLP(nn.Module): # pour la diffusion
    def __init__(self, in_size, out_size, mlp_size, num_layers, activation="LipSwish", tanh=False, tscale=1, stat_augmented=False, softplus=False):
        """
        Initialisation of perceptron

        :param in_size:     Size of data input
        :param out_size:    Output data size
        :param mlp_size:    Number of neurons in each hidden layer
        :param num_layers:  Number of hidden layers
        :param activation:  Activation function to use between layers.
        :param tanh:        Whether to apply tanh activation to final linear layer
        :param tscale:      Custom scaler to tanh layer
        """
        super(MLP, self).__init__()
        self.in_size = in_size
        self.stat_augmented = stat_augmented

        if activation != "LipSwish":
            self.activation = getattr(torch.nn, activation)
        else:
            self.activation = LipSwish

        model = [torch.nn.Linear(in_size, mlp_size), self.activation()]
        for _ in range(num_layers):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(self.activation())
        model.append(torch.nn.Linear(mlp_size, out_size))

        if tanh:
            model.append(ScaledTanh(tscale))
        if softplus:
            model.append(torch.nn.Softplus()) 

        self._model = torch.nn.Sequential(*model)

    def forward(self, z):
        if self.stat_augmented:
            z = self.concat_zstat(z)
        out = self._model(z)
        return out
    
    def augment(self, zstat=None):
        self.zstat = zstat
    
    def concat_zstat(self, z):
        if self.zstat is not None:
            return torch.cat([z, self.zstat],dim=1) 
        else:
            return z
        

class MLP_standard(nn.Module): # pour la diffusion
    def __init__(self, in_size, out_size, mlp_size, num_layers, activation="LipSwish", tanh=False, tscale=1, softplus=False):
        """
        Initialisation of perceptron

        :param in_size:     Size of data input
        :param out_size:    Output data size
        :param mlp_size:    Number of neurons in each hidden layer
        :param num_layers:  Number of hidden layers
        :param activation:  Activation function to use between layers.
        :param tanh:        Whether to apply tanh activation to final linear layer
        :param tscale:      Custom scaler to tanh layer
        """
        super(MLP_standard, self).__init__()
        self.in_size = in_size

        if activation != "LipSwish":
            self.activation = getattr(torch.nn, activation)
        else:
            self.activation = LipSwish

        model = [torch.nn.Linear(in_size, mlp_size), self.activation()]
        for _ in range(num_layers):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(self.activation())
        model.append(torch.nn.Linear(mlp_size, out_size))

        if tanh:
            model.append(ScaledTanh(tscale))
        if softplus:
            model.append(torch.nn.Softplus()) 

        self._model = torch.nn.Sequential(*model)

    def forward(self, z):
        out = self._model(z)
        return out
    

class MLP_wtime(nn.Module): # pour la diffusion
    def __init__(self, in_size, out_size, mlp_size, num_layers, activation="LipSwish", tanh=False, tscale=1, softplus=False):
        """
        Initialisation of perceptron

        :param in_size:     Size of data input
        :param out_size:    Output data size
        :param mlp_size:    Number of neurons in each hidden layer
        :param num_layers:  Number of hidden layers
        :param activation:  Activation function to use between layers.
        :param tanh:        Whether to apply tanh activation to final linear layer
        :param tscale:      Custom scaler to tanh layer
        """
        super(MLP_wtime, self).__init__()
        self.in_size = in_size

        if activation != "LipSwish":
            self.activation = getattr(torch.nn, activation)
        else:
            self.activation = LipSwish

        model = [torch.nn.Linear(in_size, mlp_size), self.activation()]
        for _ in range(num_layers):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(self.activation())
        model.append(torch.nn.Linear(mlp_size, out_size))

        if tanh:
            model.append(ScaledTanh(tscale))
        if softplus:
            model.append(torch.nn.Softplus()) 

        self._model = torch.nn.Sequential(*model)

    def forward(self, t, z):
        out = self._model(z)
        return out



""" ______________________________________________________________________  

        Bottleneck Structure MLP 
    ______________________________________________________________________"""

def bottleneck_structure_MLP(num_layers=3, in_size=4, out_size=4, nhidden_number=20, activation='Tanh'):
       
    if activation != "LipSwish":
        act = getattr(torch.nn, activation)()
    else:
        act = LipSwish()
    
    # Difference for layer sizes from input to hidden and hidden to output
    diff_in_hidden = nhidden_number - in_size
    diff_hidden_out = nhidden_number - out_size
    layers_enc = []
    layers_dec = []

    for i in range(num_layers):
        if num_layers == 1:
            layers_enc.append(nn.Linear(in_size, nhidden_number))
            layers_dec.append(nn.Linear(nhidden_number, out_size))
        else:
            fact_i = i / num_layers
            fact_o = (i + 1) / num_layers
            # Encoder structure: in_size -> nhidden_number
            c_i_in = in_size + int(np.round(diff_in_hidden * fact_i))
            c_o_in = in_size + int(np.round(diff_in_hidden * fact_o))
            # Decoder structure: nhidden_number -> out_size
            c_i_out = nhidden_number - int(np.round(diff_hidden_out * fact_i))
            c_o_out = nhidden_number - int(np.round(diff_hidden_out * fact_o))

            layers_enc.append(nn.Linear(c_i_in, c_o_in))
            layers_dec.append(nn.Linear(c_i_out, c_o_out))

        layers_enc.append(act)
        layers_dec.append(act)

    layers_enc.append(nn.Linear(nhidden_number, nhidden_number))
    layers_enc.append(act)
    layers_dec = layers_dec[:-1]
    layers = layers_enc + layers_dec
    
    return nn.Sequential(*layers)



""" ______________________________________________________________________  

    MultiNDEs -- Multimodal Neural Differential Equation (from MultiNODEs) 
    ______________________________________________________________________"""

class MultiNDEs(nn.Module): 
    def __init__(self, config, in_size=4, out_size=4, nhidden_number=20):
        super(MultiNDEs, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        # self.estim_event_rate = config.estim_event_rate
        self.sde = config.sde
        self.lambda_dim = config.lambda_dim
        self.model = bottleneck_structure_MLP(config.num_ode_layers, in_size, out_size, nhidden_number, config.act_ode)
        self.nfe = 0
        # if self.estim_event_rate:
        #     # self.log_lambda_net = MLP(in_size=self.out_size, out_size=1, mlp_size=config.lambda_mlp_size, num_layers=config.lambda_num_layers, activation=config.lambda_act, tanh=False)
        #     self.log_lambda_net = MLP(in_size=self.out_size, out_size=config.lambda_dim, mlp_size=config.lambda_mlp_size, num_layers=config.lambda_num_layers, activation=config.lambda_act, tanh=False)

    def forward(self, t, z):
        self.nfe += 1
        # if self.estim_event_rate:
        #     # if sde is True, z = [t, z_, Lambda]
        #     zt = z[:,:-self.lambda_dim]
        #     dz_dt = self.model(zt)
        #     # if self.sde:
        #     #     z_ = zt[:,1:]
        #     # else: 
        #     z_ = zt
        #     dLambda_dt = torch.exp(self.log_lambda_net(z_))
        #     out = torch.cat([dz_dt, dLambda_dt], dim=1)
        # else: 
        out = self.model(z) 
        return out

    # def extract_event_rate(self, z_Lambda):
    #     if self.estim_event_rate:
    #         int_lambda = z_Lambda[:,:,self.out_size:]        
    #         z = z_Lambda[:,:,:self.out_size] 
    #         log_lambda = self.log_lambda_net(z)
    #         return z, int_lambda, log_lambda
    #     else: 
    #         raise RuntimeError("`extract_event_rate` should not be called when `estim_event_rate` is False in config.")

""" ______________________________________________________________________  

        StatMoNDEs -- Static-Modulated Neural Differential Equation  
    ______________________________________________________________________"""

class StatMoNDEs(nn.Module):
    def __init__(self, config, in_size=4, out_size=4, nhidden_number=20):
        super(StatMoNDEs, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        # self.estim_event_rate = config.estim_event_rate
        self.sde = config.sde
        self.lambda_dim = config.lambda_dim
        self.model = bottleneck_structure_MLP(config.num_ode_layers, in_size, out_size, nhidden_number, config.act_ode)
        self.nfe = 0
        # if self.estim_event_rate:
        #     # self.log_lambda_net = MLP(in_size=self.out_size, out_size=1, mlp_size=config.lambda_mlp_size, num_layers=config.lambda_num_layers, activation=config.lambda_act, tanh=False)
        #     self.log_lambda_net = MLP(in_size=self.out_size, out_size=config.lambda_dim, mlp_size=config.lambda_mlp_size, num_layers=config.lambda_num_layers, activation=config.lambda_act, tanh=False)

    def forward(self, t, z):
        self.nfe += 1
        # if self.estim_event_rate:
        #     # if sde is True, z = [t, z_, Lambda]
        #     zt = z[:,:-self.lambda_dim]
        #     z_zstat = self.concat_zstat(zt)
        #     dz_dt = self.model(z_zstat)
        #     # if self.sde:
        #     #     z_ = zt[:,1:]
        #     # else: 
        #     z_ = zt
        #     dLambda_dt = torch.exp(self.log_lambda_net(z_)) # ici on pourrait faire dépendre des statiques en mettant z_zstat et en changeant la taille de log_lambda_net
        #     out = torch.cat([dz_dt, dLambda_dt], dim=1)
        # else: 
        z = self.concat_zstat(z)
        out = self.model(z)
        return out
    
    def augment(self, zstat=None):
        self.zstat = zstat
    
    def concat_zstat(self, z):
        if self.zstat is not None:
            return torch.cat([z, self.zstat], dim=1) 
        else:
            return z
        
    # def extract_event_rate(self, z_Lambda):
    #     if self.estim_event_rate:
    #         int_lambda = z_Lambda[:,:,self.out_size:]        
    #         z = z_Lambda[:,:,:self.out_size] 
    #         log_lambda = self.log_lambda_net(z)
    #         return z, int_lambda, log_lambda
    #     else: 
    #         raise RuntimeError("`extract_event_rate` should not be called when `estim_event_rate` is False in config.")
        


""" ______________________________________________________________________  

        HyperNDEs -- Hypernetworks-based Neural Differential Equation  
    ______________________________________________________________________"""

class HyperNDEs(nn.Module):

    def __init__(self, config, in_size, out_size, nhidden_number):
        super(HyperNDEs, self).__init__()

        self.config = config

        if config.act_ode != "LipSwish":
            self.act = getattr(torch.nn, config.act_ode)()
        else:
            self.act = LipSwish()

        self.sde = config.sde
        self.in_size = in_size
        self.out_size = out_size
        # self.estim_event_rate = config.estim_event_rate
        self.ode_model = bottleneck_structure_MLP(config.num_ode_layers, in_size, out_size, nhidden_number, config.act_ode)
        self.nfe = 0
        self.lambda_dim = config.lambda_dim
        
        # if self.estim_event_rate:
        #     # self.log_lambda_net = MLP(in_size=self.out_size, out_size=1, mlp_size=config.lambda_mlp_size, num_layers=config.lambda_num_layers, activation=config.lambda_act, tanh=False)
        #     self.log_lambda_net = MLP(in_size=self.out_size, out_size=config.lambda_dim, mlp_size=config.lambda_mlp_size, num_layers=config.lambda_num_layers, activation=config.lambda_act, tanh=False)

        tot_num_params, params_shape = self._get_model_shape(self.ode_model)
        self.tot_num_params = tot_num_params
        self.params_shape = params_shape
        self.nb_layer = len(self.params_shape)

        # disable gradient computation for HyperNDEs's parameters (only the ones of Hypernetwork)
        for param in self.ode_model.parameters():
            param.requires_grad = False

    def _get_model_shape(self, model):
        tot_num_params = sum(p.numel() for p in model.parameters())
        params_shape = []
        for layer in self.ode_model:
            if isinstance(layer, nn.Linear):
                params_shape.append(layer.weight.shape)
        return tot_num_params, params_shape 

    def set_params_model(self, weights, biases):
        self.weights = weights  # List of weight tensors for each layer, each of shape (batch_size, in_dim, out_dim).
        self.biases = biases    # List of bias tensors for each layer, each of shape (batch_size, out_dim).

    def compute_ode_model(self, z):
        out = z
        for i in range(len(self.weights)):
            # print('shape weights i', self.weights[i].shape)
            # print('shape biases i', self.biases[i].shape)
            out = torch.bmm(out.unsqueeze(1), self.weights[i]).squeeze(1) + self.biases[i] # (batch_size, out_dim)
            if i < self.nb_layer-1:
                out = self.act(out)
        return out

    def forward(self, t, z):
        self.nfe += 1
        # if self.estim_event_rate:
        #     # if sde is True, z = [t, z_, Lambda]
        #     zt = z[:,:-self.lambda_dim] 
        #     dz_dt = self.compute_ode_model(zt)
        #     # z_ = zt
        #     # log_lambda = self.log_lambda_net(z_)
        #     log_lambda = self.log_lambda_net(zt)
        #     # log_lambda = torch.clamp(log_lambda, min=-10, max=10)
        #     dLambda_dt = torch.exp(log_lambda)
        #     out = torch.cat([dz_dt, dLambda_dt], dim=1)
        # else: 
        out = self.compute_ode_model(z)
        return out 

    # def extract_event_rate(self, z_Lambda):
    #     if self.estim_event_rate:
    #         int_lambda = z_Lambda[:,:,self.out_size:]        
    #         z = z_Lambda[:,:,:self.out_size] 
    #         log_lambda = self.log_lambda_net(z)
    #         return z, int_lambda, log_lambda
    #     else: 
    #         raise RuntimeError("`extract_event_rate` should not be called when `estim_event_rate` is False in config.")

    
class Hypernetwork(nn.Module):
    def __init__(self, config, latent_dim_stat, num_params, main_model_shapes, max_hidden_dim=None):
        super(Hypernetwork, self).__init__()

        self.main_model_shapes = main_model_shapes
        self.num_params = num_params
        if config.act_hypernetwork != "LipSwish":
            self.act = getattr(torch.nn, config.act_hypernetwork)()
        else:
            self.act = LipSwish()

        hidden_dims = [latent_dim_stat, 64]
        next_hidden_dim = int(hidden_dims[-1]*2)
        while next_hidden_dim < self.num_params or len(hidden_dims) < config.num_hypernet_layers:
            hidden_dims.append(next_hidden_dim)
            next_hidden_dim = int(hidden_dims[-1]*2)
        hidden_dims.append(self.num_params) 

        # Build the model
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i < len(hidden_dims) - 2:  # No activation after the final output layer
                layers.append(self.act)
        self.hypermodel = nn.Sequential(*layers)

    def forward(self, z_stat):
        """
        Generate parameters for the main model.
        Args:
            z_stat: Input latent static variables of shape (batch_size, l_dim_stat).
        Returns:
            A list of (weight, bias) tensors for the ode MLP layers.
        """
        params_flat = self.hypermodel(z_stat)  # Shape: (batch_size, num_params)
        weights, biases = self._reshape_params(params_flat)
        return weights, biases
    
    def _reshape_params(self, params_flat):
        batch_size = params_flat.shape[0]
        weights = []  # List of weight tensors for each layer, each of shape (batch_size, in_dim, out_dim).
        biases = []   # List of bias tensors for each layer, each of shape (batch_size, out_dim).
        start = 0
        for out_dim, in_dim in self.main_model_shapes:
            weight_size = in_dim * out_dim
            bias_size = out_dim
            weight = params_flat[:, start:start + weight_size].view(batch_size, in_dim, out_dim)
            start += weight_size
            weights.append(weight)
            bias = params_flat[:, start:start + bias_size].view(batch_size, out_dim)
            start += bias_size
            biases.append(bias)
        return weights, biases


""" ______________________________________________________________________  

        Model for the event rate 
    ______________________________________________________________________"""

 
class EventRate(nn.Module):
    def __init__(self, config, out_size):
        super(EventRate, self).__init__()
        self.out_size = out_size
        self.config = config
        self.device = torch.device('cuda') if config.GPU else torch.device('cpu')
        self.log_lambda_net = MLP_standard(in_size=self.out_size, out_size=config.lambda_dim + 1, mlp_size=config.lambda_mlp_size, num_layers=config.lambda_num_layers, activation=config.lambda_act, tanh=False).to(self.device)
    
    def _integrate_trapezoidal(self, lambda_rates, T, mask):
        """
        Vectorized trapezoidal integration restricted to each sample's observed window.
        
        Parameters
        ----------
        lambda_rates : (B, N, D)
            Predicted λ(t) values on full grid T.
        T : (N,)
            Time grid (monotonic increasing).
        mask : (B, N, D)
            Binary mask indicating observed points per variable.

        Returns
        -------
        int_lambda_obs : (B, D)
            Integrated λ(t) only over [t0_obs, tK_obs] for each sample n and variable d.
        """
        B, N, D = lambda_rates.shape
        device, dtype = lambda_rates.device, lambda_rates.dtype

        # Time differences
        dt = T[1:] - T[:-1]  # (N-1,)
        dt = dt.view(1, N-1, 1).to(device)

        # Trapezoidal increments
        avg_rates = 0.5 * (lambda_rates[:, :-1, :] + lambda_rates[:, 1:, :])  # (B, N-1, D)
        integral = avg_rates * dt  # (B, N-1, D)
        cum_integral = torch.cumsum(integral, dim=1)  # (B, N-1, D)

        # Pad with 0 at start to align with T
        zero_pad = torch.zeros(B, 1, D, device=device, dtype=dtype)
        cum_integral = torch.cat([zero_pad, cum_integral], dim=1)  # (B, N, D)

        assert(torch.sum(cum_integral[:,0,:]) == 0.)
        assert(torch.sum(cum_integral[0,-1,:] <= 0) == 0.)

        # ---- Identify observed windows per sample (min and max indices) ----
        # Collapse mask across D dims to detect any observed variable
        mask_any = mask.any(dim=-1)  # (B, N)
        idxs = torch.arange(N, device=device).view(1, N).expand(B, N)

        # first observed index
        i0 = torch.where(mask_any, idxs, N)  # replace False with large number
        i0 = torch.min(i0, dim=1).values  # (B,)

        # last observed index
        iK = torch.where(mask_any, idxs, -1)
        iK = torch.max(iK, dim=1).values  # (B,)

        # gather integration limits
        i0_exp = i0.view(B, 1, 1).expand(-1, 1, D)
        iK_exp = iK.view(B, 1, 1).expand(-1, 1, D)

        # extract integrated values at tK and t0
        int_lambda_tK = torch.gather(cum_integral, 1, iK_exp).squeeze(1)  # (B, D)
        int_lambda_t0 = torch.gather(cum_integral, 1, i0_exp).squeeze(1)  # (B, D)

        # integral over [t0_obs, tK_obs]
        int_lambda_obs = int_lambda_tK - int_lambda_t0  # (B, D)

        return int_lambda_obs

    # def _integrate_trapezoidal(self, lambda_rates, T):
    #     """
    #     Fast vectorized trapezoidal integration over irregular grid.
    #     lambda_rates: (batch_size, time_points, lambda_dim)
    #     T: (time_points,)
    #     Returns: integrated lambda (batch_size, time_points, lambda_dim)
    #     """
    #     batch_size, time_points, lambda_dim = lambda_rates.shape

    #     # Time differences (T[1] - T[0], ..., T[N] - T[N-1]) => shape (time_points-1,)
    #     dt = T[1:] - T[:-1]  # (N-1,)
        
    #     # Trapezoidal integration step-by-step
    #     avg_rates = 0.5 * (lambda_rates[:, :-1, :] + lambda_rates[:, 1:, :])  # (B, N-1, D)
    #     dt = dt.unsqueeze(0).unsqueeze(-1)  # (1, N-1, 1)
    #     integral = avg_rates * dt  # (B, N-1, D)

    #     # Cumulative sum along the time axis
    #     cum_integral = torch.cumsum(integral, dim=1)  # (B, N-1, D)

    #     # Pad with zeros at the start to match full time length
    #     zero_pad = torch.zeros(batch_size, 1, lambda_dim, device=integral.device, dtype=integral.dtype)
    #     int_lambda = torch.cat([zero_pad, cum_integral], dim=1)  # (B, N, D)

    #     return int_lambda[:,-1,:]
    
    def _integrate_simpson(self, lambda_rates, T):
        """
        Integrate using Simpson's rule for better accuracy on irregular grid.
        """
        batch_size, time_points, lambda_dim = lambda_rates.shape
        
        # Initialize integrated lambda
        int_lambda = torch.zeros_like(lambda_rates)
        
        # Simpson's rule requires even number of intervals
        if time_points % 2 == 0:
            # Even number of points
            for i in range(2, time_points, 2):
                dt = T[i] - T[i-2]
                # Simpson's rule: (f0 + 4*f1 + f2) * dt / 6
                simpson_rate = (lambda_rates[:, i-2, :] + 4*lambda_rates[:, i-1, :] + lambda_rates[:, i, :]) / 6.0
                int_lambda[:, i, :] = int_lambda[:, i-2, :] + simpson_rate * dt.unsqueeze(0).unsqueeze(-1)
        else:
            # Odd number of points, use trapezoidal for last interval
            for i in range(2, time_points-1, 2):
                dt = T[i] - T[i-2]
                simpson_rate = (lambda_rates[:, i-2, :] + 4*lambda_rates[:, i-1, :] + lambda_rates[:, i, :]) / 6.0
                int_lambda[:, i, :] = int_lambda[:, i-2, :] + simpson_rate * dt.unsqueeze(0).unsqueeze(-1)
            
            # Last interval with trapezoidal rule
            dt_last = T[-1] - T[-2]
            avg_rate = (lambda_rates[:, -2, :] + lambda_rates[:, -1, :]) / 2.0
            int_lambda[:, -1, :] = int_lambda[:, -2, :] + avg_rate * dt_last.unsqueeze(0).unsqueeze(-1)
        
        assert(torch.sum(int_lambda[:,0,:]) == 0.)
        assert(torch.sum(int_lambda[0,-1,:] <= 0) == 0.)
        return int_lambda[:,-1,:]
    
    
    def _integrate_ode(self, lambda_rates, T):
        """
        Integrate using ODE solver for maximum accuracy.
        """
        batch_size, time_points, lambda_dim = lambda_rates.shape
        
        # Create interpolation function for lambda rates
        def lambda_interp(t):
            # Find closest time indices
            t_idx = torch.searchsorted(T, t, right=False)
            t_idx = torch.clamp(t_idx, 0, time_points - 1)
            return lambda_rates[torch.arange(batch_size), t_idx]
        
        # ODE function for integration
        def integrate_ode(t, int_lambda_state):
            return lambda_interp(t)
        
        # Initial condition: zero integrated lambda
        int_lambda_init = torch.zeros(batch_size, lambda_dim, device=lambda_rates.device)
        
        # Integrate using ODE solver
        from torchdiffeq import odeint
        int_lambda_trajectory = odeint(integrate_ode, int_lambda_init, T, 
                                     rtol=self.config.rtol, atol=self.config.atol, 
                                     method=self.config.method_solver)
        
        # Return in correct shape
        int_lambda = int_lambda_trajectory.permute(1, 0, 2)  # (batch_size, time_points, lambda_dim)
        assert(torch.sum(int_lambda[:,0,:]) == 0.)
        assert(torch.sum(int_lambda[0,-1,:] <= 0) == 0.)
        return int_lambda[:,-1,:] 
    
    def compute_event_rate_complete(self, z_trajectory, T, mask, method='trapezoidal'):
        """
        Complete event rate computation with integration on irregular grid.
        
        Args:
            z_trajectory: (batch_size, time_points, out_size) - latent trajectory
            T: (time_points,) - irregular time grid
            method: 'trapezoidal', 'simpson', or 'ode'
            
        Returns:
            z_trajectory, int_lambda, log_lambda
        """
        # Compute log_lambda for all time points
        # log_lambda = self.log_lambda_net(z_trajectory)  # (batch_size, time_points, lambda_dim)

        # Version where log_lambda depends on time as well (time-augmented input)
        T_expanded = T.view(1, -1, 1).expand(z_trajectory.shape[0], -1, 1).to(z_trajectory.device)
        z_with_t = torch.cat([z_trajectory, T_expanded], dim=-1)  # Shape: (batch_size, time_points, latent_dim + 1)
        log_lambda = self.log_lambda_net(z_with_t)  # (batch_size, time_points, lambda_dim)
        
        # Integrate lambda rates based on method
        if method == 'trapezoidal':
            int_lambda = self._integrate_trapezoidal(torch.exp(log_lambda), T, mask)
        elif method == 'simpson':
            int_lambda = self._integrate_simpson(torch.exp(log_lambda), T)
        elif method == 'ode':
            int_lambda = self._integrate_ode(torch.exp(log_lambda), T)
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        return int_lambda, log_lambda
        


""" ______________________________________________________________________  

        Latent SDE Model on the residual 
    ______________________________________________________________________"""


class NSDE(nn.Module):
    def __init__(self, config, lat_size_long_in, in_size_ode_model, out_size_ode_model, drift_model=None):
        super(NSDE, self).__init__()

        self.sde_type = config.sde_type # for torchsde
        self.noise_type = config.diff_type # for torchsde

        self.latent_model = config.latent_model
        self.diff_type = config.diff_type
        self.device = torch.device('cuda') if config.GPU else torch.device('cpu')
        # self.estim_event_rate = config.estim_event_rate

        self.out_size_ode_model = out_size_ode_model
        self.in_size_ode_model = in_size_ode_model
        self.lat_size_long_in = lat_size_long_in

        if self.noise_type == "diagonal":
            self._diff_shape = 1
        else:
            self._diff_shape = config.diff_shape    # size of the Brownian Motion

        # add here the case where the diffusion only depends on the time variable (not z). 
        
        if drift_model is not None:
            self._drift = drift_model
        else: # only on residual
            self._drift = MLP_wtime(lat_size_long_in + 1, # time-augmented process in input
                        out_size_ode_model,
                        config.res_drift_mlp_size, 
                        config.res_drift_mlp_num_layers, 
                        config.act_res_drift, 
                        tanh=True).to(self.device)
        
        self._diffusion = MLP_wtime(lat_size_long_in + 1, 
                            out_size_ode_model * self._diff_shape, 
                            config.diff_mlp_size, 
                            config.diff_mlp_num_layers, 
                            config.act_diff, 
                            tanh=True, 
                            softplus=False).to(self.device)
        
        # self._initial = MLP(config.init_noise_size, 
        #                     in_size_ode_model - 1, 
        #                     config.init_mlp_size, 
        #                     config.init_mlp_num_layers, 
        #                     config.act_init, 
        #                     tanh=False).to(self.device)

        
    def f_and_g(self, t, z):
        # t has shape (), z has shape (batch_size, hidden_size)
        t_exp = t.expand(z.size(0), 1)
        tz = torch.cat([t_exp, z], dim=1)
        drift = self._drift(t, tz)
        # if self.estim_event_rate:
        #     vec_zeros = torch.zeros(tz[:,-1:].shape, device=self.device)
        #     tz_ = tz[:,:-1]
        #     if self.noise_type == "diagonal":
        #         diffusion = self._diffusion(tz_)
        #     else:
        #         diffusion = self._diffusion(tz_).view(z.size(0), self.out_size, self._diff_shape)
        #     diffusion = torch.cat([diffusion, vec_zeros], dim=1)
        # else:
        if self.noise_type == "diagonal":
            diffusion = self._diffusion(t, tz)
        else:
            diffusion = self._diffusion(t, tz).view(z.size(0), self.out_size_ode_model, self._diff_shape)    
        return drift, diffusion

    def f(self, t, z):
        t_exp = t.expand(z.size(0), 1)
        tz = torch.cat([t_exp, z], dim=1) # t concat to z or [z, Lambda]
        return self._drift(t, tz)

    def g(self, t, z):
        t = t.expand(z.size(0), 1)
        tz = torch.cat([t, z], dim=1)
        # if self.estim_event_rate:
        #     vec_zeros = torch.zeros(tz[:,-1:].shape, device=self.device)
        #     tz_ = tz[:,:-1] 
        #     if self.noise_type == "diagonal":
        #         diffusion = self._diffusion(tz_)
        #     else:
        #         diffusion = self._diffusion(tz_).view(z.size(0), self.out_size, self._diff_shape)
        #     diffusion = torch.cat([diffusion, vec_zeros], dim=1)
        # else:
        if self.noise_type == "diagonal":
            diffusion = self._diffusion(t, tz)
        else:
            diffusion = self._diffusion(t, tz).view(z.size(0), self.out_size_ode_model, self._diff_shape) 
        return diffusion


""" ______________________________________________________________________  

        Complete Longitudinal Latent Neural SDE or ODE Model 
    ______________________________________________________________________"""


class Longitudinal_Latent_Model(nn.Module):
    def __init__(self, config, lat_size_long, lat_size_stat, nhidden_number):
        super(Longitudinal_Latent_Model, self).__init__()

        self.latent_model = config.latent_model
        self.device = torch.device('cuda') if config.GPU else torch.device('cpu')
        self.estim_event_rate = config.estim_event_rate

        self.lat_size_stat = lat_size_stat
        self.lat_size_long_in = lat_size_long   
        self.lat_size_long_out = lat_size_long        

        # add here the case where the diffusion only depends on the time variable (not z). 

        # Model 
        if config.latent_model == 'MultiNDEs':
            if config.sde and not config.sde_split_training:
                self.in_size = self.lat_size_long_in + self.lat_size_stat + 1 # time-augmented process in input
            else:
                self.in_size = self.lat_size_long_in + self.lat_size_stat
            self.out_size = self.lat_size_long_out + self.lat_size_stat
            if config.fixed_init_cond == False and config.type_enc == 'none':
                self._initial = MLP_standard(config.init_noise_size + self.lat_size_stat, self.lat_size_long_out, config.init_mlp_size, 0, config.act_init, tanh=False).to(self.device) 
            self._drift = MultiNDEs(config, 
                                    in_size=self.in_size, 
                                    out_size=self.out_size, 
                                    nhidden_number=nhidden_number).to(self.device)
            if config.sde:
                if config.sde_split_training:
                    # Model for the diffusion (residual part)
                    self._residual = NSDE(config, self.lat_size_long_in + self.lat_size_stat, self.in_size, self.out_size).to(self.device)
                else:
                    self._latent_sde = NSDE(config, self.lat_size_long_in + self.lat_size_stat, self.in_size, self.out_size, drift_model=self._drift).to(self.device)
            if config.estim_event_rate:
                self._event_rate = EventRate(config, self.out_size + 1).to(self.device)

        elif config.latent_model == 'StatMoNDEs':
            if config.sde and not config.sde_split_training:
                self.in_size = self.lat_size_long_in + self.lat_size_stat + 1 # time-augmented process in input
            else:
                self.in_size = self.lat_size_long_in + self.lat_size_stat
            self.out_size = self.lat_size_long_out
            if config.fixed_init_cond == False and config.type_enc == 'none':
                self._initial = MLP_standard(config.init_noise_size + self.lat_size_stat, self.out_size, config.init_mlp_size, 0, config.act_init, tanh=False).to(self.device) 
            self._drift = StatMoNDEs(config, 
                                     in_size=self.in_size, 
                                     out_size=self.out_size, 
                                     nhidden_number=nhidden_number).to(self.device)
            if config.sde:
                if config.sde_split_training:
                    # Model for the diffusion (residual part)
                    self._residual = NSDE(config, self.lat_size_long_in, self.in_size, self.out_size).to(self.device)
                else:
                    self._latent_sde = NSDE(config, self.lat_size_long_in, self.in_size, self.out_size, drift_model=self._drift).to(self.device)
            if config.estim_event_rate:
                self._event_rate = EventRate(config, self.out_size + 1).to(self.device)

        elif config.latent_model == 'HyperNDEs':
            if config.sde and not config.sde_split_training:
                self.in_size = self.lat_size_long_in + 1 # time-augmented process in input
            else:
                self.in_size = self.lat_size_long_in 
            self.out_size = self.lat_size_long_out
            if config.fixed_init_cond == False and config.type_enc == 'none':
                self._initial = MLP_standard(config.init_noise_size + self.lat_size_stat, self.out_size, config.init_mlp_size, 0, config.act_init, tanh=False).to(self.device)                 
            self._drift = HyperNDEs(config, 
                                    in_size=self.in_size, 
                                    out_size=self.out_size, 
                                    nhidden_number=nhidden_number).to(self.device)
            self._hypernetwork_drift = Hypernetwork(config, 
                                                    latent_dim_stat=self.lat_size_stat, 
                                                    num_params=self._drift.tot_num_params, 
                                                    main_model_shapes=self._drift.params_shape).to(self.device)
            # disable gradient computation for HyperNDEs's parameters (only the ones of Hypernetwork)
            for param in self._drift.ode_model.parameters():
                param.requires_grad = False
            if config.sde:
                if config.sde_split_training:
                    # Model for the diffusion (residual part)
                    self._residual = NSDE(config, self.lat_size_long_in, self.in_size, self.out_size).to(self.device)
                else:
                    self._latent_sde = NSDE(config, self.lat_size_long_in, self.in_size, self.out_size, drift_model=self._drift).to(self.device)
            if config.estim_event_rate:
                self._event_rate = EventRate(config, self.out_size + 1).to(self.device)
        
        else:
            raise ValueError(f"Unknown latent_model_type: {config.latent_model}")
        
        self._initialize_weights(config)
        
    
    def _initialize_weights(self, config):
        # init_network_weights(self._initial)
        init_network_weights(self._drift)
        # if config.latent_model == 'HyperNDEs':
            # init_network_weights(self._hypernetwork_drift)
        if config.sde:
            if config.sde_split_training:
                init_network_weights(self._residual)
            else:
                init_network_weights(self._latent_sde)
        if config.estim_event_rate:
            init_network_weights(self._event_rate)
        
        
    def augment(self, zstat=None):
        if self.latent_model == 'StatMoNDEs':
            self._drift.augment(zstat)















