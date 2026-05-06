def base_parser(return_unknown=False):
        
    import argparse
    import sys 
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--time_normalization', 
                        type=bool,
                        default=True, 
                        help='Time normalization before computing sigker')
    parser.add_argument('--weight_decay_diff', 
                        type=float,
                        default=0.0,
                        help='')
    parser.add_argument('--weight_decay_drift', 
                        type=float,
                        default=0.0,
                        help='')
    parser.add_argument('--t0_2_static',
                        type=bool,
                        default=False, 
                        help='Whether to append t0 to the static variables')
    

    parser.add_argument('--sde_degree_sig', 
                        type=int,
                        default=3,
                        help='Degree of the signature for the sigker discriminator')
    parser.add_argument('--sde_lead_lag', 
                        type=bool,
                        default=False, 
                        help='Whether to apply lead-lag transform before computing sigker')
    parser.add_argument('--sde_split_training', 
                        type=bool,
                        default=True,
                        help='If True, split between deterministic and stochastic training steps for NSDEs.')
 
 
    # General things
    parser.add_argument('--exp_name', 
                        type=str,
                        default='simu')
    parser.add_argument('--GPU', 
                        type=bool, 
                        default=False,
                        help='Set False for CPU running')
    # parser.add_argument('--mode',  
    #                     type=str,
    #                     default='train_full',
    #                     choices=['train_full', 'only_train', 'only_rec', 'only_prior', 'only_posterior', 'opti_hyperparams'],
    #                     help='Mode of the script')
    parser.add_argument('--dataset', 
                        type=str,
                        default='Simu_SIR',
                        help='Name of the dataset in the data folder')
    parser.add_argument('--seed', 
                        type=int,
                        default=666)
    parser.add_argument('--train_dir', 
                        type=str,
                        default='/datasets')
    parser.add_argument('--save_dir', 
                        type=str,
                        default='/experiments/models',
                        help='Directory to save the model')
    # parser.add_argument('--exp_name', 
    #                     type=str,
    #                     default='debug')
    parser.add_argument('--file_dataset', 
                        type=str, 
                        default='data.csv')
    parser.add_argument('--config_file', 
                        type=str, 
                        default='config.yml', 
                        help='Name of the config file')
    parser.add_argument('--init_path', 
                        type=str, 
                        default='/path/to/HyperNSDE', 
                        help='Path to the project on running machine')  
    parser.add_argument('--optuna_study', 
                        type=str, 
                        default='optuna_study',
                        help="Name of the optuna study")
    parser.add_argument('--max_norm', 
                        type=float,
                        default=1.0,
                        help='max norm for the gradient clipping')
    parser.add_argument('--long_normalization', 
                        type=str, 
                        default='none',
                        choices=['max', 'zscore', ''], 
                        help="Way of normalizing the longitudinal data")


    # Data parameters
    parser.add_argument('--n_long_var', 
                        type=int,
                        default=19,
                        help='Number of longitudinal variables of the dataset')
    parser.add_argument('--t_visits', 
                        type=int,
                        default=500,
                        help='Number of visits')
    # parser.add_argument('--t_visits_obs', 
    #                     type=int,
    #                     default=100,
    #                     help='Number of visits observed')
    
    
    # Static data parameters
    parser.add_argument('--static_data', 
                        type=bool,
                        default=False, 
                        help='True if static data is used')
    parser.add_argument('--z_dim_static', 
                        type=int,
                        default=3,
                        help='Dimension of the zn of the Gaussian Mixture of the static data')
    parser.add_argument('--s_dim_static', 
                        type=int,
                        default=2,
                        help='Dimension of the sn of the Gaussian Mixture of the static data')
    parser.add_argument('--batch_norm_static', 
                        type=bool,
                        default=False,
                        help='True if batch normalization is used for static data')
    # parser.add_argument('--added_init_static', 
    #                     type=int,
    #                     default=0,
    #                     help='Always 0 at the beginning')
    
    
    # Latent model parameters
    parser.add_argument('--num_ode_layers', 
                        type=int,
                        default=3,
                        help='Number of layers for the ODE')
    parser.add_argument('--act_ode', 
                        type=str,
                        default='Identity',
                        choices=['Identity', 'Tanh', 'ReLU', 'Sigmoid', 'Softplus'], 
                        help='Activation function for the NODE')
    parser.add_argument('--num_hypernet_layers', 
                        type=int,
                        default=3,
                        help='Number of layers for the hypernetwork')
    parser.add_argument('--act_hypernetwork', 
                        type=str,
                        default='ReLU',
                        choices=['Identity', 'Tanh', 'ReLU', 'Sigmoid', 'Softplus'], 
                        help='Activation function for the hypernetwork')
    parser.add_argument('--latent_dim', 
                        type=int, 
                        default=3,
                        help='Latent dimension of the longitudinal variables')
    parser.add_argument('--nhidden', 
                        type=int,
                        default=64,
                        help='Hidden size latent model')
    parser.add_argument('--solver', 
                        type=str,
                        default='Adjoint',
                        choices=['Adjoint', 'Normal', 'Julia'],
                        help='Solver for the ODE system')
    parser.add_argument('--method_solver', 
                        type=str,
                        default='dopri5',
                        choices=['dopri5', 'dopri8', 'bosh3', 'fehlberg2', 'adaptive_heun', 'rk4', 'euler'],
                        help='Solver for the ODE system')
    parser.add_argument('--rtol', 
                        type=float,
                        default=1e-3,
                        help='rtol option for the odeint solver')
    parser.add_argument('--atol', 
                        type=float,
                        default=1e-3,
                        help='atol option for the odeint solver')
    
    # Memory optimization parameters
    parser.add_argument('--chunk_size', 
                        type=int,
                        default=None,
                        help='Chunk size for processing large time grids')
    # parser.add_argument('--use_mixed_precision', 
    #                     type=bool,
    #                     default=False,
    #                     help='Use mixed precision training to reduce memory usage')
    parser.add_argument('--memory_efficient_ode', 
                        type=bool,
                        default=False,
                        help='Use memory-efficient ODE integration')
    
    # parser.add_argument('--step_size_solver', 
    #                     type=float,
    #                     default=0.01,
    #                     help='h option for the odeint solver')

    # Lambda
    parser.add_argument('--estim_event_rate', 
                        type=bool,
                        default=False,
                        help='Either to estimate or not the event rate.')
    parser.add_argument('--lambda_dim',
                        type=int,
                        default=1,
                        help='Dimension of the lambda function, must be 1 or the number of longitudinal variables.')
    parser.add_argument('--lambda_mlp_size',  
                        type=int,
                        default=32,
                        help='For lambda MLP network: number of neurons in each hidden layer.')
    parser.add_argument('--lambda_num_layers',  
                        type=int,
                        default=1,
                        help='For lambda MLP network: number of layers.')
    parser.add_argument('--lambda_act',  
                        type=str,
                        default='Tanh',
                        help='For lambda MLP network: activation function.')
    

    # SDE
    parser.add_argument('--latent_model',  
                        type=str,
                        default='HyperNDEs',
                        choices=['MultiNDEs', 'StatMoNDEs', 'HyperNDEs'],
                        help='Type of model used for the longitudinal latent variable.')
    parser.add_argument('--sde',  
                        type=bool,
                        default=True,
                        help='True if the longitudinal latent model is supposed to be an SDE, False for just an ODE.') 
    parser.add_argument('--sde_type',  
                        type=str,
                        default='stratonovich',
                        choices=['stratonovich', 'ito'],
                        help='Type of solver for the SDE (Stratonovich or Ito) for torchsde.')
    parser.add_argument('--diff_type',  
                        type=str,
                        default='diagonal',
                        choices=["scalar", "additive", "diagonal", "general"],
                        help='Type of diffusion function in the SDE model for torchsde.')
    parser.add_argument('--diff_shape',  
                        type=int,
                        default=1,
                        help='Size of the Brownian Motion')
    parser.add_argument('--diff_mlp_size',  
                        type=int,
                        default=64,
                        help='For diffusion MLP network: number of neurons in each hidden layer.')
    parser.add_argument('--diff_mlp_num_layers',  
                        type=int,
                        default=3,
                        help='For diffusion MLP network: number of hidden layers.')
    parser.add_argument('--act_diff',  
                        type=str,
                        default='LipSwish',
                        help='For diffusion MLP network: activation function to use between layers.')
    parser.add_argument('--res_drift_mlp_size',  
                        type=int,
                        default=16,
                        help='For residual drift MLP network: number of neurons in each hidden layer.')
    parser.add_argument('--res_drift_mlp_num_layers',  
                        type=int,
                        default=2,
                        help='For residual drift MLP network: number of hidden layers.')
    parser.add_argument('--act_res_drift',  
                        type=str,
                        default='LipSwish',
                        help='For residual drift MLP network: activation function to use between layers.')
    parser.add_argument('--init_noise_size',  
                        type=int,
                        default=1,
                        help='How many noise dimensions to sample at the start of the SDE.')
    parser.add_argument('--init_mlp_size',  
                        type=int,
                        default=16,
                        help='For initial condition MLP network: number of neurons in each hidden layer.')
    # parser.add_argument('--init_mlp_num_layers',  
    #                     type=int,
    #                     default=3,
    #                     help='For initial condition MLP network: number of hidden layers.')
    parser.add_argument('--act_init',  
                        type=str,
                        default='LipSwish',
                        help='For initial condition MLP network: activation function to use between layers.')
    parser.add_argument('--fixed_init_cond', 
                        type=bool,
                        default=False, 
                        help='Whether to fix the starting point of the SDE or not.')
    
    # Training SDE 
    parser.add_argument('--sde_training',  
                        type=str,
                        default='MMD_SigKer',
                        choices=['MMD_FDM', 'MMD_SigKer'],
                        help='Type of method to train the latent Neural SDE model.')
    parser.add_argument('--subtract_initial_point',  
                        type=bool,
                        default=True,
                        help='Subtract initial point before calculating sde loss. You almost always want this to be true.')
    parser.add_argument('--sigma_kernel',  
                        type=float,
                        default=1.0,
                        help='Sigma in RBF kernel')
    parser.add_argument('--kernel_type',  
                        type=str,
                        default='rbf',
                        choices=['rbf', 'linear'],
                        help='Type of kernel to use in the Sigker discriminator.')
    parser.add_argument('--n_scalings_kernel',  
                        type=int,
                        default=8,
                        help='Number of samples to draw from Exp(1). ~8 tends to be a good choice.')
    parser.add_argument('--max_batch_kernel',  
                        type=int,
                        default=16,
                        help='Maximum batch size to pass through the Sigker discriminator.')
    
    
    
    # Decoder parameters
    parser.add_argument('--act_dec', 
                        type=str,
                        default='ReLU',
                        choices=['Identity', 'Tanh', 'ReLU'], 
                        help='Activation function for the decoder')
    parser.add_argument('--drop_dec', 
                        type=float,
                        default=0.1, 
                        help='Dropout rate for the decoder')
    parser.add_argument('--type_dec', 
                        type=str,
                        default='linear',
                        choices=['linear', 'nonlinear', 'multiNODEs'])
    parser.add_argument('--nhidden_dec', 
                        type=float,
                        default=16,
                        help='Number of hidden units for the decoder')
    # parser.add_argument('--type_dec', 
    #                     type=str,
    #                     default='LSTM',
    #                     choices=['RNN', 'LSTM', 'orig', 'orig_w_static', 'Linear'])
    # parser.add_argument('--nhidden_dec', 
    #                     type=float,
    #                     default=0.6974659448314735,
    #                     help='in percentage')
    # parser.add_argument('--dec_sig', 
    #                     type=str,
    #                     default='constant',
    #                     choices=['constant', 'continue', 'computed', 'none'], 
    #                     help='Type of the decoder for the variance')   
    

    # Encoder parameters 
    parser.add_argument('--type_enc', 
                        type=str,
                        default='none',
                        choices=['RNN', 'LSTM', 'none'])
    parser.add_argument('--nhidden_enc', 
                        type=float,
                        default=0.49866725741363976,
                        help='in percentage')
    parser.add_argument('--act_rnn', 
                        type=str,
                        default='Identity',
                        choices=['Identity', 'Tanh', 'ReLU', 'Sigmoid', 'Softplus'], 
                        help='Activation function for the RNN Encoder/Decoder, usefull if type_enc = RNN or type_dec = RNN.')    

    
    # Training parameters
    parser.add_argument('--train_set_size', 
                        type=float, 
                        default=0.75, 
                        help='Percentage of data in train set')
    parser.add_argument('--val_set_size', 
                        type=float, 
                        default=0.15, 
                        help='Percentage of data in validation set')
    parser.add_argument('--test_set_size', 
                        type=float, 
                        default=0.10, 
                        help='Percentage of data in test set')
    parser.add_argument('--batch_size', 
                        type=float,
                        default=0.3200910825235765,
                        help='in percentage of the dataset')
    parser.add_argument('--from_best', 
                        type=bool,
                        default=False, 
                        help='True if the model is loaded from the best model')
    parser.add_argument('--lr', 
                        type=float,
                        default=0.00601775920584171,
                        help='Learning rate')
    parser.add_argument('--num_epochs', 
                        type=int,
                        default=1900, 
                        help='Number of epochs')
    parser.add_argument('--epoch_init', 
                        type=int,
                        default=1, 
                        help='Initial epoch')
    parser.add_argument('--save_freq', 
                        type=int, 
                        default=300,
                        help='Frequency of saving the model')
    parser.add_argument('--print_freq', 
                        type=int, 
                        default=1,
                        help='Frequency of printing the results')
    parser.add_argument('--patience', 
                        type=int,
                        default=10, 
                        help='Number of epochs before early stopping, -1 if no early stopping')
    parser.add_argument('--loss_scaling_poisson',
                        type=float,
                        default=1,
                        help='Scaling factor of the Poisson loss.')
    parser.add_argument('--loss_scaling_ode',
                        type=float,
                        default=1,
                        help='Scaling factor of the ODE loss.')
    parser.add_argument('--loss_scaling_sde',
                        type=float,
                        default=1,
                        help='Scaling factor of the sde loss.')
    parser.add_argument('--loss_scaling_static',
                        type=float,
                        default=1,
                        help='Scaling factor of the static loss.')


    # Fix to LS for now
    parser.add_argument('--loss_aggregation', 
                        type=str,
                        default='LS',
                        choices=['UW', 'DWA', 'LS', 'MultiNODEs'], 
                        help='Aggregation method of the different losses to compute a global one: LS is Linear Scalarization, DWA is Dynamic Weight Average, UW is Uncertainty Weighting.')    
    
    if return_unknown:
        # Check if running in a Jupyter Notebook
        if 'ipykernel' in sys.modules:
            # If in a Jupyter Notebook, use default arguments
            config, unknown = parser.parse_known_args(args=[])
        else:
            # If not in a Jupyter Notebook, parse command-line arguments
            config, unknown = parser.parse_known_args()
        return config, unknown
    else:
        # Check if running in a Jupyter Notebook
        if 'ipykernel' in sys.modules:
            # If in a Jupyter Notebook, use default arguments
            config = parser.parse_args(args=[])
        else:
            # If not in a Jupyter Notebook, parse command-line arguments
            config = parser.parse_args()

        return config




    # parser.add_argument('--N_pop', 
    #                         type=int,
    #                         default=1,
    #                         help='Describes how often a complete population is generated')
    # parser.add_argument('--sigma_long', 
    #                     type=int,
    #                     default=1, 
    #                     help='Describes the level of noise in the reparametrization trick when using posterior sampling') 
    # parser.add_argument('--sigma_stat', 
    #                     type=int,
    #                     default=1,
    #                     help='Describes the level of noise in the reparametrization trick when using posterior sampling') 
    # parser.add_argument('--s_prob',
    #                     default=[],
    #                     help='vector with propability of s during prior sampling')
    # parser.add_argument('--degree', 
    #                     type=int,
    #                     default=4, 
    #                     help="Degree of the polynome in case type_ode='polynomial' ")



    """to check """

    #  # Specific simulation parameters
    # parser.add_argument('--time_max', 
    #                     type=int,
    #                     default=1,
    #                     help='Upper limit of simulated time')
    # parser.add_argument('--time_min', 
    #                     type=int,
    #                     default=0,
    #                     help='Lower limit of simulated time')
    # parser.add_argument('--time_steps', 
    #                     type=int,
    #                     default=2000,
    #                     help='Number of steps in the simulated time')