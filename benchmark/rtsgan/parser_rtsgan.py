import argparse
import sys 
import random
from sys import maxsize
import time

def base_parser(return_unknown=False):
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--init_path", required=True, dest="init_path", help="initial path of project")   
    parser.add_argument("--config_file", required=True, dest="config_file", help="name of the config file .yml")   
    parser.add_argument("--dataset", required=False, dest="dataset", help=".pkl file to use")     
    parser.add_argument("--force", default="", dest="force", help="schedule")     
    parser.add_argument("--devi", default="0", dest="devi", help="gpu")
    parser.add_argument("--epochs", default=800, dest="epochs", type=int,
                        help="Number of full passes through training set for autoencoder")
    parser.add_argument("--iterations", default=15000, dest="iterations", type=int,
                        help="Number of iterations through training set for WGAN")
    parser.add_argument("--d_update", default=5, dest="d_update", type=int,
                        help="discriminator updates per generator update")
    parser.add_argument("--log-dir", default="../OU_result", dest="log_dir",
                        help="Directory where to write logs / serialized models")
    parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                        help="Name for this task, use a comprehensive one")
    parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                        help="Random seed of Python and NumPy")
    parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")
    parser.add_argument("--fix_ae", dest="fix_ae", default=None, help="Test mode")
    parser.add_argument("--fix_gan", dest="fix_gan", default=None, help="Test mode")

    parser.add_argument("--ae_batch_size", default=128, dest="ae_batch_size", type=int,
                        help="Minibatch size for autoencoder")
    parser.add_argument("--gan_batch_size", default=512, dest="gan_batch_size", type=int,
                        help="Minibatch size for WGAN")
    parser.add_argument("--embed_dim", default=512, dest="embed_dim", type=int, help="dim of hidden state")
    parser.add_argument("--hidden_dim", default=128, dest="hidden_dim", type=int, help="dim of GRU hidden state")
    parser.add_argument("--layers", default=3, dest="layers", type=int, help="layers")
    parser.add_argument("--ae_lr", default=1e-3, dest="ae_lr", type=float, help="autoencoder learning rate")
    parser.add_argument("--weight_decay", default=0, dest="weight_decay", type=float, help="weight decay")
    parser.add_argument("--dropout", default=0.0, dest="dropout", type=float,
                        help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")

    parser.add_argument("--gan_lr", default=1e-4, dest="gan_lr", type=float, help="GAN learning rate")
    parser.add_argument("--gan_alpha", default=0.99, dest="gan_alpha", type=float, help="for RMSprop")
    parser.add_argument("--noise_dim", default=512, dest="noise_dim", type=int, help="dim of WGAN noise state")

    parser.add_argument("--N_generated_datasets", default=50, dest="N_generated_datasets", type=int,
                        help="Number of generated synthetic datasets")


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

