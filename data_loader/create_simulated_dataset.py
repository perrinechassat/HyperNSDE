# Code for creating a simulated dataset from the simulated_data_bis.py file

import numpy as np
import sys
sys.path.append('../')
from data_loader.simulated_data_models import Simulation


if __name__ == "__main__":

    # Define the parameters of the simulation
    seed = 0
    n_samples = 1000
    n_sampling_times = 200
    end_time = 7
    n_static_feats = 2
    dim = 3
    scale = np.array([0.2, 1, 0.5, 0.05])
    xi = 0.2
    hurst = 0.4
    threshold = 100
    percent_pts = 0.5 # if 1: 100% of the points are used (no missing values), if < 1: subsampling
    missing = True # if False: same irregular grid per patient if percent_pts < 1, if True: different irregular grid per patient if percent_pts < 1. Delault: True.
    var_static = 0.07
    model = "ou" # "cir" # "ou"
    lambda_func = lambda t:  ((percent_pts*100/end_time) + (1 - (percent_pts*100/end_time)) * (1/2) * np.sin(-2 * np.pi * t / end_time))
    missing_static_rate = 0
    correlated = False
    cond_init = True
    subfolder = "simulated_dataset_sde_loss_2"
    folder = f"/path/to/HyperNSDE/datasets/Simu_{model}/{subfolder}"

    # Create the simulated dataset
    simulated_dataset = Simulation(seed=seed).generate_simulated_dataset(n_samples=n_samples, 
                                                                         n_sampling_times=n_sampling_times, 
                                                                         end_time=end_time, 
                                                                         n_static_feats=n_static_feats, 
                                                                         dim=dim, 
                                                                         scale=scale, 
                                                                         xi=xi, 
                                                                         hurst=hurst, 
                                                                         threshold=threshold, 
                                                                         percent_pts=percent_pts, 
                                                                         missing=missing, 
                                                                         var_static=var_static, 
                                                                         lambda_func=lambda_func, 
                                                                         missing_static_rate=missing_static_rate, 
                                                                         model=model, 
                                                                         correlated=correlated, 
                                                                         cond_init=cond_init)

    # Save the simulated dataset
    simulated_dataset.save(folder)

    print(f"Simulated dataset saved to {folder}")
