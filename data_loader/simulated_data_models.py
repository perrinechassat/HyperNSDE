import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
from stochastic.processes.continuous import FractionalBrownianMotion, BrownianMotion
from scipy.stats import poisson
from torch.distributions.poisson import Poisson
import scipy.special as scsp
import sys
sys.path.append('../')
from data_loader.simulated_dataset import SimulatedDataset

class Simulation():
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def get_static_feats(self, n_samples, n_features=5, var=None, missing_rate=0):
        n_numeric_features = int(n_features / 2)
        if var is None:
            var = 1e-1
        cov = np.diag(var * np.ones(n_numeric_features))
        W_numeric = self.rng.multivariate_normal(np.zeros(n_numeric_features), cov, size=n_samples)

        n_binary_features = n_features - n_numeric_features
        cov_bin = np.diag(1e-2 * np.ones(max(n_binary_features,0)))
        if n_binary_features > 0:
            pre_W_bin = self.rng.multivariate_normal(np.zeros(n_binary_features), cov_bin, size=n_samples)
            coefs = self.rng.standard_normal(n_binary_features)
            # probit = 0.5 * (1 + scsp.erf(pre_W_bin * coefs / np.sqrt(2)))
            probit = norm.cdf(pre_W_bin * coefs)
            u = self.rng.random((n_samples, n_binary_features))
            W_bin = (u < probit).astype(int)
        else:
            W_bin = np.empty((n_samples,0))

        W = np.concatenate((W_numeric , W_bin), axis=1) if W_bin.size else W_numeric
        W_types = []
        for i in range(n_numeric_features):
            W_types.append(['real', 1, 1])
        for i in range(n_binary_features):
            W_types.append(['cat', 2, 1])
        W_types = np.array(W_types)

        W_full = W.copy()
        if missing_rate > 0:
            n_missing = int(n_samples * missing_rate)
            missing_indices = self.rng.choice(np.arange(n_samples), n_missing, replace=False)
            missing_features = self.rng.choice(np.arange(n_features), n_missing)
            W_missing = np.ones(W.shape, dtype=int)
            for i in range(n_missing):
                W[missing_indices[i], missing_features[i]] = np.nan
                W_missing[missing_indices[i], missing_features[i]] = 0
        else:
            W_missing = np.ones(W.shape, dtype=int)

        return W_full, W, W_types, W_missing

    def get_path(self, n_samples, n_sampling_times, end_time, static_features, dim=3, scale=0.7, xi=0.5, hurst=0.5, model='ou', correlated=False, cond_init=False):
        model = model.lower()
        if model == 'ou':
            return self._get_path_ou(n_samples=n_samples, n_sampling_times=n_sampling_times, end_time=end_time, static_features=static_features, dim=dim, scale=scale, xi=xi, hurst=hurst, correlated=correlated, cond_init=cond_init)
        # elif model == 'cir':
        #     return self._get_path_correlated_cir_fbm(n_samples=n_samples, n_sampling_times=n_sampling_times, end_time=end_time, static_features=static_features, dim=dim, scale=scale, hurst=hurst)
        # elif model == 'gompertz':
        #     return self._get_path_gompertz(n_samples=n_samples, n_sampling_times=n_sampling_times, end_time=end_time, static_features=static_features, dim=dim, scale=scale, hurst=hurst)
        else:
            raise ValueError(f"Unknown model: {model}")

    def get_irregular_path(self, paths, surv_time, percent_pts=0.7, missing=True, lambda_func=None):
        sampling_times = np.array(paths[0, :, 0])
        n_samples = paths.shape[0]
        dim = paths.shape[-1] - 1
        irr_paths = pd.DataFrame(columns=['long_feature_%s' % (l + 1) for l in range(dim)])
        n_sampling_times = len(sampling_times)

        if percent_pts >= 1:
            irr_paths_with_nan = paths.copy()
            for i in range(n_samples):
                irr_paths.loc[i] = [pd.Series(paths[i, :, l+1], index=sampling_times) for l in range(dim)]
        else:
            if missing:
                irr_paths_with_nan = np.full(paths.shape, np.nan)
                irr_paths_with_nan[:, :, 0] = paths[:, :, 0]
                for i in range(n_samples):
                    if lambda_func=="from_paths":
                        grid = torch.from_numpy(sampling_times.astype(np.float32))
                        paths_i = torch.from_numpy(paths[i, :, 1:].astype(np.float32))
                        N_obs_mean = percent_pts*100
                        constant = N_obs_mean / (sampling_times[-1] * np.mean(
                            np.exp(-paths_i.sum(dim=1)).numpy()
                        ))
                        new_grid, idx_new_grid = self.extract_poisson_process_from_regular_grid_int(grid, paths_i, frequence=constant)
                        sampling_times_i = new_grid.detach().numpy()
                        sampling_times_i_idx = idx_new_grid.detach().numpy()
                    else:
                        if lambda_func is not None:
                            grid = torch.from_numpy(sampling_times.astype(np.float32))
                            new_grid, idx_new_grid = self.extract_poisson_process_from_regular_grid(grid, lambda_func)
                            sampling_times_i = new_grid.detach().numpy()
                            sampling_times_i_idx = idx_new_grid.detach().numpy()
                        else:
                            n_sampling_times_i = self.rng.choice(np.arange(int(n_sampling_times*percent_pts), n_sampling_times))
                            # n_sampling_times_i = self.rng.integers(max(1,int(n_sampling_times*percent_pts)), n_sampling_times+1)
                            sampling_times_i_idx = np.sort(self.rng.choice(np.arange(n_sampling_times), min(n_sampling_times_i, n_sampling_times), replace=False))

                    irr_paths_with_nan[i, sampling_times_i_idx, 1:] = paths[i, sampling_times_i_idx, 1:]
                    path_i = []
                    sampling_times_i = []
                    for t in sampling_times_i_idx:
                        path_i.append(paths[i, t, 1:])
                        sampling_times_i.append(sampling_times[t])
                    irr_paths.loc[i] = [pd.Series(np.array(path_i)[:, l], index=sampling_times_i) for l in range(dim)]
            else:
                n_sampling_times_irreg = max(1, int(n_sampling_times*percent_pts))
                sampling_times_irreg_idx = np.sort(self.rng.choice(np.arange(n_sampling_times), n_sampling_times_irreg, replace=False))
                irr_paths_with_nan = np.full((paths.shape[0], n_sampling_times_irreg, paths.shape[2]), np.nan)
                irr_paths_with_nan[:, :, :] = paths[:, sampling_times_irreg_idx, :]
                for i in range(n_samples):
                    irr_paths.loc[i] = [pd.Series(paths[i, sampling_times_irreg_idx, l+1], index=sampling_times_irreg_idx) for l in range(dim)]

        return irr_paths, irr_paths_with_nan

    def get_trajectory(self, path):
        return np.array(path)[:, 1:] if path is not None else None

    def get_survival_label(self, paths, end_time, threshold=2.5):
        max_paths = paths[:,:,1:].sum(axis=2) - threshold
        idxs = np.argmax(max_paths > 0, axis=1)
        sampling_times_ext = np.array(paths[:, :, 0])
        surv_times = np.take(sampling_times_ext, idxs)
        surv_times[surv_times == 0] = end_time
        surv_inds = (idxs != 0)
        return np.array(surv_times), np.array(surv_inds)

    def generate_simulated_dataset(self, n_samples, n_sampling_times, end_time, n_static_feats=5, dim=3, scale=0.7, xi=0.5, threshold=5, hurst=0.5, percent_pts=0.7, missing=True, var_static=None, lambda_func=None, missing_static_rate=0, model='ou', correlated=False, cond_init=False):
        static_feats, static_feats_miss, static_types, static_missing = self.get_static_feats(n_samples=n_samples, n_features=n_static_feats, var=var_static, missing_rate=missing_static_rate)
        paths, paths_sde, sampling_times, params = self.get_path(n_samples=n_samples, n_sampling_times=n_sampling_times, end_time=end_time,
                                            static_features=static_feats, dim=dim, scale=scale, xi=xi, hurst=hurst, model=model, correlated=correlated, cond_init=cond_init)
        surv_times, surv_inds = self.get_survival_label(paths=paths_sde, end_time=end_time, threshold=threshold)
        irr_paths, irr_paths_NaN = self.get_irregular_path(paths=paths_sde, surv_time=surv_times, percent_pts=percent_pts, missing=missing, lambda_func=lambda_func)

        return SimulatedDataset(
            seed=self.seed,
            n_samples=n_samples,
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
            paths=paths,
            paths_sde=paths_sde,
            irr_paths=irr_paths,
            irr_paths_NaN=irr_paths_NaN,
            static_feats=static_feats,
            static_feats_miss=static_feats_miss,
            static_types=static_types,
            static_missing=static_missing,
            sampling_times=sampling_times,
            survival_times=surv_times,
            survival_inds=surv_inds,
            params_model=params, 
            correlated=correlated, 
            cond_init=cond_init
        )

    def generate_grid_inhomogeneous_poisson_process(self, length, lambda_func):
        t = torch.linspace(0, length, 500)
        intensity = lambda_func(t)
        max_intensity = torch.max(intensity)
        Lambda = torch.sum(intensity) * (t[1] - t[0]) # mean number of points in grid
        # N = int(self.rng.poisson(Lambda.item())) if Lambda.item()>0 else 0
        N = poisson.rvs(Lambda.item()) # number of points in the generated grid
        points = []
        while len(points) < N:
            # x_sample = self.rng.random() * length
            # u = self.rng.random() * max_intensity.item()
            x_sample = torch.rand(1).item() * length
            u = torch.rand(1).item() * max_intensity

            if u < lambda_func(torch.tensor(x_sample)):
                points.append(x_sample)
        return points

    def extract_poisson_process_from_regular_grid(self, reg_grid, lambda_func):
        T = reg_grid[-1].item() if isinstance(reg_grid[-1], torch.Tensor) else float(reg_grid[-1])
        intensities = lambda_func(reg_grid)
        lambda_max = torch.max(intensities)
        N = int(torch.distributions.Poisson(T * lambda_max).sample().item())
        K = len(reg_grid)
        if N == 0:
            return torch.tensor([]), torch.tensor([])
        I = torch.randint(0, K, (N,))
        M = torch.zeros(I.shape)
        for l in range(N):
            num_interval = I[l].item()
            u = torch.rand(1).item() * lambda_max
            # u = self.rng.random() * lambda_max.item()
            t_left = num_interval * T / K
            t_right = (num_interval + 1) * T / K
            if u <= ((lambda_func(torch.tensor(t_left)) + lambda_func(torch.tensor(t_right))) / 2).item():
                M[l] = 1.
        if M.sum() == 0:
            return torch.tensor([]), torch.tensor([])
        idx_new_grid, _ = torch.sort(torch.unique(I[M.bool()]))
        new_grid = reg_grid[idx_new_grid]
        return new_grid, idx_new_grid
    
    def extract_poisson_process_from_regular_grid_int(self, reg_grid, paths_i, frequence):
        T = reg_grid[-1].item() if isinstance(reg_grid[-1], torch.Tensor) else float(reg_grid[-1])
        intensity = frequence*torch.exp(-paths_i.sum(dim=1))
        lambda_max = torch.max(intensity)
        N = int(torch.distributions.Poisson(T * lambda_max).sample().item())
        K = len(reg_grid)
        if N == 0:
            return torch.tensor([]), torch.tensor([])
        I = torch.randint(0, K, (N,))
        M = torch.zeros(I.shape)
        for l in range(N):
            num_interval = I[l].item()
            u = torch.rand(1).item() * lambda_max
            # u = self.rng.random() * lambda_max.item()
            # t_left = num_interval * T / K
            # t_right = (num_interval + 1) * T / K
            if u <= ((intensity[num_interval-1] + intensity[num_interval]) / 2).item():
                M[l] = 1.
        if M.sum() == 0:
            return torch.tensor([]), torch.tensor([])
        idx_new_grid, _ = torch.sort(torch.unique(I[M.bool()]))
        new_grid = reg_grid[idx_new_grid]
        return new_grid, idx_new_grid  


    # ------------------ Specific models ------------------
    def _get_path_ou(self, n_samples, n_sampling_times, end_time, static_features, dim=3, scale=0.7, xi=0.5, hurst=0.7, correlated=False, cond_init=False):
        d = dim
        sampling_times = np.linspace(0, end_time, n_sampling_times)
        paths = np.zeros((n_samples, n_sampling_times, d+1))
        paths_sde = np.zeros_like(paths)
        paths[:,:,0] = sampling_times
        paths_sde[:,:,0] = sampling_times
        # Map parameters
        params, weights = self.map_params(static_features, n_params=2, d=d, scale=scale, rng=self.rng, return_weights=True, mapping_type="linear")
        theta_params = params[:,:,0]
        mu_params = params[:,:,1]
        # Brownian motion
        if correlated:
            spatial_fbm_cov = self._random_correlation_matrix(d)
            L_space = np.linalg.cholesky(spatial_fbm_cov + 1e-12 * np.eye(d))
        fbm = FractionalBrownianMotion(t=end_time, hurst=hurst, rng=self.rng)
        # Generate paths
        for i in range(n_samples):
            # x_ode = np.zeros(d)
            # x_sde = np.zeros(d)
            paths[i,0,1:] = np.zeros(d)
            paths_sde[i,0,1:] = np.zeros(d)
            theta = theta_params[i]
            mu = mu_params[i]

            if correlated:
                fbm_uncorr = np.array([fbm.sample(n_sampling_times - 1) for _ in range(d)]) 
                fbm_corr = L_space @ fbm_uncorr
                dB = np.diff(fbm_corr, axis=1)
            else:
                fbm_paths = np.array([fbm.sample(n_sampling_times - 1) for _ in range(d)])  # shape (d, n_sampling_times - 1)
                dB = np.diff(fbm_paths, axis=1)  # shape (d, n_sampling_times - 1)

            for t in range(1, n_sampling_times):
                dt = sampling_times[t] - sampling_times[t-1]
                paths[i,t,1:] = paths[i,t-1,1:] - theta * (paths[i,t-1,1:] - mu) * dt
                paths_sde[i,t,1:] = paths_sde[i,t-1,1:] - theta * (paths_sde[i,t-1,1:] - mu) * dt + xi * dB[:, t-1]   
                # x_ode += - theta * (x_ode - mu) * dt
                # drift = - theta * (x_sde - mu) * dt
                # diffusion = xi * dB[:, t-1]     
                # x_sde = x_sde + drift + diffusion
                # paths[i,t,1:] = x_ode
                # paths_sde[i,t,1:] = x_sde   # paths with noise included

        if cond_init: 
            init_values = self.map_params(static_features, n_params=1, d=d, scale=scale[-1], rng=self.rng, return_weights=False, mapping_type="poly")
            paths[:, :, 1:] = paths[:, :, 1:] + init_values[:, :, 0][:, np.newaxis, :]
            paths_sde[:, :, 1:] = paths_sde[:, :, 1:] + init_values[:, :, 0][:, np.newaxis, :]

        params = {'theta': theta_params, 'mu': mu_params, 'weights': weights}
        return paths, paths_sde, sampling_times, params


    def _random_correlation_matrix(self, d, ridge=1e-6):
        A = self.rng.normal(size=(d, d))
        cov = A @ A.T
        cov += ridge * np.eye(d)  # Regularize to ensure positive definiteness
        D_inv = np.diag(1 / np.sqrt(np.diag(cov)))
        R = D_inv @ cov @ D_inv
        return R
        

    # ---- Parameter Mappers ----
    def map_params(self, static_features, n_params, d, scale = 1.0, rng = None, return_weights = False, mapping_type="linear"):
        """
        Maps static features to per-sample parameter vectors via a linear model with random weights.

        Args:
            static_features: Array of shape (n_samples, n_static_features).
            n_params: Number of distinct parameter sets to generate (e.g., 2 for theta and mu).
            dim: Number of dimensions per parameter vector (e.g., number of longitudinal features).
            scale: Multiplicative scale applied to the linear mapping outputs.
            rng: Optional numpy Generator for reproducibility; if None, uses default.
            return_weights: If True, also returns the random weights used.
            mapping_type: Type of mapping to use.

        Returns:
            params: Array of shape (n_samples, dim, n_params).
            weights (optional): Array of shape (n_params, n_static_features, dim).
        """
        if rng is None:
            rng = np.random.default_rng()
        n_samples, n_static_features = static_features.shape
        # weights: one matrix per parameter set
        weights = rng.standard_normal(size=(n_params, n_static_features, d))

        if mapping_type == "linear":
            params = np.empty((n_samples, d, n_params))
            if np.isscalar(scale):
                for p in range(n_params):
                    params[:, :, p] = scale * static_features @ weights[p]
            else:
                for p in range(n_params):
                    params[:, :, p] = scale[p] * static_features @ weights[p]

        elif mapping_type == "poly":
            params = np.empty((n_samples, d, n_params))
            if np.isscalar(scale):
                for p in range(n_params):
                    params[:, :, p] = scale * static_features @ weights[p] + 0.1 * (static_features ** 2) @ weights[p]
            else:
                for p in range(n_params):
                    params[:, :, p] = scale[p] * static_features @ weights[p] + 0.1 * (static_features ** 2) @ weights[p]
                
        elif mapping_type == "sigmoid":
            params = np.empty((n_samples, d, n_params))
            if np.isscalar(scale):
                for p in range(n_params):
                    params[:, :, p] = scale * torch.sigmoid(static_features @ weights[p]).numpy() * 2.0
            else:
                for p in range(n_params):
                    params[:, :, p] = scale[p] * torch.sigmoid(static_features @ weights[p]).numpy() * 2.0
                

        elif mapping_type == "exp":
            params = np.empty((n_samples, d, n_params))
            if np.isscalar(scale):
                for p in range(n_params):
                    params[:, :, p] = scale * torch.exp(0.1 * (static_features @ weights[p])).numpy()
            else:
                for p in range(n_params):
                    params[:, :, p] = scale[p] * torch.exp(0.1 * (static_features @ weights[p])).numpy()

        elif mapping_type == "softplus":
            params = np.empty((n_samples, d, n_params))
            if np.isscalar(scale):
                for p in range(n_params):
                    params[:, :, p] = scale * torch.nn.functional.softplus(torch.tensor(static_features @ weights[p])).numpy()
            else:
                for p in range(n_params):
                    params[:, :, p] = scale[p] * torch.nn.functional.softplus(torch.tensor(static_features @ weights[p])).numpy()

        else:
            raise ValueError(f"Unknown mapping_type: {mapping_type}")

        if return_weights:
            return params, weights
        return params

    # def _get_path_correlated_cir_fbm(self, n_samples, n_sampling_times, end_time, static_features, dim=3, scale=0.7, hurst=0.5, full_truncation=True):
    #     """
    #     Simulate multivariate correlated CIR driven by fractional Brownian motion increments.
        
    #     SDE (componentwise):
    #     dX_l(t) = kappa_l * (theta_l - X_l(t)) dt + sigma_l * sqrt(X_l(t)) dB^H_l(t)
    #     where B^H has correlated components according to `spatial_fbm_cov`. 
    #     """
    #     d = dim
    #     sampling_times = np.linspace(0, end_time, n_sampling_times)
    #     paths = np.zeros((n_samples, n_sampling_times, d+1))
    #     paths_sde = np.zeros_like(paths)
    #     paths[:,:,0] = sampling_times
    #     paths_sde[:,:,0] = sampling_times

    #     params, weights = self.map_params(static_features, n_params=3, d=d, scale=scale, rng=self.rng, return_weights=True, mapping_type="softplus")
    #     kappa_params = params[:,:,0]
    #     theta_params = params[:,:,1]
    #     sigma_params = params[:,:,2]
    #     print(f"kappa min: {kappa_params.min()}, kappa max: {kappa_params.max()}")
    #     print(f"theta min: {theta_params.min()}, theta max: {theta_params.max()}")
    #     print(f"sigma min: {sigma_params.min()}, sigma max: {sigma_params.max()}")
    #     params = {'kappa': kappa_params, 'theta': theta_params, 'sigma': sigma_params, 'weights': weights}
    #     spatial_fbm_cov = self._random_correlation_matrix(d)
    #     L_space = np.linalg.cholesky(spatial_fbm_cov + 1e-12 * np.eye(d))
    #     print(spatial_fbm_cov)
    #     fbm = FractionalBrownianMotion(t=end_time, hurst=hurst, rng=self.rng)

    #     for i in range(n_samples):
    #         kappa = kappa_params[i]
    #         theta = theta_params[i]
    #         raw_sigma_raw = sigma_params[i]
    #         raw_frac = raw_sigma_raw  # this is static_features @ weights_sigma (numpy)
    #         frac = 0.2 + 0.8 * (1.0 / (1.0 + np.exp(-raw_frac)))  # map to [0.2,1.0]
    #         max_sigma = np.sqrt(np.maximum(2.0 * kappa * theta, 1e-12))
    #         sigma = frac * max_sigma  # preserves Feller but fraction is feature-dependent
    #         feller = 2.0 * kappa * theta - sigma**2
    #         print(f"feller condition: {feller}")

    #         fbm_uncorr = np.zeros((d, n_sampling_times))
    #         for j in range(d):
    #             fbm_uncorr[j, :] = fbm.sample(n_sampling_times-1)
    #         fbm_corr = L_space @ fbm_uncorr
    #         fbm_paths = np.transpose(fbm_corr, (1,0))
    #         dB = np.diff(fbm_paths, axis=0)

    #         # x = theta.copy()
    #         x = np.zeros(d)
    #         paths[i,0,1:] = x
    #         paths_sde[i,0,1:] = x
    #         for k in range(1, n_sampling_times):
    #             dt = sampling_times[k] - sampling_times[k-1]
    #             sqrt_x = np.sqrt(np.maximum(x, 0.0))
    #             dB_step = dB[k-1, :]
    #             drift = kappa * (theta - x) * dt
    #             diffusion = sigma * sqrt_x * dB_step
    #             paths[i,k,1:] = paths[i,k-1,1:] + drift
    #             x_temp = x + drift + diffusion
    #             x = np.maximum(x_temp, 0.0)
    #             paths_sde[i,k,1:] = x

    #     return paths, paths_sde, sampling_times, params
    
    # def _get_path_gompertz(self, n_samples, n_sampling_times, end_time, static_features, dim=3, scale=0.7, hurst=0.5):
    #     d = dim
    #     sampling_times = np.linspace(0, end_time, n_sampling_times)
    #     paths = np.zeros((n_samples, n_sampling_times, d+1))
    #     paths_sde = np.zeros_like(paths)
    #     paths[:,:,0] = sampling_times
    #     paths_sde[:,:,0] = sampling_times

    #     params, weights = self.map_params(static_features, n_params=3, d=d, scale=scale, rng=self.rng, return_weights=True, mapping_type="softplus")
    #     alpha_params = params[:,:,0]
    #     K_params = params[:,:,1]
    #     sigma_params = params[:,:,2]
    #     params = {'alpha': alpha_params, 'K': K_params, 'sigma': sigma_params, 'weights': weights}
    #     gamma_comp = 0.1 * (self.rng.normal(size=(d,d)))

    #     spatial_fbm_cov = self._random_correlation_matrix(d)
    #     L_space = np.linalg.cholesky(spatial_fbm_cov + 1e-12 * np.eye(d))
    #     print(spatial_fbm_cov)
    #     fbm = FractionalBrownianMotion(t=end_time, hurst=hurst, rng=self.rng)

    #     for i in range(n_samples):
    #         alpha = alpha_params[i]
    #         K = K_params[i] 
    #         sigma = sigma_params[i]

    #         fbm_uncorr = np.zeros((d, n_sampling_times))
    #         for j in range(d):
    #             fbm_uncorr[j, :] = fbm.sample(n_sampling_times-1)
    #         fbm_corr = L_space @ fbm_uncorr
    #         # fbm_corr = fbm_uncorr
    #         fbm_paths = np.transpose(fbm_corr, (1,0))
    #         dB = np.diff(fbm_paths, axis=0)

    #         x = 0.1 * K
    #         paths[i,0,1:] = x
    #         paths_sde[i,0,1:] = x
    #         for t in range(1, n_sampling_times):
    #             dt = sampling_times[t] - sampling_times[t-1]
    #             interaction = (gamma_comp @ x)
    #             drift = (alpha * x * np.log(np.maximum(K / np.maximum(x,1e-8), 1e-8)) - interaction) * dt
    #             diffusion = sigma * x * dB[t-1, :]
    #             x = np.maximum( x + drift + diffusion, 0.0)
    #             paths[i,t,1:] = paths[i,t-1,1:] + drift
    #             paths_sde[i,t,1:] = x

    #     return paths, paths_sde, sampling_times, params

    # def _get_path_ou(self, n_samples, n_sampling_times, end_time, static_features, dim=3, scale=0.7, xi=0.5, hurst=0.7):
    #     d = dim
    #     sampling_times = np.linspace(0, end_time, n_sampling_times)
    #     paths = np.zeros((n_samples, n_sampling_times, d+1))
    #     paths_sde = np.zeros_like(paths)
    #     paths[:,:,0] = sampling_times
    #     paths_sde[:,:,0] = sampling_times
    #     # Map parameters
    #     params, weights = self.map_params(static_features, n_params=2, d=d, scale=scale, rng=self.rng, return_weights=True, mapping_type="linear")
    #     theta_params = params[:,:,0]
    #     mu_params = params[:,:,1]
    #     # Brownian motion
    #     fbm = FractionalBrownianMotion(t=end_time, hurst=hurst, rng=self.rng)
    #     # Generate paths
    #     for i in range(n_samples):
    #         x = np.zeros(d)
    #         paths[i,0,1:] = x
    #         paths_sde[i,0,1:] = x
    #         theta = theta_params[i]
    #         mu = mu_params[i]
    #         for t in range(1, n_sampling_times):
    #             dt = sampling_times[t] - sampling_times[t-1]
    #             drift = - theta * (x - mu)
    #             x = x + drift * dt
    #             paths[i,t,1:] = x
    #         for l in range(1, d+1):
    #             paths_sde[i,:,l] = paths[i,:,l] + xi * fbm.sample(n_sampling_times - 1)

    #     params = {'theta': theta_params, 'mu': mu_params, 'weights': weights}
    #     return paths, paths_sde, sampling_times, params

    # def _get_path_cir(self, n_samples, n_sampling_times, end_time, static_features, dim=3, scale=0.7, xi=0.5, hurst=0.5):
    #     d = dim
    #     sampling_times = np.linspace(0, end_time, n_sampling_times)
    #     paths = np.zeros((n_samples, n_sampling_times, d+1))
    #     paths_sde = np.zeros_like(paths)
    #     paths[:,:,0] = sampling_times
    #     paths_sde[:,:,0] = sampling_times

    #     params, weights = self.map_params(static_features, n_params=3, d=d, scale=scale, rng=self.rng, return_weights=True, mapping_type="softplus")
    #     kappa_params = params[:,:,0]
    #     theta_params = params[:,:,1]
    #     sigma_params = params[:,:,2]
    #     params = {'kappa': kappa_params, 'theta': theta_params, 'sigma': sigma_params, 'weights': weights}

    #     R = self._random_correlation_matrix(d, ridge=1e-6)
    #     L = np.linalg.cholesky(R)

    #     for i in range(n_samples):
    #         kappa = kappa_params[i]
    #         theta = theta_params[i]
    #         raw_sigma_raw = sigma_params[i]

    #         raw_frac = raw_sigma_raw  # this is static_features @ weights_sigma (numpy)
    #         frac = 0.2 + 0.8 * (1.0 / (1.0 + np.exp(-raw_frac)))  # map to [0.2,1.0]
    #         max_sigma = np.sqrt(np.maximum(2.0 * kappa * theta, 1e-12))
    #         sigma = frac * max_sigma  # preserves Feller but fraction is feature-dependent

    #         feller = 2.0 * kappa * theta - sigma**2
    #         print(f"feller condition: {feller}")

    #         x = theta.copy()
    #         paths[i,0,1:] = x
    #         paths_sde[i,0,1:] = x
    #         for t in range(1, n_sampling_times):
    #             dt = sampling_times[t] - sampling_times[t-1]
    #             sqrt_x = np.sqrt(np.maximum(x, 0.0))
    #             dW_indep = self.rng.normal(size=d)  # independent normals
    #             dW_corr = L @ dW_indep * np.sqrt(dt)  # correlated increments
    #             dx = kappa * (theta - x) * dt + sigma * sqrt_x * dW_corr
    #             x = np.maximum(x + dx, 0.0)
    #             paths[i,t,1:] = x - sigma * sqrt_x * dW_corr
    #             paths_sde[i,t,1:] = x

    #     return paths, paths_sde, sampling_times, params

    # def _get_path_lotka(self, n_samples, n_sampling_times, end_time, static_features, dim=3, scale=0.7, xi=0.5):
    #     d = dim
    #     sampling_times = np.linspace(0, end_time, n_sampling_times)
    #     paths = np.zeros((n_samples, n_sampling_times, d+1))
    #     paths_sde = np.zeros_like(paths)
    #     paths[:,:,0] = sampling_times
    #     paths_sde[:,:,0] = sampling_times

    #     n_static = static_features.shape[1]
    #     Alpha_map = self.rng.normal(scale=0.3, size=(n_static, d))
    #     Beta_base = 0.1 * self.rng.normal(size=(d,d))
    #     Sigma_map = self.rng.normal(scale=0.2, size=(n_static, d))

    #     params = {'alpha': np.zeros((n_samples,d)), 'beta': Beta_base, 'sigma': np.zeros((n_samples,d))}
    #     for i in range(n_samples):
    #         alpha = static_features[i].dot(Alpha_map)
    #         sigma = np.abs(static_features[i].dot(Sigma_map)) + 1e-3
    #         params['alpha'][i,:] = alpha
    #         params['sigma'][i,:] = sigma
    #         x = np.abs(0.5 + 0.1*self.rng.normal(size=d))
    #         paths[i,0,1:] = x
    #         paths_sde[i,0,1:] = x
    #         for t in range(1, n_sampling_times):
    #             dt = sampling_times[t] - sampling_times[t-1]
    #             interaction = x * (alpha + Beta_base @ x)
    #             drift = interaction
    #             dW = self.rng.normal(size=d) * np.sqrt(dt)
    #             x_det = x + drift * dt
    #             x = np.maximum(x_det + sigma * x * dW, 0.0)
    #             paths[i,t,1:] = x_det
    #             paths_sde[i,t,1:] = x
    #     return paths, paths_sde, sampling_times, params
    

        
    


