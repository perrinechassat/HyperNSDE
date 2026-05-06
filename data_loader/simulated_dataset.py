from re import S
import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass, fields
import dill
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


def compute_max(x_train, mask_train):
    # Ensure the '-inf' tensor is on the same device and has the same dtype as x_train
    x_train = abs(x_train)
    neg_inf = torch.tensor(float('-inf'), device=x_train.device, dtype=x_train.dtype)

    # Compute max (ignoring nan / masked values)
    max_vals = torch.max(torch.where(mask_train.bool(), x_train, neg_inf), dim=1).values
    max_vals = torch.max(max_vals, dim=0).values  # Final max per feature

    return max_vals

def compute_min(x_train, mask_train):
    pos_inf = torch.tensor(float('inf'), device=x_train.device, dtype=x_train.dtype)
    x_masked = torch.where(mask_train.bool(), x_train, pos_inf)

    min_vals = x_masked.min(dim=1).values
    min_vals = min_vals.min(dim=0).values

    return min_vals

def compute_mean_std(x_train, mask_train):
    # Compute mean and std (ignoring masked values)
    sum_vals = torch.sum(x_train * mask_train, dim=[0, 1])  # sum over patients and time
    count_vals = torch.sum(mask_train, dim=[0, 1])  # number of observed values
    mean_vals = sum_vals / count_vals

    squared_diff = (x_train - mean_vals)**2 * mask_train
    std_vals = torch.sqrt(torch.sum(squared_diff, dim=[0, 1]) / count_vals)

    return mean_vals, std_vals

def apply_normalization(x, mask, method="absmax"):
    x = torch.tensor(x)
    mask = torch.tensor(mask)
    
    # Normalize x
    if method == "absmax":
        max_vals = compute_max(x, mask)
        max_val = torch.max(max_vals)
        norm_x = x / max_val
    elif method == "zscore":
        mean, std = compute_mean_std(x, mask)
        norm_x = (x - mean) / std
    elif method == "min_max":
        min_vals = compute_min(x, mask)
        max_vals = compute_max(x, mask)
        norm_x = (x - min_vals) / (max_vals - min_vals + 1e-8)
    else:
        raise ValueError("Unknown normalization method. Use 'max' or 'zscore'.")

    return norm_x.numpy()

@dataclass
class SimulatedDataset:
    # Parameters
    seed: int
    n_samples: int
    n_sampling_times: int
    end_time: float
    n_static_feats: int
    dim: int
    scale: float
    xi: float
    hurst: float
    threshold: float
    percent_pts: float
    missing: bool
    var_static: float
    model: str
    lambda_func: callable = None # Optional, may not be serializable
    missing_static_rate: float = 0.0

    # Time series
    paths: np.ndarray = None # (n_samples, n_sampling_times, dim+1) 
    paths_sde: np.ndarray = None # (n_samples, n_sampling_times, dim+1)
    irr_paths: any = None # (n_samples, n_sampling_times, dim+1)
    irr_paths_NaN: any = None # (n_samples, n_sampling_times, dim+1)
    
    # Static features
    static_feats: np.ndarray = None # (n_samples, n_static_feats)
    static_feats_miss: np.ndarray = None # (n_samples, n_static_feats)
    static_types: np.ndarray = None # (n_static_feats, 3)
    static_missing: np.ndarray = None # (n_samples, n_static_feats)
    
    # Time information
    sampling_times: np.ndarray = None # (n_sampling_times)
    
    # # Survival labels
    survival_times: np.ndarray = None # (n_samples)        
    survival_inds: np.ndarray = None # (n_samples)
    
    # Other parameters
    params_model: dict = None 
    correlated: bool = None
    cond_init: bool = None


    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)

        # Save parameters except arrays and callable lambda_func
        params = {}
        for field in fields(self):
            val = getattr(self, field.name)
            if isinstance(val, (int, float, str, bool)) or val is None:
                params[field.name] = val
        
        with open(os.path.join(folder, "parameters.txt"), "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")

        # Save lambda_func
        with open(os.path.join(folder, "lambda_func.pkl"), "wb") as f:
            dill.dump(self.lambda_func, f)

        # Save numpy arrays in their expected formats
        np.savetxt(os.path.join(folder, "data_long_regular_ode.csv"),
                   self.paths.reshape((self.n_samples, -1)), delimiter=",")
        np.savetxt(os.path.join(folder, "data_long_regular_sde.csv"),
                   self.paths_sde.reshape((self.n_samples, -1)), delimiter=",")
        np.savetxt(os.path.join(folder, "data_long.csv"),
                   self.irr_paths_NaN.reshape((self.n_samples, -1)), delimiter=",")

        np.savetxt(os.path.join(folder, "data_survival_times.csv"), self.survival_times, delimiter=",")

        np.savetxt(os.path.join(folder, "data_static_full.csv"), self.static_feats, delimiter=",")
        np.savetxt(os.path.join(folder, "data_static.csv"), self.static_feats_miss, delimiter=",")
        np.savetxt(os.path.join(folder, "data_static_types.csv"), self.static_types, delimiter=",", fmt='%s')
        np.savetxt(os.path.join(folder, "data_static_missing.csv"), self.static_missing, delimiter=",")
        np.save(os.path.join(folder, "params_model.npy"), self.params_model)

    @staticmethod
    def load(folder: str):
        # Load parameters first (very simple parsing of parameters.txt)
        params = {}
        with open(os.path.join(folder, "parameters.txt"), "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Try to convert numeric types
                    try:
                        val = float(val)
                        if val.is_integer():
                            val = int(val)
                    except:
                        pass
                    if val == "True":
                        val = True
                    elif val == "False":
                        val = False
                    elif val == "None":
                        val = None
                    params[key] = val

        # Cast to int or float where applicable
        for key in ['n_samples', 'n_sampling_times', 'n_static_feats', 'dim']:
            if key in params:
                params[key] = int(params[key])

        n_samples = params.get("n_samples")

        # Load arrays irr_paths_NaN, paths and paths_sde
        paths = np.loadtxt(os.path.join(folder, "data_long_regular_ode.csv"), delimiter=",")
        paths = paths.reshape((n_samples, paths.shape[1]//(params.get("dim")+1), params.get("dim")+1))
        paths_sde = np.loadtxt(os.path.join(folder, "data_long_regular_sde.csv"), delimiter=",")
        paths_sde = paths_sde.reshape((n_samples, paths_sde.shape[1]//(params.get("dim")+1), params.get("dim")+1))
        irr_paths_NaN = np.loadtxt(os.path.join(folder, "data_long.csv"), delimiter=",")
        irr_paths_NaN = irr_paths_NaN.reshape((n_samples, irr_paths_NaN.shape[1]//(params.get("dim")+1), params.get("dim")+1))  # reshape here might need dim adjustment depending on your data
        sampling_times = irr_paths_NaN[0,:,0]

        survival_times = np.loadtxt(os.path.join(folder, "data_survival_times.csv"), delimiter=",")
        static_feats = np.loadtxt(os.path.join(folder, "data_static_full.csv"), delimiter=",")
        static_feats_miss = np.loadtxt(os.path.join(folder, "data_static.csv"), delimiter=",")
        static_types = np.loadtxt(os.path.join(folder, "data_static_types.csv"), delimiter=",", dtype=str)
        static_missing = np.loadtxt(os.path.join(folder, "data_static_missing.csv"), delimiter=",")

        # Load lambda_func
        with open(os.path.join(folder, "lambda_func.pkl"), "rb") as f:
            lambda_func = dill.load(f)

        params_model = np.load(os.path.join(folder, "params_model.npy"), allow_pickle=True).item()

        # Recreate the dataset object
        return SimulatedDataset(
            seed=params.get("seed"),
            n_samples=params.get("n_samples"),
            n_sampling_times=params.get("n_sampling_times"),
            end_time=params.get("end_time"),
            n_static_feats=params.get("n_static_feats"),
            dim=params.get("dim"),
            scale=params.get("scale"),
            xi=params.get("xi"),
            hurst=params.get("hurst"),
            threshold=params.get("threshold"),
            percent_pts=params.get("percent_pts"),
            missing=params.get("missing"),
            var_static=params.get("var_static"),
            lambda_func=lambda_func,
            missing_static_rate=params.get("missing_static_rate"),
            model=params.get("model"),
            paths=paths,
            paths_sde=paths_sde,
            # irr_paths=irr_paths,
            irr_paths_NaN=irr_paths_NaN,
            static_feats=static_feats,
            static_feats_miss=static_feats_miss,
            static_types=static_types,
            static_missing=static_missing,
            sampling_times=sampling_times,
            survival_times=survival_times,
            # survival_inds=surv_inds,
            params_model=params_model, 
            correlated=params.get("correlated"), 
            cond_init=params.get("cond_init")
        )


    def plot_static_feats(self, use_missing=True, feat_comparison_name=None):
        """
        Plot static features in the dataset.

        Args:
        - use_missing (bool): If True, use self.static_feats_miss, else use self.static_feats.
        - feat_comparison_name (str): Optional, name of feature to compare others against (used for hue/grouping).
        """
        # Select which static features to plot
        data = self.static_feats_miss if use_missing else self.static_feats
        feat_types = self.static_types  # shape (n_static_feats, 3) -> [type, dim, num_categories]

        # Convert to DataFrame for easier handling
        feat_names = [f"feat_{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=feat_names)

        # Build a structured feature type list like in original code
        feat_types_dict = []
        for i, (ftype, fdim, ncat) in enumerate(feat_types):
            feat_types_dict.append({
                "name": feat_names[i],
                "type": ftype,
                "dim": int(fdim),
                "ncat": int(ncat) if ncat != '' else None
            })

        # Identify comparison feature index if provided
        feat_comparison_index = None
        if feat_comparison_name is not None:
            for i, f in enumerate(feat_types_dict):
                if f["name"] == feat_comparison_name:
                    feat_comparison_index = i
                    break

        # Layout
        num_features = len(feat_types_dict)
        if feat_comparison_name is not None:
            n_cols = (num_features - 1) // 2 + (num_features - 1) % 2
        else:
            n_cols = num_features // 2 + num_features % 2
        fig, axes = plt.subplots(n_cols, 2, figsize=(18, 2.5 * num_features))
        plt.subplots_adjust(wspace=0.2, hspace=0.35)
        axes = axes.flatten()

        # Plot each feature
        for i, feature in enumerate(feat_types_dict):
            if feat_comparison_index is not None and i == feat_comparison_index:
                continue  # skip comparison feature
            ax = axes[i if feat_comparison_index is None else (i if i < feat_comparison_index else i - 1)]
            feature_type = feature['type']
            feat_name = feature['name']
            if feature_type in ['cat', 'ordinal']:
                if feat_comparison_index is not None:
                    sns.countplot(data=df, x=feat_name, hue=feat_names[feat_comparison_index], alpha=0.8, ax=ax)
                    ax.legend(title=feat_names[feat_comparison_index]).set_visible(True)
                else:
                    sns.countplot(data=df, x=feat_name, alpha=0.8, ax=ax)
                ax.set_title(f"Count plot of {feat_name} ({feature_type})", fontsize=16, fontweight="bold")
                ax.set_xlabel("")
                n_class = df[feat_name].nunique()
                if n_class > 20:
                    ticks = ax.get_xticks()
                    labels = ax.get_xticklabels()
                    step = max(1, len(labels) // 10)
                    ax.set_xticks(ticks[::step])
                    ax.set_xticklabels([label.get_text() for label in labels[::step]])
            else:  # real-valued
                if feat_comparison_index is not None:
                    sns.violinplot(data=df, x=feat_names[feat_comparison_index], y=feat_name, ax=ax)
                    ax.set_xlabel(feat_names[feat_comparison_index], fontsize=16, fontweight="semibold")
                else:
                    sns.histplot(df[feat_name], kde=True, ax=ax)
                ax.set_title(f"Distribution of {feat_name} ({feature_type})", fontsize=16, fontweight="bold")
            ax.grid(True)
            ax.set_ylabel("Count", fontsize=16, fontweight="semibold")

        plt.show()

    def plot_model_params(self, feat_comparison_name=None):
        """
        Plot model parameters from params_model (all except 'weights') for each longitudinal feature.
        Colors points/densities according to the comparison static feature.
        
        - Categorical feature → n distinct colors.
        - Continuous feature → color gradient.
        """

        params = self.params_model
        if not isinstance(params, dict):
            raise ValueError("params_model should be a dict of arrays.")

        param_keys = [k for k in params.keys() if k.lower() != "weights"]
        if len(param_keys) == 0:
            raise ValueError("No parameters to plot (only 'weights' found).")

        hue_data = None
        if feat_comparison_name is not None:
            feat_names = [f"feat_{i}" for i in range(self.static_feats.shape[1])]
            if feat_comparison_name not in feat_names:
                raise ValueError(f"Feature {feat_comparison_name} not found in static features.")
            comparison_idx = feat_names.index(feat_comparison_name)
            hue_data = self.static_feats[:, comparison_idx]
            hue_type = self.static_types[comparison_idx, 0]
            if hue_type == 'cat':
                hue_is_categorical = True
            else:
                hue_is_categorical = False

        for param_name in param_keys:
            arr = np.array(params[param_name])
            if arr.ndim != 2:
                raise ValueError(f"Parameter '{param_name}' must be 2D (n_samples, n_longitudinal_features).")
            n_samples, n_long_features = arr.shape
            n_rows = n_long_features // 3 + int(n_long_features % 3 > 0)

            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
            axes = np.atleast_1d(axes).flatten()
            plt.suptitle(f"Parameter: {param_name}", fontsize=16, fontweight="bold")

            for i in range(n_long_features):
                df_param = pd.DataFrame({"value": arr[:, i]})
                if hue_data is not None:
                    df_param[feat_comparison_name] = hue_data

                    if hue_is_categorical:
                        sns.histplot(data=df_param, x="value", hue=feat_comparison_name,
                                    element="step", stat="density", common_norm=False,
                                    palette="tab10", ax=axes[i])
                        # axes[i].legend(title=feat_comparison_name)
                        # plt.legend(title=feat_comparison_name)
                    else:
                        # Continuous color: scatter plot for distribution
                        sns.scatterplot(data=df_param, x=np.arange(len(df_param)),
                                        y="value", hue=feat_comparison_name,
                                        palette="viridis", hue_norm=(hue_data.min(), hue_data.max()),
                                        ax=axes[i], s=20, edgecolor=None)
                        axes[i].set_xlabel("Sample index")
                else:
                    sns.histplot(df_param["value"], kde=True, ax=axes[i])

                axes[i].set_title(f"{param_name} - Long. feat. {i}", fontsize=14, fontweight="semibold")
                axes[i].grid(True)
                axes[i].set_ylabel("Value", fontsize=12)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

    def plot_longitudinal_feats(self, feat_comparison_name=None, n_samples_to_plot=None, method_scale=None):
        
        hue_data = None
        if feat_comparison_name is not None:
            feat_names = [f"feat_{i}" for i in range(self.static_feats.shape[1])]
            if feat_comparison_name not in feat_names:
                raise ValueError(f"Feature {feat_comparison_name} not found in static features.")
            comparison_idx = feat_names.index(feat_comparison_name)
            hue_data = self.static_feats[:, comparison_idx]
            hue_type = self.static_types[comparison_idx, 0]
            if hue_type == 'cat':
                hue_is_categorical = True
            else:
                hue_is_categorical = False
        else:
            hue_is_categorical = None

        # Sélection aléatoire d'échantillons
        if n_samples_to_plot is None or n_samples_to_plot > self.n_samples:
            n_samples_to_plot = self.n_samples
        sample_indices = np.random.choice(self.n_samples, n_samples_to_plot, replace=False)

        # Dictionnaire des trois matrices de trajectoires
        trajectories_dict = {
            "paths": self.paths,
            "paths_sde": self.paths_sde,
            "irr_paths_NaN": self.irr_paths_NaN
        }

        if method_scale is not None:
            trajectories_dict["paths_sde_scaled"] = self.paths_sde
            trajectories_dict["paths_sde_scaled"][:, :, 1:] = apply_normalization(self.paths_sde[:, :, 1:], np.ones(self.paths_sde[:, :, 1:].shape), method=method_scale)

        for traj_name, traj_data in trajectories_dict.items():
            if traj_data is None:
                continue  # sauter si pas dispo

            time_points = self.sampling_times
            n_features = traj_data.shape[2] - 1  # colonne 0 = temps

            fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5), sharex=True)
            if n_features == 1:
                axes = [axes]

            # Couleurs selon type de feature
            if hue_is_categorical:
                categories = np.unique(hue_data)
                cmap = cm.get_cmap("tab10")
                color_map = {cat: cmap(i) for i, cat in enumerate(categories)}
            else:
                cmap = cm.get_cmap("gnuplot")
                norm = plt.Normalize(vmin=np.min(hue_data), vmax=np.max(hue_data))

            for i_feat in range(n_features):
                ax = axes[i_feat]
                for idx in sample_indices:
                    y_values = traj_data[idx, :, i_feat+1]  # +1 car col0 = temps
                    if hue_is_categorical is not None:
                        if hue_is_categorical:
                            color = color_map[hue_data[idx]]
                        else:
                            color = cmap(norm(hue_data[idx]))
                        ax.plot(time_points, y_values, color=color, alpha=0.6)
                    else:
                        ax.plot(time_points, y_values, alpha=0.6)

                ax.set_title(f"{traj_name} - Feat. {i_feat+1}")
                ax.set_ylabel("Value")
                ax.grid(True)

            axes[-1].set_xlabel("Time")

            # Ajouter légende
            if hue_is_categorical:
                handles = [plt.Line2D([0], [0], color=color_map[cat], lw=2) for cat in categories]
                labels = [str(cat) for cat in categories]
                fig.legend(handles, labels, title=feat_comparison_name, loc="upper right")
            else:
                # Add colorbar for continuous values
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                
                # Create colorbar and position it
                cbar = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=20)
                cbar.set_label(feat_comparison_name, rotation=270, labelpad=15)
                fig.subplots_adjust(right=1)  # Make space for colorbar
                cbar.ax.set_position([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

            fig.suptitle(f"{traj_name} - Colored by {feat_comparison_name}")
            plt.tight_layout()
            plt.show()



    def plot_lambda_func_vs_events(self):
        # Extract all event times into a flat array
        all_events = []
        for i in range(self.irr_paths_NaN.shape[0]):  # Loop over patients
            mask = ~np.isnan(self.irr_paths_NaN[i, :, :].sum(axis=1))  # Keep time points with any non-NaN variable
            times = self.sampling_times[mask]  # Assuming you have a matching time grid array
            all_events.extend(times)
        all_events = np.array(all_events)

        # --------------------------
        # PLOT λ(t) AND HISTOGRAM
        # --------------------------
        lam_vals = [self.lambda_func(t) for t in self.sampling_times]
        fig, ax1 = plt.subplots(figsize=(8, 4))

        # Plot λ(t)
        ax1.plot(self.sampling_times, lam_vals, 'tab:orange', label=r"$\lambda(t)$ (theoretical)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Intensity", color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:orange')
        ax1.grid(True)

        # Plot histogram of events on second y-axis
        ax2 = ax1.twinx()
        ax2.hist(all_events, bins=30, density=True, edgecolor='black',color='tab:blue', alpha=0.5, label="Empirical event times")
        ax2.set_ylabel("Number of events", color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Legends
        fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
        plt.title("Theoretical λ(t) vs Empirical Event Distribution")
        plt.tight_layout()
        plt.show()


