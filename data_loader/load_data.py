import os
import numpy as np
import pandas as pd
import torch
from .datasets import Simulation_Dataset

def compute_max(x, mask, abs_val=False):
    if abs_val:
        x = abs(x)    
    neg_inf = torch.tensor(float('-inf'), device=x.device, dtype=x.dtype)
    max_vals = torch.max(torch.where(mask.bool(), x, neg_inf), dim=1).values
    max_vals = torch.max(max_vals, dim=0).values  # Final max per feature
    return max_vals

def compute_min(x, mask):
    pos_inf = torch.tensor(float('inf'), device=x.device, dtype=x.dtype)
    x_masked = torch.where(mask.bool(), x, pos_inf)
    min_vals = x_masked.min(dim=1).values
    min_vals = min_vals.min(dim=0).values
    return min_vals

def compute_mean_std(x, mask):
    sum_vals = torch.sum(x * mask, dim=[0, 1])  # sum over patients and time
    count_vals = torch.sum(mask , dim=[0, 1])  # number of observed values
    mean_vals = sum_vals / count_vals
    squared_diff = (x - mean_vals)**2 * mask
    std_vals = torch.sqrt(torch.sum(squared_diff, dim=[0, 1]) / count_vals)
    return mean_vals, std_vals


def apply_normalization(x, mask, method, param_norm):

    if method == "absmax":
        norm_x = x / param_norm['max']
    elif method == "zscore":
        # norm_x = (x - param_norm['mean']) / param_norm['std']
        norm_x = x / param_norm['std']
    elif method == "classic zscore":
        norm_x = (x - param_norm['mean']) / param_norm['std']
    elif method == "min_max":
        norm_x = (x - param_norm['min']) / (param_norm['max'] - param_norm['min'] + 1e-8)
    else:
        raise ValueError("Unknown normalization method. Use 'absmax', 'min_max' or 'zscore'.")
    
    if mask is not None:
        norm_x = norm_x * mask  # Set masked values to 0 after normalization

    return norm_x


def apply_denormalization(norm_x, mask, method, param_norm):
    
    if method == "absmax":
        x = norm_x * param_norm['max']
    elif method == "zscore":
        # x = (norm_x * param_norm['std']) + param_norm['mean']
        x = (norm_x * param_norm['std']) 
    elif method == "classic zscore":
        x = (norm_x * param_norm['std']) + param_norm['mean']
    elif method == "min_max":
        x = norm_x * (param_norm['max'] - param_norm['min'] + 1e-8) + param_norm['min']
    else:
        raise ValueError("Unknown normalization method. Use 'absmax', 'min_max' or 'zscore'.")
    
    if mask is not None:
        x = x * mask  
        
    return x


def normalize_long(x, mask, method):
    # if isinstance(data, (torch.Tensor, np.ndarray)):
    #     x = data
    # else:
    #     data_norm = data
    #     x = data_norm[0]
    
    if method == "absmax":
        max_vals = compute_max(x, mask, abs_val=True)
        max_val = torch.max(max_vals)
        denom = max_val if max_val != 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)
        norm_x = x / denom
        norm_x = norm_x * mask  # Set masked values to 0 after normalization
        params_norm = {'max': denom}
    elif method == "zscore":
        mean_vals, std_vals = compute_mean_std(x, mask)
        # norm_x = (x - mean_vals) / std_vals
        norm_x = x / std_vals
        norm_x = norm_x * mask  # Set masked values to 0 after normalization
        params_norm = {'mean': mean_vals, 'std': std_vals}
    elif method == "min_max":
        max_vals = compute_max(x, mask)
        min_vals = compute_min(x, mask)
        norm_x = (x - min_vals) / (max_vals - min_vals + 1e-8)
        norm_x = norm_x * mask  # Set masked values to 0 after normalization
        params_norm = {'max': max_vals, 'min': min_vals}
    elif method == "classic zscore":
        mean_vals, std_vals = compute_mean_std(x, mask)
        norm_x = (x - mean_vals) / std_vals
        norm_x = norm_x * mask  # Set masked values to 0 after normalization
        params_norm = {'mean': mean_vals, 'std': std_vals}
    else:
        raise ValueError("Unknown normalization method. Use 'absmax', 'min_max' or 'zscore'.")
    
    return norm_x, params_norm
    
    # if isinstance(data, (torch.Tensor, np.ndarray)):
    #     return norm_x, params_norm
    # else:
    #     data_norm[0] = norm_x
    #     return data_norm, params_norm

    

def read_csv_values(path, to_torch=False, sep=',', header=None, index_col=None, engine='python', method='pandas'):
    """
    Read a CSV file and return its values as a numpy array or a torch tensor.
    
    Parameters
    ----------
        path : str
            Path to the CSV file.
        to_torch : bool, optional
            If True, convert the numpy array to a torch tensor. Default is False.
        sep : str, optional
            Delimiter to use. Default is ','.
        header : int, list of int, None, optional
            Row(s) to use as the column names. Default is None.
        index_col : int, str, sequence, or False, optional
            Column(s) to set as index. Default is None. 
        engine : str, optional
            Parser engine to use. Default is 'python'.
    Returns
    -------
        data : numpy.ndarray or torch.Tensor
            The values from the CSV file as a numpy array or torch tensor.
    """
    if method == 'pandas':
        data = pd.read_csv(path, sep=sep, header=header, index_col=index_col, engine=engine)
        data = data.values
    elif method == 'numpy':
        # Step 1: Load with NumPy as float32
        data = np.loadtxt(path, delimiter=sep, dtype=np.float32)

    if to_torch:
        data = torch.from_numpy(data).float()
    return data


def weighter(data):
    """
    Create a weight matrix and a values matrix from the input data.

    Parameters
    ----------
        data : torch.Tensor
            The input data tensor with shape (n_samples, n_timepoints, n_features).
    Returns
    -------
        values_matrix : torch.Tensor
            The values matrix with the same shape as the input data, where NaN values are replaced with 0.0.
        weight_matrix : torch.Tensor
            The weight matrix with the same shape as the input data, where NaN values are replaced with 0.0 and non-NaN values are replaced with 1.0.
    """
    mask = torch.isnan(data)
    weight_matrix = (~mask).float()
    values_matrix = torch.where(mask, torch.tensor(0.0), data)
    return values_matrix, weight_matrix


def read_static_data(data, types, missing):
    """
    Read static data and create a mask for missing values.

    Parameters
    ----------
        data : torch.Tensor
            The input data tensor with shape (n_samples, n_features).
        types : list
            A list of tuples indicating the type of each feature.
        missing : numpy.ndarray or None
            A numpy array indicating the missing values in the data. If None, no missing values are considered.
    Returns
    -------
        data : torch.Tensor
            The input data tensor with NaN values replaced by 0.0.
        types : list
            A list of tuples indicating the type of each feature.
        true_miss_mask : torch.Tensor
            A mask indicating the true missing values in the data.
    """

    # Sustitute NaN values by 0.0 and create the real missing value mask
    # true_miss_mask = torch.ones(data.shape[0], len(types))
    # if missing is not None:
    #     # The -1 is because the indexes in the csv start at 1
    #     true_miss_mask[missing[:, 0]-1, missing[:, 1]-1] = 0

    if missing is not None:
        true_miss_mask = missing
    else:
        true_miss_mask = torch.ones(data.shape)
    # Mask NaN values in the data
    nan_mask = torch.isnan(data)

    # Fill data depending on the types
    data_filler = []
    for i in range(len(types)):
        if types[i][0] in {'cat', 'ordinal'}:
            unique_vals = data[:, i][~torch.isnan(data[:, i])].unique()
            if unique_vals.numel() > 0:
                data_filler.append(unique_vals[0].item())
            else:
                data_filler.append(int(0))
        else:
            data_filler.append(0.0)

    # Replace NaN values with the corresponding filler for each column
    for i, filler in enumerate(data_filler):
        data[:, i] = torch.where(nan_mask[:, i], torch.tensor(filler, dtype=data.dtype, device=data.device), data[:, i])

    # Construct the data matrices
    data_complete = []
    for i in range(data.shape[1]):

        if types[i][0] == 'cat':
            # Get categories
            cat_data = data[:, i].to(torch.int64)
            categories, indexes = torch.unique(cat_data, return_inverse=True)

            # Transform categories to a vector of 0:num_categories
            new_categories = torch.arange(int(types[i][1]), dtype=torch.int64)
            cat_data = new_categories[indexes]

            # Create one hot encoding for the categories
            aux = torch.zeros((data.shape[0], len(new_categories)), dtype=torch.float32)
            aux[torch.arange(data.shape[0]), cat_data] = 1
            data_complete.append(aux)

        elif types[i][0] == 'ordinal':
            # Get categories
            cat_data = data[:, i].to(torch.int64)
            categories, indexes = torch.unique(cat_data, return_inverse=True)

            # Transform categories to a vector of 0:num_categories
            new_categories = torch.arange(int(types[i][1]), dtype=torch.int64)
            cat_data = new_categories[indexes]

            # Create thermometer encoding for the categories
            aux = torch.zeros((data.shape[0], 1 + len(new_categories)), dtype=torch.float32)
            aux[:, 0] = 1
            aux[torch.arange(data.shape[0]), 1 + cat_data] = 1
            aux = torch.cumsum(aux, dim=1)
            data_complete.append(aux)

        else:
            aux = data[:, i].unsqueeze(1)  # Add an extra dimension
            data_complete.append(aux)

    data_end = torch.cat(data_complete, dim=1)
    return data_end, types, true_miss_mask


def get_data(config, normalize_long=False):

    train_dir = os.path.join(config.train_dir, config.file_dataset)
    data = read_csv_values(os.path.join(train_dir, 'data_long.csv'), header=None, index_col=None, method='numpy')

    n_patients = len(data)
    data = torch.from_numpy(data).view(n_patients, config.t_visits, config.n_long_var + 1).float()

    x, mask = weighter(data[:,:,1:])
    config.t_steps = data[0, :, 0]
    config.t_over = (config.t_steps - config.t_steps[0]) / (config.t_steps[-1] - config.t_steps[0])

    if config.time_normalization:
        T = config.t_over    ## scaling time between 0 and 1
    else:
        T = config.t_steps 

    # First extract t0 and subtract it from longitundinal data
    if config.fixed_init_cond and config.t0_2_static:
        print("Subtracting t0 from longitundinal data and adding it to static data.")
        t0_values = torch.full((x.shape[0], x.shape[2]), float('nan'), device=x.device)
        for i in range(x.shape[0]):  # loop over patients
            for j in range(x.shape[2]):  # loop over variables
                # valid_idxs = torch.where(~torch.isnan(x[i, :, j]))[0]
                valid_idxs = torch.where(mask[i, :, j] > 0)[0]
                if len(valid_idxs) > 0:
                    first_idx = valid_idxs[0]
                    t0 = x[i, first_idx, j]
                    t0_values[i, j] = t0
                    x[i, :, j] = x[i, :, j] - t0 * mask[i, :, j]  # subtract t0 from observed points in the entire trajectory

    # Second normalize longitundinal data
    if normalize_long:
        if config.long_normalization != 'none':
            print("Normalizing longitundinal data using method:", config.long_normalization)
            x, params_norm = normalize_long(x, mask, method=config.long_normalization)
        else:
            print("No normalization applied to longitundinal data, set config.long_normalization to 'absmax', 'min_max' or 'zscore' to apply normalization.")
            params_norm = None
    else:
        params_norm = None

    if config.static_data:
        static_vals = read_csv_values(os.path.join(train_dir, 'data_static.csv'), to_torch=True)
        static_types = read_csv_values(os.path.join(train_dir, 'data_static_types.csv'), header=None)
        static_missing = read_csv_values(os.path.join(train_dir, 'data_static_missing.csv'), to_torch=True)

        if static_types.shape[1] > 3:
            static_types = static_types[:, -3:]

        if config.fixed_init_cond and config.t0_2_static:
            if ~torch.all(t0_values == 0.):
                static_vals = torch.cat([static_vals, t0_values], dim=1)
                new_static_types = np.array([['real', 1, 1]] * t0_values.shape[1], dtype=static_types.dtype)
                static_types = np.concatenate([static_types, new_static_types], axis=0)
                new_static_missing = torch.ones_like(t0_values, dtype=int)
                static_missing = torch.cat([static_missing, new_static_missing], dim=1)

        static_vals_dim_ind = static_vals.shape[0]
        static_vals_dim = static_vals.shape[1]
        static_onehot, static_types, static_true_miss_mask = read_static_data(static_vals, static_types, static_missing)
        static_onehot_dim = static_onehot.shape[1]
    else:
        static_vals_dim_ind = None
        static_vals_dim, static_onehot_dim = None, None
        static_onehot, static_types, static_true_miss_mask = None, None, None

    config.s_vals_dim_ind = static_vals_dim_ind
    config.s_vals_dim = static_vals_dim
    config.s_onehot_dim = static_onehot_dim
    return config, x, mask, T, static_onehot, static_types, static_true_miss_mask, static_vals, params_norm 


def load_dataset(config, only_data=False):
    """
    Load the dataset and create data loaders for training, validation, and testing.

    Parameters
    ----------
        config : object
            Configuration object containing dataset parameters.
        only_data : bool, optional      
            If True, return only the datasets without data loaders. Default is False.
    Returns
    -------
        config : object
            Updated configuration object with batch size.
        train_dataloader : DataLoader
            DataLoader for the training dataset.        
        val_dataloader : DataLoader
            DataLoader for the validation dataset.
        test_dataloader : DataLoader
            DataLoader for the testing dataset.
    """
    # Load data 
    config, x, mask, T, static_onehot, static_types, static_true_miss_mask, static_vals, params_norm = get_data(config, normalize_long=False)

    # Compute indices for train, validation, and test splits
    train_end = int(config.train_set_size * len(x))
    val_end = train_end + int(config.val_set_size * len(x))
    test_end = min(val_end + int(config.test_set_size * len(x)), len(x))
    
    # Define splits for data
    data_splits = {
        "train": [x[:train_end], mask[:train_end], T, static_onehot[:train_end], static_types, static_true_miss_mask[:train_end], static_vals[:train_end], params_norm],
        "validation": [x[train_end:val_end], mask[train_end:val_end], T, static_onehot[train_end:val_end], static_types, static_true_miss_mask[train_end:val_end], static_vals[train_end:val_end], params_norm],
        "test": [x[val_end:test_end], mask[val_end:test_end], T, static_onehot[val_end:test_end], static_types, static_true_miss_mask[val_end:test_end], static_vals[val_end:test_end], params_norm]
    }

    if config.long_normalization != 'none':
        print("Applying longitundinal normalization using method:", config.long_normalization)
        data_splits["train"][0], params_norm = normalize_long(data_splits["train"][0], data_splits["train"][1], method=config.long_normalization)
        data_splits["validation"][0] = apply_normalization(data_splits["validation"][0], data_splits["validation"][1], method=config.long_normalization, param_norm=params_norm)
        data_splits["test"][0] = apply_normalization(data_splits["test"][0], data_splits["test"][1], method=config.long_normalization, param_norm=params_norm)
    else:
        print("No normalization applied to longitundinal data.")

    train_data = data_splits["train"]
    val_data = data_splits["validation"]
    test_data = data_splits["test"]

    # Create datasets
    train_dataset = Simulation_Dataset(config, train_data)
    validation_dataset = Simulation_Dataset(config, val_data)
    test_dataset = Simulation_Dataset(config, test_data)
    
    if only_data:
        return train_dataset, validation_dataset, test_dataset
    else:
        # batch_size = int(np.round(config.batch_size * len(train_data[0])))
        # config.batch_size = batch_size
        batch_size = int(config.batch_size)
        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True) #, num_workers=0)
        val_dataloader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = True) #, num_workers=0)
        test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True) #, num_workers=0)
        return config, train_dataloader, val_dataloader, test_dataloader


def load_dataset_CV(data, idx_train, idx_test, only_data=False):
    """"
    Load the dataset and create data loaders for training and validation.

    Parameters
    ----------
        data : tuple    
            Tuple containing the dataset and its parameters.
        idx_train : list
            List of indices for the training set.
        idx_test : list
            List of indices for the validation set. 
        only_data : bool, optional
            If True, return only the datasets without data loaders. Default is False.
    Returns
    -------
        config : object
            Updated configuration object with batch size.
        train_dataloader or train_dataset : DataLoader or Dataset
            DataLoader or Dataset for the training dataset.
        val_dataloader or validation_dataset : DataLoader or Dataset
            DataLoader or Dataset for the validation dataset.  
    """
    config, x, mask, T, static_onehot, static_types, static_true_miss_mask, static_vals, params_norm = data

    # Define splits for data
    data_splits = {
        "train": [x[idx_train], mask[idx_train], T, static_onehot[idx_train], static_types, static_true_miss_mask[idx_train], static_vals[idx_train], params_norm],
        "validation": [x[idx_test], mask[idx_test], T, static_onehot[idx_test], static_types, static_true_miss_mask[idx_test], static_vals[idx_test], params_norm],
    }

    if config.long_normalization != 'none':
        print("Applying longitundinal normalization using method:", config.long_normalization)
        data_splits["train"][0], params_norm = normalize_long(data_splits["train"][0], data_splits["train"][1], method=config.long_normalization)
        data_splits["validation"][0] = apply_normalization(data_splits["validation"][0], data_splits["validation"][1], method=config.long_normalization, param_norm=params_norm)
    else:    
        print("No normalization applied to longitundinal data.")
    
    train_data = data_splits["train"]
    val_data = data_splits["validation"]

    # Create datasets
    train_dataset = Simulation_Dataset(config, train_data)
    validation_dataset = Simulation_Dataset(config, val_data)

    if only_data:
        return config, train_dataset, validation_dataset
    else:
        # batch_size = max(1,int(round(config.batch_size * len(data_splits["train"][0]))))
        batch_size = int(config.batch_size)
        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last = False, num_workers=4, pin_memory=True) 
        val_dataloader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = False, drop_last = False, num_workers=4, pin_memory=True) 
        return config, train_dataloader, val_dataloader
    

def get_loader(config, dataset):
    """"
    Load the dataset and create a data loader.

    Parameters
    ----------
        config : object
            Configuration object containing dataset parameters.
        dataset : Dataset
            Dataset object to be loaded.
    Returns
    -------
        config : object
            Updated configuration object with batch size.
        dataloader : DataLoader
            DataLoader for the dataset.
    """
    batch_size = int(config.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = False, num_workers=4, pin_memory=True) 
    return dataloader


def get_full_original_data(path, t_visits, n_long_var, normalize_time=False, static_data=True, long_normalization=None):

    full_orig_data_dict = {}

    data_long = read_csv_values(os.path.join(path, 'data_long.csv'), header=None, index_col=None, method='numpy')

    n_patients = len(data_long)
    data_long = torch.from_numpy(data_long).view(n_patients, t_visits, n_long_var + 1).float()

    x, mask = weighter(data_long[:,:,1:])
    t_steps = data_long[0, :, 0]
    t_over = (t_steps - t_steps[0]) / (t_steps[-1] - t_steps[0])

    if normalize_time:
        T = t_over    ## scaling time between 0 and 1
    else:
        T = t_steps 

    full_orig_data_dict["Long_Values"] = x
    full_orig_data_dict["Time_Grid"] = T 
    full_orig_data_dict["Mask"] = mask

    if long_normalization is not None:
        full_orig_data_dict["Long_Values"], params_norm = normalize_long(full_orig_data_dict["Long_Values"], full_orig_data_dict["Mask"], method=long_normalization)
        # if long_normalization == 'absmax':
        #     max_vals = compute_max(full_orig_data_dict["Long_Values"], abs_val=True)
        #     mean, std, min_vals = None, None, None
        # elif long_normalization == 'min_max':
        #     max_vals = compute_max(full_orig_data_dict["Long_Values"])
        #     min_vals = compute_min(full_orig_data_dict["Long_Values"])
        #     mean, std = None, None
        # elif long_normalization == 'zscore':
        #     mean, std = compute_mean_std(full_orig_data_dict["Long_Values"])
        #     params_norm
        #     max_vals, min_vals = None, None    
        # full_orig_data_dict["Long_Values"] = apply_normalization(None, full_orig_data_dict["Long_Values"], method=long_normalization, mean=mean, std=std, max_vals=max_vals, min_vals=min_vals)
        

    try: 
        data_long_reg_ode = read_csv_values(os.path.join(path, 'data_long_regular_ode.csv'), header=None, index_col=None, method='numpy')
        data_long_reg_sde = read_csv_values(os.path.join(path, 'data_long_regular_sde.csv'), header=None, index_col=None, method='numpy')
        data_survival_times = read_csv_values(os.path.join(path, 'data_survival_times.csv'), header=None, index_col=None, method='numpy')
        full_orig_data_dict["Long_Values_RegODE"] = torch.from_numpy(data_long_reg_ode).view(n_patients, t_visits, n_long_var + 1)[:,:,1:].float()
        full_orig_data_dict["Long_Values_RegSDE"] = torch.from_numpy(data_long_reg_sde).view(n_patients, t_visits, n_long_var + 1)[:,:,1:].float()
        full_orig_data_dict["Survival_Times"] = data_survival_times
        if long_normalization is not None:
            full_orig_data_dict["Long_Values_RegODE"] = apply_normalization(full_orig_data_dict["Long_Values_RegODE"], torch.ones_like(full_orig_data_dict["Long_Values_RegODE"]), method=long_normalization, param_norm=params_norm)
            full_orig_data_dict["Long_Values_RegSDE"] = apply_normalization(full_orig_data_dict["Long_Values_RegSDE"], torch.ones_like(full_orig_data_dict["Long_Values_RegSDE"]), method=long_normalization, param_norm=params_norm)
    except:
        pass

    if static_data:
        static_vals = read_csv_values(os.path.join(path, 'data_static.csv'), to_torch=True)
        full_orig_data_dict["Stat_Values"] = static_vals
        try: 
            static_vals_full = read_csv_values(os.path.join(path, 'data_static_full.csv'))
            full_orig_data_dict["Stat_Values_Full"] = static_vals_full
        except:
            pass
        static_types = read_csv_values(os.path.join(path, 'data_static_types.csv'), header=None)
        static_missing = read_csv_values(os.path.join(path, 'data_static_missing.csv'), to_torch=True)

        if static_types.shape[1] > 3:
            static_types = static_types[:, -3:]

        static_onehot, static_types, static_true_miss_mask = read_static_data(static_vals, static_types, static_missing)
        full_orig_data_dict["Stat_OneHot"] = static_onehot
        full_orig_data_dict["Stat_Types"] = static_types
        full_orig_data_dict["Stat_TrueMissMask"] = static_true_miss_mask 

    # Add Var_Names and Var_Names_Static

    return full_orig_data_dict