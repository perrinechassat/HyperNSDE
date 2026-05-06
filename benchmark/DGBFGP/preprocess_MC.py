import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Enter configuration for generating data')
    parser.add_argument('--destination', type=str, default='./preprocess_datasets', help='Path to save the generated dataset')
    parser.add_argument('--P_test', type=int, default=100, help='Number of unique instances in the test set')
    parser.add_argument('--P_val', type=int, default=100, help='Number of unique instances in the validation set')
    parser.add_argument('--T_test', type=int, default=0, help='Timepoint index to split test/val trajectories (out of 200)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    return vars(parser.parse_args())

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

def preprocess_simulation_data(folder_input, mc_id):
    """
    Loads the user's simulation data and merges them into the DGBFGP format.
    Replace the mock data generation below with your actual pd.read_csv() logic.
    """
    # ---------------------------------------------------------
    # Load your actual CSVs here.
    # df_long = pd.read_csv('data_long.csv') # Should be formatted to have columns: [ID, Time, L1, L2, L3]
    # df_static = pd.read_csv('data_static.csv') # Should have columns: [ID, Static_Real, Static_Cat]
    # -----------------------------------------

    INIT_PATH = "/path/to/HyperNSDE"
    # INIT_PATH = "/path/to/HyperNSDE"

    path_data_OU = INIT_PATH + '/datasets/Simu_OU/{}/simulated_dataset_MC_{}'.format(folder_input, mc_id)
    t_visits = 200
    n_long_var = 3

    data = read_csv_values(os.path.join(path_data_OU, 'data_long.csv'), header=None, index_col=None, method='numpy')
    n_patients = len(data)
    data = torch.from_numpy(data).view(n_patients, t_visits, n_long_var + 1).float()

    idx_train, idx_val = train_test_split(np.arange(n_patients), test_size=0.2, random_state=mc_id, shuffle=True)
    idx_train, idx_out = train_test_split(idx_train, test_size=0.75, random_state=mc_id, shuffle=True)
    idx_train_val = np.concatenate((idx_train,idx_val))
    data = data[idx_train_val]

    x, mask = weighter(data[:,:,1:])
    time_grid = data[0, :, 0]
    N, T, V = data[:,:,1:].shape
    time = torch.tensor(time_grid).float()

    # Flatten Longi:
    patient_ids = torch.arange(N).repeat_interleave(T)
    times = time.repeat(N)
    values = data[:,:,1:].reshape(N*T, V)
    var_cols = [f"L{i+1}" for i in range(V)]
    df_long = pd.DataFrame(values.numpy(), columns=var_cols)
    df_long.insert(0, "Time", times.numpy())
    df_long.insert(0, "ID", patient_ids.numpy())

    # Static:
    static_vals = read_csv_values(os.path.join(path_data_OU, 'data_static.csv'), to_torch=True)
    N_static, v_s = static_vals.shape
    var_cols_stat = [f"S{i+1}" for i in range(v_s)]
    df_static = pd.DataFrame(static_vals.numpy(), columns=var_cols_stat)
    df_static.insert(0, "ID", np.arange(N_static))


    merged_df = pd.merge(df_long, df_static, on='ID', how='left')
    X_cols = ['ID', 'Time'] + var_cols_stat
    X = merged_df[X_cols].values
    Y = merged_df[var_cols].values
    Y_mask = ~np.isnan(Y)
    Y_mask = Y_mask.astype(int)
    Y = np.nan_to_num(Y, nan=0.0) # Replace NaNs with 0 (mask handles the missingness)
    
    return {"X": X, "y": Y, "y_mask": Y_mask}

def train_val_test_split(x, y, y_mask, P_test, P_val, T_test, seed):
    """
    Adapted from Physionet_generate.py to split longitudinal trajectories.
    Subjects are split, and test/val subjects have their future trajectories masked.
    """
    unique_ids = np.unique(x[:, 0])
    
    # Select P_test + P_val patients for testing and validation
    # idx_train, idx_val = train_test_split(unique_ids, test_size=0.2, random_state=seed, shuffle=True)
    idx_train, idx_val = train_test_split(unique_ids, test_size=0.5, random_state=seed, shuffle=False)
    # val_test_ids = np.random.choice(unique_ids, P_test + P_val, replace=False) 
    val_test_ids = idx_val
    test_ids = val_test_ids[:P_test]
    val_ids = val_test_ids[P_test:]
    
    val_test_subject_flag = np.isin(x[:, 0], val_test_ids)

    x_test, y_test, y_test_init, y_mask_test = [], [], [], []
    x_val, y_val, y_val_init, y_mask_val = [], [], [], []
    x_train_half, y_train_half, y_mask_train_half = [], [], []

    # Process test and validation subjects
    for id in val_test_ids:
        subject_x = x[x[:, 0] == id]
        subject_y = y[x[:, 0] == id]
        subject_y_mask = y_mask[x[:, 0] == id]
        
        for i in range(subject_x.shape[0]):
            if i >= T_test: # Timepoints after T_test go to validation/testing
                if id in test_ids:
                    x_test.append(subject_x[i])
                    y_test.append(subject_y[i])
                    y_mask_test.append(subject_y_mask[i])
                    y_test_init.append(subject_y[0]) # Initial state
                else:
                    x_val.append(subject_x[i])
                    y_val.append(subject_y[i])
                    y_mask_val.append(subject_y_mask[i])
                    y_val_init.append(subject_y[0])
            else: # Earlier timepoints act as training context
                x_train_half.append(subject_x[i])
                y_train_half.append(subject_y[i])
                y_mask_train_half.append(subject_y_mask[i])
    
    # Non-test subjects are entirely put into the train set
    x_train = list(x[~val_test_subject_flag]) + x_train_half
    y_train = list(y[~val_test_subject_flag]) + y_train_half
    y_mask_train = list(y_mask[~val_test_subject_flag]) + y_mask_train_half

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_mask_train = np.array(y_mask_train)

    # Initialize y_train_init
    y_train_init = np.zeros_like(y_train)
    for id in np.unique(x_train[:, 0]):
        mask = x_train[:, 0] == id
        y_train_init[mask] = y_train[mask][0]

    return (x_train, np.array(x_val), np.array(x_test), 
            y_train, np.array(y_val), np.array(y_test), 
            y_mask_train, np.array(y_mask_val), np.array(y_mask_test), 
            y_train_init, np.array(y_val_init), np.array(y_test_init), idx_train, idx_val)


if __name__ == "__main__":
    opt = parse_arguments()

    # 1. Load and format data
    folder_input = "monte_carlo"
    mc_id = 0
    np.random.seed(mc_id)

    data = preprocess_simulation_data(folder_input, mc_id)
    x, y, y_mask = data["X"], data["y"], data["y_mask"]

    # Ensure IDs start from 0 and are continuous
    ids = np.unique(x[:, 0])
    id_dict = {id: i for i, id in enumerate(ids)}
    x[:, 0] = [id_dict[id] for id in x[:, 0]]

    # 2. Create directories
    destination = opt['destination']
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(destination, split), exist_ok=True)

    # Re-apply NaNs for proper standard scaling calculations
    y[~y_mask.astype(bool)] = np.nan

    # 3. Split the data
    (x_train, x_val, x_test, y_train, y_val, y_test, 
     y_mask_train, y_mask_val, y_mask_test, 
     y_train_init, y_val_init, y_test_init, idx_train, idx_val) = train_val_test_split(
        x, y, y_mask, opt['P_test'], opt['P_val'], opt['T_test'], seed=mc_id
    )

    # 4. Standardize Covariates (Index 1=Time, Index 2=Static_Real) 
    # Index 3 (Static_Cat) is left unscaled.
    numerical_covariates = [1, 2] 
    x_scaler = StandardScaler()
    x_train[:, numerical_covariates] = x_scaler.fit_transform(x_train[:, numerical_covariates])
    x_val[:, numerical_covariates] = x_scaler.transform(x_val[:, numerical_covariates])
    x_test[:, numerical_covariates] = x_scaler.transform(x_test[:, numerical_covariates])

    # 5. Standardize Targets (Y)
    y_train[~y_mask_train.astype(bool)] = np.nan
    y_val[~y_mask_val.astype(bool)] = np.nan
    y_test[~y_mask_test.astype(bool)] = np.nan

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    y_train_init = y_scaler.transform(y_train_init)
    y_val_init = y_scaler.transform(y_val_init)
    y_test_init = y_scaler.transform(y_test_init)

    # Fill NaNs back to 0 so the PyTorch dataloader doesn't break
    y_train[np.isnan(y_train)] = 0
    y_val[np.isnan(y_val)] = 0
    y_test[np.isnan(y_test)] = 0
    y_train_init = np.nan_to_num(y_train_init)
    y_val_init = np.nan_to_num(y_val_init)
    y_test_init = np.nan_to_num(y_test_init)

    # 6. Save Arrays to CSV/NPY
    fmt_x = ['%d', '%.5f', '%.5f', '%d'] # ID(int), Time(float), StaticReal(float), StaticCat(int)
    fmt_y = '%.5f'
    fmt_mask = '%d'

    def save_split(split_name, x_arr, y_arr, mask_arr, init_arr):
        np.savetxt(os.path.join(destination, f'{split_name}/label.csv'), x_arr, delimiter=',', fmt=fmt_x)
        np.savetxt(os.path.join(destination, f'{split_name}/data.csv'), y_arr, delimiter=',', fmt=fmt_y)
        np.savetxt(os.path.join(destination, f'{split_name}/mask.csv'), mask_arr, delimiter=',', fmt=fmt_mask)
        np.save(os.path.join(destination, f'{split_name}/init_data.csv.npy'), init_arr) # Required for DGBFGP's AVI

    save_split('train', x_train, y_train, y_mask_train, y_train_init)
    save_split('val', x_val, y_val, y_mask_val, y_val_init)
    save_split('test', x_test, y_test, y_mask_test, y_test_init)

    print("Data successfully generated and saved to:", destination)