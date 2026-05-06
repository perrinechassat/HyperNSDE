import numpy as np
import torch
from torch.utils.data import Dataset


def make_list_static_BN(static_types, static_true_miss_mask):

    s_onehot_types = []
    for i in range(static_types.shape[0]):
        for j in range(static_types[i, 1]):
            s_onehot_types.append(static_types[i])
    s_onehot_types = np.array(s_onehot_types)

    s_onehot_missing = []
    for i in range(static_true_miss_mask.shape[1]):
        for j in range(static_types[i, 1]):
            s_onehot_missing.append(static_true_miss_mask[:, i])
    s_onehot_missing = torch.stack(s_onehot_missing)
    s_onehot_missing = torch.transpose(s_onehot_missing, 0, 1)

    return s_onehot_types, s_onehot_missing



class Simulation_Dataset(Dataset):
    def __init__(self, config, data):

        # x, mask ---> Patients, num_visits, num_long_var
        self.x, self.mask, self.T = data[0], data[1], data[2]
        self.static_onehot = data[3]
        self.static_types = data[4]
        self.static_true_miss_mask = data[5]
        self.static_vals = data[6]
        self.params_norm = data[7]

        self.static = config.static_data

        if config.batch_norm_static:
            self.bn_s = True
            data_ = make_list_static_BN(self.static_types,
                self.static_true_miss_mask)

            self.s_onehot_types =  data_[0]
            self.s_onehot_missing = data_[1]
        else:
            self.bn_s = False

        if config.static_data:
            self.var_names_static = ["Var Static "+str(i) for i in range(1,len(self.static_types)+1)]
        self.var_names_long = ["Var Longitudinal "+str(i) for i in range(1,config.n_long_var+1)]


    def __getitem__(self, idx):

        T = self.T
        x, mask = self.x[idx, :], self.mask[idx, :]
        if self.static:
            S_OneHot = self.static_onehot[idx, :]
            S_True_MMask = self.static_true_miss_mask[idx, :]
        else:
            return x, mask
        
        if self.bn_s:
            s_ohm = self.s_onehot_missing[idx, :]
            return T, x, mask, S_OneHot.float(), S_True_MMask, s_ohm
        else:
            return T, x, mask, S_OneHot.float(), S_True_MMask

    def get_onehot_static(self):
        if self.bn_s:
            return self.static_onehot.float(), self.static_true_miss_mask, self.s_onehot_missing
        else:
            return self.static_onehot.float(), self.static_true_miss_mask, None

    def __len__(self):
        return len(self.x)

    def get_T(self):
        return self.T

    def get_x_mask(self):
        return self.x, self.mask

    def get_static(self):
        return self.static_onehot.float(), self.static_types, self.static_true_miss_mask
    
    def get_full_static(self):
        return self.static_onehot.float(), self.static_types, self.static_true_miss_mask, self.static_vals

    def get_static_types(self):
        return self.static_types

    # def get_onehot_static(self):
    #     return self.s_onehot_types, self.s_onehot_missing

    def get_var_names(self):
        return self.var_names_long, self.var_names_static
    
    def get_params_norm(self):
        return self.params_norm
    



class Simulation_Dataset_Static(Dataset):
    def __init__(self, data, batch_norm_static=True):

        self.static_onehot = data[0]
        self.static_types = data[1]
        self.static_true_miss_mask = data[2]
        self.static_vals = data[3]

        if batch_norm_static:
            self.bn_s = True
            data_ = make_list_static_BN(self.static_types, self.static_true_miss_mask)
            self.s_onehot_types =  data_[0]
            self.s_onehot_missing = data_[1]
        else:
            self.bn_s = False
        self.var_names_static = ["Var Static "+str(i) for i in range(1,len(self.static_types)+1)]


    def __getitem__(self, idx):
        S_OneHot = self.static_onehot[idx, :]
        S_True_MMask = self.static_true_miss_mask[idx, :]
        if self.bn_s:
            s_ohm = self.s_onehot_missing[idx, :]
            return S_OneHot.float(), S_True_MMask, s_ohm
        else:
            return S_OneHot.float(), S_True_MMask

    def get_onehot_static(self):
        if self.bn_s:
            return self.static_onehot.float(), self.static_true_miss_mask, self.s_onehot_missing
        else:
            return self.static_onehot.float(), self.static_true_miss_mask, None

    def __len__(self):
        return len(self.static_vals)

    def get_static(self):
        return self.static_onehot.float(), self.static_types, self.static_true_miss_mask
    
    def get_full_static(self):
        return self.static_onehot.float(), self.static_types, self.static_true_miss_mask, self.static_vals

    def get_static_types(self):
        return self.static_types
    
    def get_var_names(self):
        return None, self.var_names_static
    