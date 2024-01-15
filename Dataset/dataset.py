from utils import open_pkl, save_pkl
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch
import os
import numpy as np
from torch.nn import AvgPool3d


class KIDataset(Dataset):
    def __init__(self,
                 data_path,
                 coordinate_data_path,
                 n,
                 length=None,
                 return_all=False,
                 forecast=False,
                 validation=False,
                 return_t=False,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.coordinate_data_path = coordinate_data_path
        self.return_all = return_all
        self.forecast = forecast
        self.validation = validation
        self.return_t = return_t
        f = os.listdir(self.data_path)
        self.filenames = []
        self.n = n
        if length is None:
            self.filenames += f
        else:
            while length > len(self.filenames):
                self.filenames += f
            self.filenames = self.filenames[:length]
        self.nitems = len(self.filenames)
        if self.validation:
            np.random.seed(0)
            self.seeds = np.random.randint(0, 1000000, self.nitems)
            if self.return_t:
                self.t_lst = np.random.randint(0, 1000, self.nitems)
        self.norm_method = kwargs['norm_method'] if 'norm_method' in kwargs else 'rescaling'
        if self.norm_method == 'normalization':
            self.a, self.b = kwargs['mean'], kwargs['std']
        elif self.norm_method == 'rescaling':
            self.a, self.b = kwargs['min'], kwargs['max']

    def to_tensor(self, x):
        return torch.FloatTensor(x)
    
    def __getitem__(self, idx):
        item_idx = self.filenames[idx]
        if self.validation:
            seed = self.seeds[idx]
            np.random.seed(seed)
            if self.return_t:
                t = int(self.t_lst[idx])
        coord_idx = int(item_idx.split('_')[1].split('.')[0])
        item_dict = open_pkl(self.data_path + item_idx)

        starting_idx = np.random.choice(item_dict['starting_idx'], 1, replace=False)[0]
        # if self.validation:
        #     print(idx, starting_idx)
        seq = np.array(item_dict['ki_maps'])[starting_idx:starting_idx + self.n]
        seq = seq.reshape(1, *seq.shape)

        if self.norm_method == 'normalization':
            seq = (seq - self.a) / self.b
        elif self.norm_method == 'rescaling':
            seq = 2 * ((seq - self.a) / (self.b - self.a)) - 1

        if self.return_all:
            lon = np.array(open_pkl(self.coordinate_data_path + '{}_lon.pkl'.format(coord_idx)))
            lat = np.array(open_pkl(self.coordinate_data_path + '{}_lat.pkl'.format(coord_idx)))
            alt = np.array(open_pkl(self.coordinate_data_path + '{}_alt.pkl'.format(coord_idx)))
            lon = 2 * ((lon - 0) / (90 - 0)) - 1
            lat = 2 * ((lat - 0) / (90 - 0)) - 1
            alt = 2 * ((alt - (-13)) / (4294 - 0)) - 1
            lon = lon.reshape(1, 1, *lon.shape)
            lat = lat.reshape(1, 1, *lat.shape)
            alt = alt.reshape(1, 1, *alt.shape)
            c = np.concatenate((alt, lon, lat), axis=0)
            if self.forecast:
                return self.to_tensor(seq[:, :4]), self.to_tensor(seq[:, 4:]), self.to_tensor(c)
            
            else:
                return self.to_tensor(seq), self.to_tensor(c)
        else:
            if self.forecast:
                if self.return_t:
                    return self.to_tensor(seq[:, :4]), self.to_tensor(seq[:, 4:]), t 
                else:
                    return self.to_tensor(seq[:, :4]), self.to_tensor(seq[:, 4:])
            else:
                return self.to_tensor(seq)

    def __len__(self):
        return self.nitems
    
class HRKIDataset(Dataset):
    def __init__(self,
                 data_path,
                 coordinate_data_path,
                 n,
                 length=None,
                 return_all=False,
                 forecast=False,
                 validation=False,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.coordinate_data_path = coordinate_data_path
        self.return_all = return_all
        self.forecast = forecast
        self.validation = validation
        f = os.listdir(self.data_path)
        self.filenames = []
        self.n = n
        if length is None:
            self.filenames += f
        else:
            while length > len(self.filenames):
                self.filenames += f
            self.filenames = self.filenames[:length]
        self.nitems = len(self.filenames)
        if self.validation:
            np.random.seed(0)
            self.seeds = np.random.randint(0, 1000000, self.nitems)

        self.norm_method = kwargs['norm_method'] if 'norm_method' in kwargs else 'rescaling'
        if self.norm_method == 'normalization':
            self.a, self.b = kwargs['mean'], kwargs['std']
        elif self.norm_method == 'rescaling':
            self.a, self.b = kwargs['min'], kwargs['max']
        self.pooling = AvgPool3d((1,2,2))

    def to_tensor(self, x):
        return torch.FloatTensor(x)
    
    def __getitem__(self, idx):
        item_idx = self.filenames[idx]
        if self.validation:
            seed = self.seeds[idx]
            np.random.seed(seed)
        coord_idx = int(item_idx.split('_')[1].split('.')[0])
        item_dict = open_pkl(self.data_path + item_idx)

        starting_idx = np.random.choice(item_dict['starting_idx'], 1, replace=False)[0]
        seq = np.array(item_dict['ki_maps'])[starting_idx:starting_idx + self.n]
        
        seq = seq.reshape(1, *seq.shape)

        if self.norm_method == 'normalization':
            seq = (seq - self.a) / self.b
        elif self.norm_method == 'rescaling':
            seq = 2 * ((seq - self.a) / (self.b - self.a)) - 1

        if self.return_all:
            lon = np.array(open_pkl(self.coordinate_data_path + '{}_lon.pkl'.format(coord_idx)))
            lat = np.array(open_pkl(self.coordinate_data_path + '{}_lat.pkl'.format(coord_idx)))
            alt = np.array(open_pkl(self.coordinate_data_path + '{}_alt.pkl'.format(coord_idx)))
            lon = 2 * ((lon - 0) / (90 - 0)) - 1
            lat = 2 * ((lat - 0) / (90 - 0)) - 1
            alt = 2 * ((alt - (-13)) / (4294 - 0)) - 1
            lon = lon.reshape(1, 1, *lon.shape)
            lat = lat.reshape(1, 1, *lat.shape)
            alt = alt.reshape(1, 1, *alt.shape)
            c = np.concatenate((alt, lon, lat), axis=0)
            if self.forecast:
                return self.to_tensor(seq[:, :4]), self.to_tensor(seq[:, 4:]), self.to_tensor(c)
            else:
                return self.to_tensor(seq), self.to_tensor(c)
        else:
            if self.forecast:
                return self.pooling(self.to_tensor(seq[:, :4])), self.to_tensor(seq[:, 4:])
            else:
                return self.to_tensor(seq)

    def __len__(self):
        return self.nitems

