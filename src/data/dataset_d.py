import torch
from torch.utils.data import Dataset
import numpy as np

class DecoderDataset(Dataset):
    """
    Three parts of data:
    - Y_low_real:  low-rank data, top pcs of real gene expression
    - Y_low_gen:   low-rank data generated from Diffusion model 
    - Y_full:  full-dimension, used as ground truth 
    """
    def __init__(self, configs, Y_low_gen, Y_full, Y_low_real=None, split=None):
        super().__init__()
        # Load data first
        self.n_gen = configs["decoder"]["gen_times"]
        self.split_ratio = configs["data"]["split_ratio"]
        self.Y_low_gen = Y_low_gen
        self.Y_full = Y_full
        self.Y_full = torch.as_tensor(self.Y_full, dtype=torch.float32)
        self.Y_low_gen = torch.as_tensor(self.Y_low_gen, dtype=torch.float32)
        if split != "test":
            self.Y_low_real = Y_low_real
            self.Y_low_real = torch.as_tensor(self.Y_low_real, dtype=torch.float32)
            self.pair()
            self._split(split)
        else:
            self.Y_low = self.Y_low_gen

    def __len__(self):
        return self.Y_low.shape[0]

    def __getitem__(self, idx):
        return self.Y_low[idx,:], self.Y_full[idx,:]
    
    def pair(self):
        # Pair the generated low-dim data with real full-dim data.
        x = self.Y_low_gen.shape[0]
        self.Y_low_real = self.Y_low_real[:x, :]
        self.Y_full = self.Y_full[:x, :]
        self.Y_low = torch.cat([self.Y_low_real, self.Y_low_gen], dim=0)
        self.Y_full = torch.cat([self.Y_full, self.Y_full.repeat((self.n_gen, 1))], dim=0)
    
    def _split(self, split):
        n_total = self.Y_low.shape[0]
        n_train = int(n_total * (self.split_ratio[0] + self.split_ratio[2]))
        n_val = int(n_total * self.split_ratio[1])
        if split == "train":
            self.Y_low, self.Y_full = self.Y_low[:n_train], self.Y_full[:n_train]
        elif split == "val":
            self.Y_low, self.Y_full = self.Y_low[n_train:n_train+n_val], self.Y_full[n_train:n_train+n_val]
