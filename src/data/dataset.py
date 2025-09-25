import re, os, pickle
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from ..utils.utils import Load_data

class DiffDataset(Dataset):
    def __init__(self, configs, split):
        super(DiffDataset, self).__init__()
        self.configs = configs
        self.split = split
        X, Y = Load_data(configs)
        self.X, self.Y = self._split(X, Y)
        self.Preprocessing(configs)
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx,:], self.Y[idx,:], self.embeddings[idx,:]
    
    def Preprocessing(self, configs):
        # Preprocessing and PCA through Scanpy
        self.sc_process()

        # Obtain the perturbation labels
        self.Y.obs["perturbation"] = self.Y.obs["condition_name"].apply(self.extract_gene_name)
        # print("Unique perturbations:", adata.obs["perturbation"].unique()) # 86 kinds
        pickle_path = configs["data"]["embedding_path"]
        with open(pickle_path, "rb") as f:
            gene_embedding = pickle.load(f)
        perturbs = self.Y.obs["perturbation"].astype(str)
        emb_list = [gene_embedding[g] for g in perturbs]
        self.embeddings = torch.tensor(np.stack(emb_list), dtype=torch.float32)

        if self.configs["data"]["pca"]:
          self.X = self.X.obsm["X_pca"].copy()
          self.Y = self.Y.obsm["X_pca"].copy()
        else:
          self.X = self.X.X.toarray() if hasattr(self.X.X, "toarray") else self.X.X
          self.Y = self.Y.X.toarray() if hasattr(self.Y.X, "toarray") else self.Y.X 
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def extract_gene_name(self, condition_name):
        if "ctrl" in condition_name and "+" in condition_name:
            match = re.search(r"_(.*?)\+ctrl", condition_name)
            if match:
                return match.group(1)
        elif condition_name.endswith("_ctrl_1"): 
            return "ctrl"
        return condition_name 
    
    def sc_process(self):
        """
        Preprocess self.X and self.Y through Scanpy:
        Normalization + Log-transform + HVG filter + PCA
        To better capture the differences between different cell types, we fit PCA independently on train/val/test set.
        And a semi-supervised decoder is used to map the PCA embeddings back to the original space.
        """
        save_dir = self.configs["dir"]["pred_dir"]
        hvg_file = os.path.join(save_dir, "hvg_genes.npy")
        n_comps = self.configs["data"]["num_pc"]
        norm_sum = self.configs["data"]["norm_sum"]
        # Scanpy Preprocessing for training set
        for data in [self.X, self.Y]:
            sc.pp.normalize_total(data, target_sum=norm_sum)
            sc.pp.log1p(data)
        # HVG selection 
        sc.pp.highly_variable_genes(self.X, n_top_genes=self.configs["data"]["num_HVG"], subset=True, flavor="seurat")
        hvg_genes = self.X.var_names[self.X.var["highly_variable"]].to_numpy()
        np.save(hvg_file, hvg_genes)
        self.Y = self.Y[:, hvg_genes]
        # Save raw Y for test set
        if self.split == "test":
            np.save(
                os.path.join(save_dir, "Y_target_full.npy"),
                self.Y.X.toarray() if hasattr(self.Y.X, "toarray") else self.Y.X
            )
        # PCA
        if self.configs["data"]["pca"]:
            sc.tl.pca(self.X, n_comps=n_comps, use_highly_variable=True, svd_solver="arpack")
            sc.tl.pca(self.Y, n_comps=n_comps, svd_solver="arpack")
        # Save top_pc(Y) for test set
        if self.split == "test":
            if self.configs["data"]["pca"]:
              y = self.Y.obsm["X_pca"]
            else:
              y = self.Y.X.toarray() if hasattr(self.Y.X, "toarray") else self.Y.X
            np.save(
                os.path.join(save_dir, "Y_target_pc.npy"),
                y
            )              

    def _split(self, X, Y):
        split_ratio = self.configs["data"].get("split", [0.8, 0.1, 0.1])
        n_total = len(Y)
        n_train = int(n_total * split_ratio[0])
        n_val   = int(n_total * split_ratio[1])

        if self.split == "train":
            return X[:n_train], Y[:n_train]
        elif self.split == "val":
            return X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
        else:
            return X[n_train+n_val:], Y[n_train+n_val:]
        
if __name__ == '__main__':
    pass