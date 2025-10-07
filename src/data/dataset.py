import re, os, pickle, joblib
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from ..utils.utils import Load_data

class DiffDataset(Dataset):
    def __init__(self, configs, split):
        self.configs = configs
        self.split = split
        X, Y = Load_data(configs)
        self.X, self.Y = self._split(X, Y)
        self.Preprocessing(configs)
        if split == "test":
          self.save_ori()
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx,:], self.Y[idx,:], self.embeddings[idx,:]
    
    def Preprocessing(self, configs):
        if self.split == "test":
          self.X_ori = self.X.copy()
          self.Y_ori = self.Y.copy()
        
        # Target for the decoder: log1p data
        self.Y_full = self.Y.copy()
        sc.pp.log1p(self.Y_full)
    
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
    
    def save_ori(self):
        X_ori = self.X_ori[:, self.hvg_genes]
        Y_ori = self.Y_ori[:, self.hvg_genes]
        X_ori = X_ori.X.toarray() if hasattr(X_ori.X, "toarray") else X_ori.X
        Y_ori = Y_ori.X.toarray() if hasattr(Y_ori.X, "toarray") else Y_ori.X
        np.savez(
            os.path.join(self.configs["dir"]["pred_dir"], "XY_ori.npz"),
            X_ori=X_ori,
            Y_ori=Y_ori
        )

    def sc_process(self):
        """
        Preprocess self.X and self.Y through Scanpy:
        Normalization + Log-transform + HVG filter + PCA
        To better capture the differences between different cell types, we fit PCA independently on train/val/test set.
        And a semi-supervised decoder will be used to map the PC data embeddings back to the original space.
        """
        save_dir = self.configs["dir"]["pred_dir"]
        stats_file = os.path.join(save_dir, f"stats.pkl")
        n_comps = self.configs["data"]["num_pc"]
        norm_sum = self.configs["data"]["norm_sum"]
        # Scanpy Preprocessing for training set
        self.X.obs["lib_size_raw"] = self.X.X.sum(axis=1).A1 if hasattr(self.X.X, "A1") else self.X.X.sum(axis=1)
        self.Y.obs["lib_size_raw"] = self.Y.X.sum(axis=1).A1 if hasattr(self.Y.X, "A1") else self.Y.X.sum(axis=1)
        for data in [self.X, self.Y]:
            sc.pp.normalize_total(data, target_sum=norm_sum)
            sc.pp.log1p(data)
        # HVG selection 
        # If for different cells or genes, we need to independent select HVG as well.
        # And use the Gene embedding to better represent the cell-gene expression.
        if self.split == "train":
            sc.pp.highly_variable_genes(self.X, n_top_genes=self.configs["data"]["num_HVG"], subset=True, flavor="seurat")
            self.hvg_genes = self.X.var_names[self.X.var["highly_variable"]].to_numpy()
        else:
            self.hvg_genes = joblib.load(stats_file)["hvg_genes"]
        self.X = self.X[:, self.hvg_genes]
        self.Y = self.Y[:, self.hvg_genes]
        self.Y_full = self.Y_full[:, self.hvg_genes]
        self.Y_full = self.Y_full.X.toarray() if hasattr(self.Y_full.X, "toarray") else self.Y_full.X
        # Scale
        # for data in [self.X, self.Y]:
        #     sc.pp.scale(data, max_value=10)
        # PCA
        if self.configs["data"]["pca"]:
            sc.tl.pca(self.X, n_comps=n_comps, svd_solver="arpack")
            sc.tl.pca(self.Y, n_comps=n_comps, svd_solver="arpack")

        # Save the Statistics for the inversion
        stats = {
            "lib_size_raw_X": self.X.obs["lib_size_raw"].to_numpy(),
            "lib_size_raw_Y": self.Y.obs["lib_size_raw"].to_numpy(),
            "mean_X": self.X.var["mean"].to_numpy() if "mean" in self.X.var else None,
            "std_X": self.X.var["std"].to_numpy() if "std" in self.X.var else None,
            "mean_Y": self.Y.var["mean"].to_numpy() if "mean" in self.Y.var else None,
            "std_Y": self.Y.var["std"].to_numpy() if "std" in self.Y.var else None,
            "hvg_genes": self.hvg_genes,
            "norm_sum": norm_sum,
        }
        joblib.dump(stats, stats_file)             

    def _split(self, X, Y):
        split_ratio = self.configs["data"].get("split", [0.8, 0.1, 0.1])
        n_total = len(Y)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        if self.split == "train":
            return X[:n_train], Y[:n_train]
        elif self.split == "val":
            return X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
        elif self.split == "test":
            return X[n_train+n_val:], Y[n_train+n_val:]
        elif self.split == "generate":
            return X[:n_train+n_val], Y[:n_train+n_val]
        
if __name__ == '__main__':
    pass