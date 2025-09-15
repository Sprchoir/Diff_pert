import re, os, pickle, joblib
import torch
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
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
        # Consider whether to reduce dimensionality of embeddings
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx,:], self.Y[idx,:], self.embeddings[idx,:]
    
    def Preprocessing(self, configs):
        # Obtain the perturbation labels
        self.Y.obs["perturbation"]= self.Y.obs["condition_name"].apply(self.extract_gene_name)
        # print("Unique perturbations:", adata.obs["perturbation"].unique())
        # Obtain the embeddings for perturbed genes
        pickle_path = configs["data"]["embedding_path"]
        with open(pickle_path, "rb") as f:
            gene_embedding = pickle.load(f)
        perturbs = self.Y.obs["perturbation"].astype(str)
        emb_list = [gene_embedding[g] for g in perturbs]
        self.embeddings = torch.tensor(np.stack(emb_list), dtype=torch.float32)

        # PCA through Scanpy
        self.sc_process()
        self.X = torch.tensor(self.X.copy(), dtype=torch.float32)
        self.Y = torch.tensor(self.Y.copy(), dtype=torch.float32)

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
        Preprocess self.X and self.Y through Scanpy
        - train/val: standard preprocessing + PCA
        - test: X processed as usual, Y uses train PCA model for projection(under consideration)
        """
        # for data in [self.X, self.Y]:
        #     sc.pp.normalize_total(data, target_sum=1e4)
        #     sc.pp.log1p(data)
        #     sc.pp.highly_variable_genes(data, n_top_genes=5000, subset=True, flavor="seurat")
        #     sc.pp.scale(data, max_value=10)
        if self.configs["data"]["pca"]:
            n_comps = self.configs["data"]["num_pc"]      
            pca_model_path = os.path.join(self.configs["dir"]["save_dir"], "pca_Y.pkl")

            if self.split == "train":
                pca_model = PCA(n_components=n_comps, svd_solver="arpack")
                self.X = pca_model.fit_transform(self.X.X)
                self.Y = pca_model.fit_transform(self.Y.X)
                # Save the PCA model for Y
                joblib.dump(pca_model, pca_model_path)
            elif self.split == "val":
                pca_model = PCA(n_components=n_comps, svd_solver="arpack")
                self.X = pca_model.fit_transform(self.X.X)
                self.Y = pca_model.fit_transform(self.Y.X)
            elif self.split == "test":
                save_dir = self.configs["dir"]["pred_dir"]
                np.save(os.path.join(save_dir, "Y_target.npy"), self.Y.X.toarray() if hasattr(self.Y.X, "toarray") else self.Y.X)
                self.X = PCA(n_components=n_comps, svd_solver="arpack").fit_transform(self.X.X)
                self.Y = PCA(n_components=n_comps, svd_solver="arpack").fit_transform(self.Y.X)

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