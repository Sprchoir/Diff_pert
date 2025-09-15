import torch
import scanpy as sc
import joblib
import numpy as np

def Load_data(configs):
    adata = sc.read_h5ad(configs["data"]["rna_path"])
    # Unperturbed and perturbed cells
    control_cells = adata[adata.obs["condition"] == "ctrl"]
    perturbed_cells = adata[adata.obs["condition"] != "ctrl"]

    # Pairing: for each perturbed cell, randomly select a control cell as baseline
    seed = configs["training"]["seed"]
    np.random.seed(seed)
    idx = np.random.choice(control_cells.shape[0], size=perturbed_cells.shape[0], replace=True)
    X = control_cells[idx]
    Y = perturbed_cells

    # Shuffle the data
    n_total = Y.shape[0]
    shuffle_idx = np.random.permutation(n_total)
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]

    return X, Y

def save_checkpoint(model, path, epoch, optimizer=None, val_loss=None):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if val_loss is not None:
        state["val_loss"] = val_loss
    torch.save(state, path)

def load_checkpoint(model, path, optimizer=None, device=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", 0)
    return model, optimizer, epoch, val_loss

def pca_inverse(data, path):
    """ Reconstruct Perturbed data from the top principal components """
    pca_model = joblib.load(path)
    Y = pca_model.inverse_transform(data)
    return Y
