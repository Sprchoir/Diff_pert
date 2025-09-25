import torch
import scanpy as sc
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

def MMD(x, y):
    """
    Empirical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
    Args:
        x: one sample, distribution P
        y: another sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx 
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz 
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    # Use median trick to get a base bandwidth
    base = torch.median(dxy.detach())
    if base.item() <= 0:
        base = torch.tensor(1.0, device=device)  # fallback
    # Create multiple bandwidths around the median
    scales = [0.5, 1, 2, 4, 8]  # you can tune this list
    bandwidths = [base * s for s in scales]
    # RBF (Gaussian) kernel
    # Bandwidths control the "width" of the Gaussian, i.e., how sensitive the kernel is to distances.
    for a in bandwidths:
        XX += torch.exp(-0.5*dxx/a)
        YY += torch.exp(-0.5*dyy/a)
        XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)
