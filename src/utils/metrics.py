import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

def MMD(x, y):
    """
    Empirical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
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

def pearson_r2_after_control(X, Y_true, Y_pred):
    # R^2 after subtracting mean control
    control_mean = np.mean(X, axis=0)
    Y_true_adj = Y_true - control_mean
    Y_pred_adj = Y_pred - control_mean
    
    # Pearson R^2 for each cell and average
    r2_list = []
    for c in range(Y_true.shape[0]): 
        r, _ = pearsonr(Y_true_adj[c, :], Y_pred_adj[c, :])
        if not np.isnan(r):
            r2_list.append(r**2)
    return np.mean(r2_list) if r2_list else np.nan

def mse_all_genes(Y_true, Y_pred):
    # MSE over all genes and samples
    Y_true = torch.tensor(Y_true, dtype=torch.float32)
    Y_pred = torch.tensor(Y_pred, dtype=torch.float32)

    criterion = nn.MSELoss()
    return criterion(Y_pred, Y_true).item()

def mse_top_DE_genes(X, Y_true, Y_pred, top_k=20):
    # MSE for top-k DE genes
    control_mean = np.mean(X, axis=0)
    de_scores = np.abs(np.mean(Y_true, axis=0) - control_mean)
    top_genes = np.argsort(de_scores)[-top_k:]
    
    Y_true_sel = torch.tensor(Y_true[:, top_genes], dtype=torch.float32)
    Y_pred_sel = torch.tensor(Y_pred[:, top_genes], dtype=torch.float32)

    criterion = nn.MSELoss()
    return criterion(Y_pred_sel, Y_true_sel).item()

def decoder_loss(y_pred, y_true, l1_lambda=1e-5, axis_lambda=0.1):
    """
    loss_mse: overall MSE loss
    loss_l1: control the sparsity
    loss_bi_axis: MSE loss across both cells and genes
    """
    loss_mse = nn.functional.mse_loss(y_pred, y_true)
    loss_l1 = y_pred.abs().mean()

    # axis 1: mse loss across genes
    cell_var = (y_pred - y_true).var(dim=1).mean()
    # axis 2: mse loss across cells
    gene_var = (y_pred - y_true).var(dim=0).mean()
    loss_bi_axis = cell_var + gene_var

    # total loss
    loss = loss_mse + l1_lambda * loss_l1 + axis_lambda * loss_bi_axis
    return loss