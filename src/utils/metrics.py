import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss

def MMD_energy(x, y):
    loss = SamplesLoss("energy", p=2)
    return loss(x, y)

def MMD_Gaussian(x, y):
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
    # 1. Subtract control mean
    control_mean = np.mean(X, axis=0)
    Y_true_adj = Y_true - control_mean
    Y_pred_adj = Y_pred - control_mean
    # 2. Pearson R² for each gene
    r2_list = []
    for i in range(Y_true_adj.shape[1]):
        y_t = Y_true_adj[:, i]
        y_p = Y_pred_adj[:, i]
        if np.std(y_t) == 0 or np.std(y_p) == 0:
            continue
        r = np.corrcoef(y_t, y_p)[0, 1]
        if np.isnan(r):
            continue
        r2_list.append(r ** 2)
    # 3. Take average
    mean_r2 = np.mean(r2_list) if len(r2_list) > 0 else 0.0
    return mean_r2

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

def decoder_loss(y_pred, y_true):
    """
    loss_overall: overall MSE
    loss_bi_axis: MSE loss across both cells and genes
    """
    # axis 1: mse loss across genes
    mse_across_genes = torch.mean(torch.mean((y_true - y_pred) ** 2, dim=0))
    # axis 2: mse loss across cells
    mse_across_cells = torch.mean(torch.mean((y_true - y_pred) ** 2, dim=1))
    loss_bi_axis = mse_across_cells + mse_across_genes

    # overall MSE
    loss_overall = nn.functional.mse_loss(y_true, y_pred)

    # total loss
    loss = loss_bi_axis + loss_overall
    return loss
