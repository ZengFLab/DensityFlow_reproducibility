import torch
import numpy as np

import torch
from torchmetrics.functional import spearman_corrcoef, r2_score, pearson_corrcoef
    
def mmd_eval(X, Y, kernel='rbf', sigma=1.0, device='cuda'):
    """
    PyTorch 加速的无偏MMD计算
    :param X: numpy数组或torch张量 (m x d)
    :param Y: numpy数组或torch张量 (n x d)
    :param kernel: 核函数 ('rbf' 或 'linear')
    :param sigma: 高斯核带宽
    :param device: 'cuda' 或 'cpu'
    :return: MMD²值
    """
    # 转换为PyTorch张量并移到GPU
    X = torch.as_tensor(X, device=device)
    Y = torch.as_tensor(Y, device=device)
    m, n = len(X), len(Y)

    # 计算核矩阵
    def rbf_kernel(x, y):
        pairwise_dist = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)) ** 2
        return torch.exp(-pairwise_dist / (2 * sigma ** 2)).squeeze(0)

    kernel_func = rbf_kernel if kernel == 'rbf' else lambda x, y: x @ y.T

    # 计算三项
    K_XX = kernel_func(X, X)
    K_YY = kernel_func(Y, Y)
    K_XY = kernel_func(X, Y)

    # 无偏估计（排除对角线）
    XX_term = (K_XX.sum() - K_XX.trace()) / (m * (m - 1))
    YY_term = (K_YY.sum() - K_YY.trace()) / (n * (n - 1))
    XY_term = K_XY.mean()

    mmd_squared = XX_term + YY_term - 2 * XY_term
    return max(0.0, mmd_squared.item())  # 确保非负


def r2_score_eval(X, Y, device='cuda'):
    X = torch.as_tensor(X, device=device)
    Y = torch.as_tensor(Y, device=device)
    return r2_score(X.mean(0), Y.mean(0)).item() 


def spearman_eval(delat_X, delta_Y, device='cuda'):
    delat_X = torch.as_tensor(delat_X, device=device)
    delta_Y = torch.as_tensor(delta_Y, device=device)
    return spearman_corrcoef(delat_X.mean(0), delta_Y.mean(0)).item()


def pearson_eval(delat_X, delta_Y, device='cuda'):
    delat_X = torch.as_tensor(delat_X, device=device)
    delta_Y = torch.as_tensor(delta_Y, device=device)
    return torch.nan_to_num(pearson_corrcoef(delat_X.mean(0), delta_Y.mean(0)), nan=0.0).item()

