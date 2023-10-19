import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mse_loss(pred, surrogate):
    loss = nn.MSELoss(reduction="mean")
    return loss(pred, surrogate)
    
def compute_var_hinge_loss(var):
    eps, gamma = 1e-4, 0.3
    var = torch.sqrt(var + eps)
    return F.relu(gamma - var).mean()