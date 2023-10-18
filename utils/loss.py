import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mse_loss(pred, surrogate):
    loss = nn.MSELoss(reduction="mean")
    return loss(pred, surrogate)
    