import numpy as np
import torch
import torch.nn.functional as F


def compute_cosine_mean_similarity(x, y):
    # x: n x d
    # y: m x d
    # assume that x, y are not normalized
    
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    return torch.mm(x, y.t()).mean()


def compute_smooth_chamfer_similarity(x, y, alpha):
    # x: n x d
    # y: m x d
    # assume that x, y are not normalized
    
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    c = torch.mm(x, y.t())
    
    left_term = torch.logsumexp(alpha*c, dim=1).sum() / (2.*alpha*x.shape[0])
    right_term = torch.logsumexp(alpha*c, dim=0).sum() / (2.*alpha*y.shape[0])
    
    return left_term + right_term


def smooth_chamfer_train(x, y, alpha):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    c = torch.mm(x, y.t())

    return torch.logsumexp(alpha*c, dim=1).sum() / (2. * alpha * x.shape[0])


# pytorch version
# def compute_dtw_similarity(x, y):
#     nx = F.normalize(x, dim=-1)
#     ny = F.normalize(y, dim=-1)
#     z = torch.mm(nx, ny.t())

#     m, n = z.shape[0], z.shape[1]
#     R = torch.ones((m+1, n+1)).to(x.device) * -float("inf")
#     R[0,0] = 0.

#     for i in range(1, m+1):
#         for j in range(1, n+1):
#             r0 = R[i-1, j-1] + 2*z[i-1, j-1]
#             r1 = R[i-1, j] + z[i-1, j-1]
#             r2 = R[i, j-1] + z[i-1, j-1]
#             R[i, j] = max(r0, r1, r2) 

#     return R[m, n] / (m + n)


# numpy version
def compute_dtw_similarity(x, y):
    nx = x / np.linalg.norm(x, axis=-1, keepdims=True)
    ny = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.matmul(nx, ny.T)

    m, n = z.shape[0], z.shape[1]
    R = np.ones((m+1, n+1)) * -np.inf
    R[0, 0] = 0.

    for i in range(1, m+1):
        for j in range(1, n+1):
            r0 = R[i-1, j-1] + 2*z[i-1, j-1]
            r1 = R[i-1, j] + z[i-1, j-1]
            r2 = R[i, j-1] + z[i-1, j-1]
            R[i, j] = max(r0, r1, r2) 

    return R[m, n] / (m + n)