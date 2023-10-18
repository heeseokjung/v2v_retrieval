import numpy as np
import torch
import torch.nn.functional as F


def cosine_mean_similarity(x, y):
    # x: n x d
    # y: m x d
    # assume that x, y are not normalized
    
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    return torch.mm(x, y.t()).mean()


def smooth_chamfer_similarity(x, y, alpha):
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


def DTW(p, q, sim):
    D = np.zeros((len(p) + 1, len(q) + 1))

    for i in range(1, len(p)+1):
        for j in range(1, len(q)+1):
            s = sim[p[i-1], q[j-1]]
            D[i,j] = max(D[i-1,j], D[i,j-1], D[i-1,j-1] + s)

    path, count = backtrack(p, q, sim, D)

    return D, path, count


def backtrack(p, q, sim, D):
    path, count = [], 0
    i, j = len(p), len(q)

    while i >= 1 and j >= 1:
        path.append((i-1, j-1))
        s = sim[p[i-1], q[j-1]]
        if D[i,j] == D[i-1,j]:
            i = i -1
        elif D[i,j] == D[i,j-1]:
            j = j - 1
        elif D[i,j] == D[i-1,j-1] + s:
            i, j = i - 1, j - 1
            count += 1
        else:
            raise ValueError
        
    return path[::-1], count