import torch
import torch.nn.functional as F
from tslearn.metrics import SoftDTWLossPyTorch


def f(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1 - torch.bmm(x, y.transpose(-2, -1))


def main():
    loss = SoftDTWLossPyTorch(gamma=0.1, normalize=True, dist_func=f)
    x = torch.randn(1, 8, 384)
    # y = torch.randn(1, 8, 384)
    y = x[:]

    # x = torch.ones(1, 8, 384)
    # y = -1.*torch.ones(1, 8, 384)

    print(1 - (loss(x, y) / (8 + 8)))


if __name__ == "__main__":
    main()