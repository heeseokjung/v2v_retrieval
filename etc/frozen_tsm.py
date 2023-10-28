import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def main():
    path = "/data/dir_moma/feats/frozen"
    for filename in tqdm(os.listdir(path)):
        x = np.load(os.path.join(path, filename))
        x = torch.from_numpy(x).float()

        x = x[:, 1:, :] # remove CLS token
        n_frames = 4
        n_clips, _, d = x.shape
        x = x.reshape(n_clips * n_frames, -1, d)
        x = x.mean(dim=1)

        x = F.normalize(x, dim=-1) # n_clips * 4 x d
        s = torch.mm(x, x.t())
        s = s.numpy()

        sns.heatmap(s)
        plt.savefig(f"tsm/{filename[:-4]}.png")
        plt.close()


if __name__ == "__main__":
    main()