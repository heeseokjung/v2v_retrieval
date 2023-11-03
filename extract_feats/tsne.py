import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    xs = []
    path = "/data/dir_moma/preprocessed/backup/s3d"
    for filename in tqdm(os.listdir(path)):
        x = np.load(os.path.join(path, filename))
        xs.append(x)
    xs = np.stack(xs, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    xs = tsne.fit_transform(xs)

    plt.scatter(xs[:,0], xs[:,1])
    plt.savefig("backup_tsne.png")
    plt.close()

if __name__ == "__main__":
    main()