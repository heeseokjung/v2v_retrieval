import os
from tqdm import tqdm


def main():
    path = "/data/dir_howto100m/feats/s3d/howto100m_s3d_features"
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".mp4.npy") or filename.endswith(".webm.npy"):
            new_filename = filename.replace(".mp4.npy", ".npy").replace(".webm.npy", ".npy")
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

if __name__ == "__main__":
    main()
    