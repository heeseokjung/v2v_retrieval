import pandas as pd
import json
import numpy as np


def main():
    # df = pd.read_csv("human_eval_triplet.csv")
    # print(df.shape)

    # qvid_list = list(df["qvid"])
    # vid1_list = list(df["vid1"])
    # vid2_list = list(df["vid2"])

    # video_ids = list(set(qvid_list + vid1_list + vid2_list))
    # print(len(video_ids))
    
    # with open('howto100m_vid_required.json', 'w') as f:
    #     json.dump(video_ids, f)

    q = np.load("/data/dir_howto100m/feats/s3d/howto100m_s3d_features/tqRzVvUF5Mo.npy").mean(axis=0)
    v1 = np.load("/data/dir_howto100m/feats/s3d/howto100m_s3d_features/6KMPQGQPbN0.npy").mean(axis=0)
    v2 = np.load("/data/dir_howto100m/feats/s3d/howto100m_s3d_features/Tg3IU0CZolc.npy").mean(axis=0)

    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

    print(np.dot(q.T, v1))
    print(np.dot(q.T, v2))


if __name__ == "__main__":
    main()