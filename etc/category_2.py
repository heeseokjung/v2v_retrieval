import os
import random
import numpy as np
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer


def compute_dtw_score(x, y, eps, w):
    nx = x / np.linalg.norm(x, axis=-1, keepdims=True)
    ny = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.matmul(nx, ny.T)

    m, n = z.shape[0], z.shape[1]
    R = np.ones((m+1, n+1))
    R[0,:], R[:,0] = -np.inf, -np.inf
    R[0,0] = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            # if abs(i - j) > w:
            #     continue
            r0 = R[i-1, j-1] 
            r1 = R[i-1, j] 
            r2 = R[i, j-1] 
            R[i, j] = max(r0, r1, r2) + z[i-1, j-1]

    # backtracking
    i, j, size = m, n, 0
    while i >= 1 and j >= 1:
        size += 1
        r0 = R[i-1, j-1] + z[i-1, j-1]
        r1 = R[i-1, j] + eps
        r2 = R[i, j-1] + eps
        rmax = max(r0, r1, r2)

        if rmax == r0:
            i, j = i - 1, j - 1
        elif rmax == r1:
            i = i - 1
        elif rmax == r2:
            j = j - 1
        else:
            raise ValueError
        
    # print(f"R[m, n]: {R[m, n]} size: {size} score: {R[m,n] / size}")
        
    return R[m, n] / size


def main():
    anno_path = "/data/dir_howto100m/anno"
    feat_path = "/data/dir_howto100m/feats/s3d/howto100m_s3d_features"
        
    with open(os.path.join(anno_path, "HowToCaption.pickle"), "rb") as f:
        caption = pickle.load(f)
    anno = pd.read_csv(os.path.join(anno_path, "HowTo100M_v1.csv"))
    
    data = anno[anno["task_id"] == 19]
    video_ids_anno = list(data["video_id"])
    video_ids = []
    for vid in video_ids_anno:
        if vid in caption:
            video_ids.append(vid)

    feats = [np.load(os.path.join(feat_path, f"{vid}.npy")).mean(axis=0) for vid in video_ids]
    feats = np.stack(feats, axis=0).astype(np.float32)

    faiss.normalize_L2(feats)
    index = faiss.IndexFlatIP(feats.shape[-1])
    index.add(feats)

    distance, knn_idx = index.search(feats, 10)
    
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    for q in range(feats.shape[0]):
        while True:
            j, k = random.sample(range(1, 10), 2)
            query_video = sbert.encode(sum(caption[video_ids[q]]["text"], []))
            video1 = sbert.encode(sum(caption[video_ids[knn_idx[q][j]]]["text"], []))
            video2 = sbert.encode(sum(caption[video_ids[knn_idx[q][k]]]["text"], []))

            rel1 = compute_dtw_score(query_video, video1, 0, 0)
            rel2 = compute_dtw_score(query_video, video2, 0, 0)

            if abs(rel1 - rel2) > 0.05:
                break

        print(f"query: {video_ids[q]} video1: {video_ids[knn_idx[q][j]]} video2: {video_ids[knn_idx[q][k]]} rel1: {rel1} rel2: {rel2} dist1: {distance[q][j]} dist2: {distance[q][k]} diff: {abs(rel1-rel2)}")


if __name__ == "__main__":
    main()