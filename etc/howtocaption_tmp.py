import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# def compute_dtw_score(x, y, eps, w):
#     nx = x / np.linalg.norm(x, axis=-1, keepdims=True)
#     ny = y / np.linalg.norm(y, axis=-1, keepdims=True)
#     z = np.matmul(nx, ny.T)

#     m, n = z.shape[0], z.shape[1]
#     R = np.ones((m+1, n+1))
#     R[0,:], R[:,0] = -np.inf, -np.inf
#     R[0,0] = 0

#     for i in range(1, m+1):
#         for j in range(1, n+1):
#             # if abs(i - j) > w:
#             #     continue
#             r0 = R[i-1, j-1] + z[i-1, j-1]
#             r1 = R[i-1, j] + eps
#             r2 = R[i, j-1] + eps
#             R[i, j] = max(r0, r1, r2)

#     # backtracking
#     i, j, size = m, n, 0
#     while i >= 1 and j >= 1:
#         size += 1
#         r0 = R[i-1, j-1] + z[i-1, j-1]
#         r1 = R[i-1, j] + eps
#         r2 = R[i, j-1] + eps
#         rmax = max(r0, r1, r2)

#         if rmax == r0:
#             i, j = i - 1, j - 1
#         elif rmax == r1:
#             i = i - 1
#         elif rmax == r2:
#             j = j - 1
#         else:
#             raise ValueError
        
#     # print(f"R[m, n]: {R[m, n]} size: {size} score: {R[m,n] / size}")
        
#     return R[m, n] / size


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
            r0 = R[i-1, j-1] + z[i-1, j-1]
            r1 = R[i-1, j] + z[i-1, j-1]
            r2 = R[i, j-1] + z[i-1, j-1]
            R[i, j] = max(r0, r1, r2)

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
    with open(os.path.join(anno_path, "HowToCaption.pickle"), "rb") as f:
        data = pickle.load(f)

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    qvid = "F8IlItoK0o0"
    vid1 = "9MMErTJeGKM"
    vid2 = "wH9isdA5K0o"

    qvid_caption = []
    for caption_list in data[qvid]["text"]:
        for caption in caption_list:
            qvid_caption.append(caption)
    vid1_caption = []
    for caption_list in data[vid1]["text"]:
        for caption in caption_list:
            vid1_caption.append(caption)
    vid2_caption = []
    for caption_list in data[vid2]["text"]:
        for caption in caption_list:
            vid2_caption.append(caption)

    print(f"qvideo: {qvid_caption}")
    print(f"video1: {vid1_caption}")
    print(f"video2: {vid2_caption}")

    with torch.no_grad():
        qvideo = sbert.encode(qvid_caption)
        video1 = sbert.encode(vid1_caption)
        video2 = sbert.encode(vid2_caption)
        # print(f"q: {qvideo.shape} v1: {video1.shape} v2: {video2.shape}")

    dtw_rel1 = compute_dtw_score(qvideo, video1, eps=0, w=-1)
    dtw_rel2 = compute_dtw_score(qvideo, video2, eps=0, w=-1)

    print(f"dtw_rel1: {dtw_rel1} dtw_rel2: {dtw_rel2}")

    qvideo = qvideo / np.linalg.norm(qvideo, axis=-1, keepdims=True)
    video1 = video1 / np.linalg.norm(video1, axis=-1, keepdims=True)
    video2 = video2 / np.linalg.norm(video2, axis=-1, keepdims=True)

    cos_rel1 = np.matmul(qvideo, video1.T).mean()
    cos_rel2 = np.matmul(qvideo, video2.T).mean()

    print(f"cos_rel1: {cos_rel1} cos_rel2: {cos_rel2}")

if __name__ == "__main__":
    main()