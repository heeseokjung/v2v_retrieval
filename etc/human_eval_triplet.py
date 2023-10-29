import os
import re
import random
import json
import pickle
import h5py
import numpy as np
import faiss
import torch
import torch.nn.functional as F
import pandas as pd

from dtw import *
from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


# GLOBAL VARIABLE (TRIPLET IDX)
triplet_idx = 0


# numpy version
def compute_dtw_similarity(x, y):
    nx = x / np.linalg.norm(x, axis=-1, keepdims=True)
    ny = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.matmul(nx, ny.T)

    m, n = z.shape[0], z.shape[1]
    R = np.ones((m+1, n+1)) * -1000000
    R[0, 0] = 0.

    for i in range(1, m+1):
        for j in range(1, n+1):
            r0 = R[i-1, j-1] + 2*z[i-1, j-1]
            r1 = R[i-1, j] + z[i-1, j-1]
            r2 = R[i, j-1] + z[i-1, j-1]
            R[i, j] = max(r0, r1, r2) 

    return R[m, n] / (m + n)


# sliding window version
def compute_dtw_sliding_window(x, y, w):
    if x.shape[0] <= w:
        xv = np.expand_dims(x, axis=0) # k x m x d
    else:
        xv = sliding_window_view(x, (w, x.shape[-1])).squeeze() # k x m x d
    if y.shape[0] <= w:
        yv = np.expand_dims(y, axis=0) # k x n x d
    else:
        yv = sliding_window_view(y, (w, y.shape[-1])).squeeze() # k x n x d

    scores = np.zeros((xv.shape[0], yv.shape[0]))
    for i, p in enumerate(xv):
        for j, q in enumerate(yv):
            scores[i][j] = compute_dtw_similarity(p, q)

    return (scores.max(axis=0).mean() + scores.max(axis=1).mean()) / 2


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def preprocess_moma():
    moma = MOMA("/data/dir_moma")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    vid2seqembs = {}
    ids_act = moma.get_ids_act()
    
    for act in tqdm(moma.get_anns_act(ids_act=ids_act), desc="[MOMA] preprocessing"):
        captions = []
        for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
            captions.append(sact.cname)
        with torch.no_grad():
            seq_emb = sbert.encode(captions)
        vid2seqembs[act.id] = seq_emb

    return vid2seqembs


def moma_triplet_generator(
        triplet_list,
        feat_type="s3d", 
        n_sample_per_class=25, 
        topk=20,
        threshold=0.1,
    ):
    root_path = "/data/dir_moma"
    feat_path = os.path.join(root_path, "feats", feat_type)

    moma = MOMA("/data/dir_moma")
    act_taxonomy = moma.taxonomy["act"]
    # vid2seqembs = preprocess_moma()
    id2cemb = torch.load("/root/cvpr24_video_retrieval/anno/moma/id2cemb.pt")

    for cname in tqdm(act_taxonomy, desc="[MOMA] generate triplets"):
        ids_act = moma.get_ids_act(cnames_act=cname)
        n_sact = []
        for act in moma.get_anns_act(ids_act=ids_act):
            n_sact.append(len(act.ids_sact))
        n_sact = np.array(n_sact)
        sorted_idx = np.argsort(n_sact)

        count = 0
        for idx in sorted_idx:
            if count == n_sample_per_class:
                break
            if n_sact[idx] < 3:
                continue
            
            qvid = ids_act[idx]
            candidates = [vid for vid in ids_act if vid != qvid]

            query_video = np.load(os.path.join(feat_path, f"{qvid}.npy")).mean(axis=0)
            while True:
                vid1, vid2 = random.sample(candidates, 2)
                video1 = np.load(os.path.join(feat_path, f"{vid1}.npy")).mean(axis=0)
                video2 = np.load(os.path.join(feat_path, f"{vid2}.npy")).mean(axis=0)
                sim1 = cosine_similarity(query_video, video1)
                sim2 = cosine_similarity(query_video, video2)
                rel1 = compute_dtw_sliding_window(id2cemb[qvid], id2cemb[vid1], w=3)
                rel2 = compute_dtw_sliding_window(id2cemb[qvid], id2cemb[vid2], w=3)

                if sim1 < sim2 and rel1 > rel2:
                    break
                elif sim1 > sim2 and rel1 < rel2:
                    break

            global triplet_idx
            triplet = {
                "tid": triplet_idx,
                "dataset": "moma-lrg",
                "qvid": qvid,
                "vid1": vid1,
                "vid2": vid2,
                "rel1": rel1,
                "rel2": rel2,
                "s1": 0,
                "s2": 0,
                "s3": 0,
                "s4": 0,
            }

            triplet_list.append(triplet)
            triplet_idx += 1
            count += 1

    return triplet_list


def preprocess_activitynet(anno_path, feat_path, feat_type):
    if feat_type == "imagenet":
        video_ids, vid2seqembs = [], {}
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

        missing = 0
        if os.path.exists("ac_video_ids.pt") and os.path.exists("ac_vid2seqembs.pt"):
            video_ids = torch.load("ac_video_ids.pt")
            vid2seqembs = torch.load("ac_vid2seqembs.pt")
        else:
            for anno_file in ["train.json", "val_1.json"]:
                with open(os.path.join(anno_path, anno_file), "r") as f:
                    anno = json.load(f)
                    for vid, data in tqdm(anno.items(), desc=f"[ActivityNet] preprocessing ({anno_file[:-5]})"):
                        if not os.path.exists(os.path.join(feat_path, f"{vid}.npy")):
                            missing += 1
                            continue
                        with torch.no_grad():
                            caption_embs = sbert.encode(data["sentences"]) # n x d
                        vid2seqembs[vid] = caption_embs
                        video_ids.append(vid)
            torch.save(video_ids, "ac_video_ids.pt")
            torch.save(vid2seqembs, "ac_vid2seqembs.pt")

        print(f"mising: {missing} avail: {len(video_ids)}")
        
        feats = [np.load(os.path.join(feat_path, f"{vid}.npy")) for vid in video_ids]
        feats = np.stack(feats, axis=0)

        return video_ids, vid2seqembs, feats
    else:
        raise NotImplementedError


def activitynet_triplet_generator(
        triplet_list,
        feat_type="imagenet",
        n_sample=500,
        topk=50,
        threshold=0.1,
    ):
    root_path = "/data/dir_activitynet"
    anno_path = os.path.join(root_path, "anno")
    feat_path = os.path.join(root_path, "feats", feat_type)

    video_ids, vid2seqembs, feats = preprocess_activitynet(anno_path, feat_path, feat_type)

    faiss.normalize_L2(feats)
    index = faiss.IndexFlatIP(feats.shape[-1])
    index.add(feats)

    sampled_idx = random.sample(range(len(video_ids)), n_sample)
    distance, knn_idx = index.search(feats[sampled_idx,:], topk)

    for i, qidx in enumerate(tqdm(sampled_idx, desc="[ActivityNet] generate triplets")):
        while True:
            j, k = random.sample(range(1, topk), 2)
            qvid = video_ids[qidx]
            vid1 = video_ids[knn_idx[i][j]]
            vid2 = video_ids[knn_idx[i][k]]

            query_video = np.load(os.path.join(feat_path, f"{qvid}.npy"))
            video1 = np.load(os.path.join(feat_path, f"{vid1}.npy"))
            video2 = np.load(os.path.join(feat_path, f"{vid2}.npy"))
            sim1 = cosine_similarity(query_video, video1)
            sim2 = cosine_similarity(query_video, video2)
            rel1 = compute_dtw_similarity(vid2seqembs[qvid], vid2seqembs[vid1])
            rel2 = compute_dtw_similarity(vid2seqembs[qvid], vid2seqembs[vid2])

            if sim1 < sim2 and rel1 > rel2 and abs(rel1- rel2) >= threshold:
                break
            elif sim1 > sim2 and rel1 < rel2 and abs(rel1 - rel2) >= threshold:
                break

        global triplet_idx
        triplet = {
            "tid": triplet_idx,
            "dataset": "activitynet-captions",
            "qvid": qvid,
            "vid1": vid1,
            "vid2": vid2,
            "rel1": rel1,
            "rel2": rel2,
            "s1": 0,
            "s2": 0,
            "s3": 0,
            "s4": 0,
        }

        triplet_list.append(triplet)
        triplet_idx += 1
    
    return triplet_list


def preprocess_howto100m(anno_path):
    anno = pd.read_csv(os.path.join(anno_path, "HowTo100M_v1.csv"))
    with open(os.path.join(anno_path, "HowToCaption.pickle"), "rb") as f:
        caption_data = pickle.load(f)

    return anno, caption_data


def howto100m_triplet_generator(
        triplet_list,
        n_sample_per_class=300,
        topk=5000,
    ):
    root_path = "/data/dir_howto100m"
    anno_path = os.path.join(root_path, "anno")
    feat_path = os.path.join(root_path, "feats", "s3d", "howto100m_s3d_features")

    anno, caption_data = preprocess_howto100m(anno_path)
    topk_class = [
        "Food and Entertaining", 
        "Home and Garden", 
        "Hobbies and Crafts", 
        "Cars & Other Vehicles", 
        "Pets and Animals",
    ]

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    for cname in topk_class:
        data = anno[anno["category_1"] == cname]
        video_ids = []
        for vid in data["video_id"]:
            if os.path.exists(os.path.join(feat_path, f"{vid}.npy")) and vid in caption_data:
                video_ids.append(vid)

        if not os.path.exists(os.path.join(root_path, "feats", f"{cname}.npy")):
            class_feats = []
            for vid in tqdm(video_ids, desc="[HowTo100M] load pre-trained features"):
                class_feats.append(np.load(os.path.join(feat_path, f"{vid}.npy")).mean(axis=0))
            class_feats = np.stack(class_feats, axis=0).astype(np.float32)
            np.save(os.path.join(root_path, "feats", f"{cname}.npy"), class_feats)
        else:
            class_feats = np.load(os.path.join(root_path, "feats", f"{cname}.npy"))

        faiss.normalize_L2(class_feats)
        index = faiss.IndexFlatIP(class_feats.shape[-1])
        index.add(class_feats)

        sampled_idx = random.sample(range(len(video_ids)), n_sample_per_class)
        distance, knn_idx = index.search(class_feats[sampled_idx,:], topk)
        
        threshold = 0.05
        for i, qidx in enumerate(tqdm(sampled_idx, desc=f"[HowTo100M] generate triplets ({cname})")):
            iter_count = 0
            while True:
                j, k = random.sample(range(1, topk), 2)
                with torch.no_grad():
                    query_video = sbert.encode(sum(caption_data[video_ids[qidx]]["text"], []))
                    video1 = sbert.encode(sum(caption_data[video_ids[knn_idx[i][j]]]["text"], []))
                    video2 = sbert.encode(sum(caption_data[video_ids[knn_idx[i][k]]]["text"], []))
                # print(f"query: {query_video.shape} video1: {video1.shape} video2: {video2.shape}")
                rel1 = compute_dtw_similarity(query_video, video1, eps=0, w=-1)
                rel2 = compute_dtw_similarity(query_video, video2, eps=0, w=-1)
                # print(f"rel1 ({video_ids[knn_idx[i][j]]}): {rel1} rel2 ({video_ids[knn_idx[i][k]]}): {rel2} diff: {abs(rel1-rel2)}")
                if abs(rel1 - rel2) >= threshold:
                    break
                
                iter_count += 1
                if iter_count > 100:
                    threshold -= 0.01

            global triplet_idx
            triplet = {
                "tid": triplet_idx,
                "dataset": "howto100m",
                "qvid": video_ids[qidx],
                "vid1": video_ids[knn_idx[i][j]],
                "vid2": video_ids[knn_idx[i][k]],
                "rel1": rel1,
                "rel2": rel2,
                "s1": 0,
                "s2": 0,
                "s3": 0,
                "s4": 0,
            }

            triplet_list.append(triplet)
            triplet_idx += 1

    return triplet_list


def main():
    random.seed(42)
    np.random.seed(42)

    triplet_list = []
    triplet_list = moma_triplet_generator(triplet_list=triplet_list)
    triplet_list = activitynet_triplet_generator(triplet_list=triplet_list)
    # triplet_list = howto100m_triplet_generator(triplet_list=triplet_list)

    df = pd.DataFrame(triplet_list)
    df.to_csv('human_eval_triplet.csv')


if __name__ == "__main__":
    main()