import os
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dtw import *
from tslearn.metrics import dtw_path_from_metric
from numba import jit
from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def preprocess():
    moma = MOMA(dir_moma="/data/dir_moma/", paradigm="standard")

    vid2seq = {}   # activity id -> sub-activity sequence
    sid2cname = {} # sub-activity id -> sub-activity class name
    for split in ["train", "val", "test"]:
        ids_act = moma.get_ids_act(split=split)
        for act in tqdm(moma.get_anns_act(ids_act=ids_act), desc=f"PREPROCESSING ({split})"):
            sact_seq = []
            for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
                sid2cname[sact.cid] = sact.cname
                sact_seq.append(sact.cid)
            vid2seq[act.id] = np.array(sact_seq)

    sid2emb = {} # sub-activity id -> sub-activity caption embedding
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    for vid, cname in sid2cname.items():
        cemb = sbert.encode(cname)
        cemb = torch.from_numpy(cemb).float()
        sid2emb[vid] = cemb
        
    # cembs = torch.zeros(len(sid2emb), 384)
    # for idx, emb in sid2emb.items():
    #     cembs[idx] = emb

    # cembs = F.normalize(cembs, dim=-1)
    # sim = torch.mm(cembs, cembs.t())
    # sim = sim.numpy()

    return vid2seq, sid2cname, sid2emb


########################################## dtw-python package ##########################################
def compute_dtw_python(x, y):
    # x = F.normalize(x, dim=-1)
    # y = F.normalize(y, dim=-1)
    # z = torch.mm(x, y.t())

    alignment = dtw(
        x, 
        y, 
        dist_method="cosine", 
        keep_internals=True,
        step_pattern=symmetric2, 
        # window_type="sakoechiba",
        # window_args={"window_size": 3},
        # open_begin=True,
        # open_end=True,
        # step_pattern=asymmetric, 
        # open_begin=True,
        # open_end=True,
    )

    return alignment
########################################################################################################


########################################### VT-TWINS version ###########################################
def s2dtw_dist_func(x, y, gamma=1e-1, threshold=0.5):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = torch.mm(x, y.t())

    m, n = z.shape[0], z.shape[1]
    a1 = torch.ones((m + 1, n + 1)) * -math.inf
    a2 = torch.ones((m + 1, n + 1)) * -math.inf
    a3 = torch.ones((m + 1, n + 1)) * -math.inf  
    a1[:m, 1:n+1] = z
    a2[1:m+1, :n] = z
    a3[1:m+1, 1:n+1] = z
    a1[0, 0], a2[0, 0], a3[0, 0] = 0, 0, 0

    # local neighborhood smoothing
    D = z + gamma * torch.log(torch.exp(a1/gamma) + torch.exp(a2/gamma) + torch.exp(a3/gamma))[:m, :n]

    D = torch.cat((D, torch.ones_like(z)*threshold), dim=1)
    D = D.reshape(2*m, n)
    D = torch.cat((torch.ones(1, n, dtype=z.dtype)*threshold, D), dim=0)
    D = torch.cat((D, torch.ones_like(D)*threshold), dim=0)
    D = D.transpose(0, 1).reshape(2*n, 2*m+1).transpose(0, 1)
    D = torch.cat((torch.ones(2*m + 1, 1, dtype=z.dtype)*threshold, D), dim=1)

    return D

    
# @jit(nopython=True)
def compute_s2dtw(x, y):
    # local neighborhood smoothing
    gamma = 1e-1
    threshold = 0.5
    D = s2dtw_dist_func(x, y, gamma, threshold)

    plt.matshow(D)

    # compute optimal alignment
    m, n = D.shape[0], D.shape[1]
    R = np.ones((m+2, n+2)) * -np.inf
    R[0, 0] = 0

    for j in range(1, n+1):
        for i in range(1, m+1):
            r0 = R[i-1, j-1] / gamma
            r1 = R[i-1, j] / gamma
            r2 = R[i, j-1] / gamma
            rmax = max(r0, r1, r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmax = gamma * (np.log(rsum) + rmax)
            R[i, j] = D[i-1, j-1] + softmax

    # backtracking
    i, j, path = R.shape[0] - 2, R.shape[1] - 2, []
    misalign = 0
    while i >= 1 and j >= 1:
        path.append((i-1, j-1))
        if (i-1) % 2 == 0 or (j-1) % 2 == 0:
            misalign += 1

        r0 = R[i-1, j-1] / gamma
        r1 = R[i-1, j] / gamma
        r2 = R[i, j-1] / gamma
        rmax = max(r0, r1, r2)

        if r0 == rmax:
            print("hi")
            i, j = i - 1, j - 1
        elif r1 == rmax:
            i = i - 1
        elif r2 == rmax:
            j = j - 1
        else:
            raise ValueError

    print(f"misalign ratio: {(misalign / len(path)) * 100.:.2f}%")

    return R, path[::-1]
########################################################################################################


########################################### DTW - my version ############################################
def dtw_dist_func(x, y, gamma=1e-1):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = torch.mm(x, y.t())

    m, n = z.shape[0], z.shape[1]
    a1 = torch.ones((m + 1, n + 1)) * -math.inf
    a2 = torch.ones((m + 1, n + 1)) * -math.inf
    a3 = torch.ones((m + 1, n + 1)) * -math.inf  
    a1[:m, 1:n+1] = z
    a2[1:m+1, :n] = z
    a3[1:m+1, 1:n+1] = z
    a1[0, 0], a2[0, 0], a3[0, 0] = 0, 0, 0

    # local neighborhood smoothing
    # D = z + gamma * torch.log(torch.exp(a1/gamma) + torch.exp(a2/gamma) + torch.exp(a3/gamma))[:m, :n]
    D = z

    return D


def temporal_augmentation(x):
    d = 1000
    while d >= 0.02:
        w = 4
        j = random.choice(range(x.shape[0]))
        window = np.arange(max(0, j-w), min(x.shape[0], j+w+1))
        window = np.delete(window, np.where(window == j))
        i = random.choice(window)

        y = x.clone()
        y[i,:], y[j,:] = y[j,:].clone(), y[i,:].clone()

        x = F.normalize(x, dim=-1)
        x_s = torch.mm(x, x.t())
        y = F.normalize(y, dim=-1)
        y_s = torch.mm(y, y.t())
        d = F.mse_loss(x_s, y_s)

    return y


def compute_dtw(x, y):
    gamma = 1e-1 # softmax (local negiborhood smoothing) parameter
    eps = -.3 # gap penalty
    D = dtw_dist_func(x, y, gamma)

    m, n = D.shape[0], D.shape[1]
    R = np.ones((m+1, n+1)) * -np.inf 
    R[0, 0] = 0 # always matching first symbol

    for i in range(1, m+1):
        for j in range(1, n+1):
            r0 = R[i-1, j-1] + D[i-1, j-1]
            r1 = R[i-1, j] - eps
            r2 = R[i, j-1] - eps
            # print(f"({i} {j}) r0: {r0} r1: {r1} r2: {r2} rmax: {max(r0, r1, r2)}")
            R[i, j] = max(r0, r1, r2)

    # backtracking
    i, j, path = m, n, []
    while i >= 1 and j >= 1:
        path.append((i, j)) # if for D, (i-1, j-1)

        r0 = R[i-1, j-1] + D[i-1, j-1]
        r1 = R[i-1, j] - eps
        r2 = R[i, j-1] - eps
        rmax = max(r0, r1, r2)

        if rmax == r0:
            i, j = i - 1, j - 1
        elif rmax == r1:
            i = i - 1
        elif rmax == r2:
            j = j - 1
        else:
            raise ValueError

    return R, path[::-1]
########################################################################################################


########################################### tslearn ####################################################
def cosine_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return -torch.mm(x, y.t()).numpy()
########################################################################################################


############################################### Standard DTW ###############################################
def compute_standard_dtw(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    s = torch.mm(x, y.t())

    m, n = s.shape[0], s.shape[1]
    R = np.ones((m+1, n+1)) * -np.inf
    R[0, 0] = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            R[i, j] = s[i-1, j-1] + max(R[i-1, j], R[i, j-1], R[i-1, j-1])

    # backtracking
    i, j, path = m, n, []
    while i >= 1 and j >= 1:
        path.append((i-1, j-1))
        rmax = s[i-1, j-1] + max(R[i-1, j], R[i, j-1], R[i-1, j-1])

        if rmax == s[i-1, j-1] + R[i-1, j]:
            i = i - 1
        elif rmax == s[i-1, j-1] + R[i, j-1]:
            j = j - 1
        elif rmax == s[i-1, j-1] + R[i-1, j-1]:
            i, j = i-1, j-1
        else: 
            raise ValueError

    return s, R, path
############################################################################################################


def main():
    # PREPROCESSING
    if os.path.exists("vid2seq.pt") and os.path.exists("sid2cname.pt") and os.path.exists("sid2emb.pt"):
        print("Load pre-processed data.")
        vid2seq = torch.load("vid2seq.pt")
        sid2cname = torch.load("sid2cname.pt")
        sid2emb = torch.load("sid2emb.pt")
    else:
        print("Start pre-processing.")
        vid2seq, sid2cname, sid2emb = preprocess()
        torch.save(vid2seq, "vid2seq.pt")
        torch.save(sid2cname, "sid2cname.pt")
        torch.save(sid2emb, "sid2emb.pt")


    ########################################### dtw-python ###########################################
    print("dtw-python")
    # x = vid2seq["24q1O3M9xiw"]
    x = np.array([23, 23, 23, 62, 62, 62, 62, 62, 62, 23, 23, 23])
    x = torch.stack([sid2emb[sid] for sid in x], dim=0)
    # y = vid2seq["Yg2oElKPCas"]
    y = np.array([24, 24, 24, 62, 62, 62, 62, 62, 62, 24, 24, 24])
    y = torch.stack([sid2emb[sid] for sid in y], dim=0)


    alignment = compute_dtw_python(x, y)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = torch.mm(x, y.t())
    # plt.matshow(alignment.costMatrix)
    plt.matshow(z)
    plt.plot(alignment.index2, alignment.index1, "w")
    plt.savefig("output1.png")
    plt.close()
    print(f"score: {1 - (alignment.distance / len(alignment.index1))} len_path: {len(alignment.index1)}")
    # print(f"score: {alignment.distance / (len(alignment.index1) + len(alignment.index2))} len_path: {len(alignment.index1)}")
    ##################################################################################################

    print("=========================================================================================")

    ############################################# S2DTW #############################################
    x = vid2seq["24q1O3M9xiw"]
    x = torch.stack([sid2emb[sid] for sid in x], dim=0)
    y = vid2seq["Yg2oElKPCas"]
    y = torch.stack([sid2emb[sid] for sid in y], dim=0)
     
    R, path = compute_s2dtw(x, y)
    # plt.matshow(R)
    plt.plot([p[1] for p in path], [p[0] for p in path], "w")
    plt.savefig("output2.png")
    plt.close()
    print(f"score: {R[-2, -2] / len(path)} len_path: {len(path)}")
    #################################################################################################

    print("=========================================================================================")

    ############################################# DTW-myversion #############################################
    print("my version")
    # x = vid2seq["24q1O3M9xiw"]
    x = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21]) # synthetic data
    x = torch.stack([sid2emb[sid] for sid in x], dim=0)
    # y = vid2seq["Yg2oElKPCas"]
    y = np.array([21, 20, 19, 18, 17, 16, 15, 14, 13]) # synthetic data
    y = torch.stack([sid2emb[sid] for sid in y], dim=0)

    cost = []
    for i in range(50):
        x_p = temporal_augmentation(x)
        y_p = temporal_augmentation(y)
        R, path = compute_dtw(x_p, y_p)
        cost.append(R[x.shape[0], y.shape[0]] / len(path))
    
    print(f"temporal augmentation: {np.array(cost).mean()}")
     
    R, path = compute_dtw(x, y)
    plt.matshow(R)
    plt.plot([p[1] for p in path], [p[0] for p in path], "w")
    plt.savefig("output3.png")
    plt.close()
    print(f"score: {R[x.shape[0], y.shape[0]] / len(path)} len_path: {len(path)}")
    #########################################################################################################

    print("=========================================================================================")

    ############################################# tslearn #############################################
    print("tslearn")
    # x = vid2seq["24q1O3M9xiw"]
    x = np.array([29, 82, 83, 84, 85, 86, 87]) # synthetic data
    x = torch.stack([sid2emb[sid] for sid in x], dim=0)
    # y = vid2seq["Yg2oElKPCas"]
    y = np.array([87, 86, 85, 84, 83, 82, 29]) # synthetic data
    y = torch.stack([sid2emb[sid] for sid in y], dim=0)

    z = cosine_distance(x, y)
    path, cost = dtw_path_from_metric(x.numpy(), y.numpy(), metric="cosine")
    plt.matshow(z)
    plt.plot([p[1] for p in path], [p[0] for p in path], "w")
    plt.savefig("output4.png")
    plt.close()
    print(f"score: {1 - (cost/ len(path))} len_path: {len(path)}")
    #########################################################################################################

    print("=========================================================================================")

    ############################################# standard DTW #############################################
    print("standard dtw")
    x = np.array([23, 23, 23, 23, 62, 62, 62, 62, 23, 23, 23, 23]) # synthetic data
    x = torch.stack([sid2emb[sid] for sid in x], dim=0)
    y = np.array([24, 24, 62, 62, 62, 62, 24, 24]) # synthetic data
    y = torch.stack([sid2emb[sid] for sid in y], dim=0)

    # cost = []
    # for i in range(100):
    #     x_p = temporal_augmentation(x)
    #     y_p = temporal_augmentation(y)
    #     s, R, path = compute_standard_dtw(x_p, y_p)
    #     cost.append(R[x.shape[0], y.shape[0]] / len(path))

    # s, R, path = compute_standard_dtw(x, y)

    # print(f"not augmented: {R[x.shape[0], y.shape[0]] / len(path)} augmented: mean({np.array(cost).mean()}) std({np.array(cost).std()})")

    s, R, path = compute_standard_dtw(x, y)
    plt.matshow(s)
    plt.plot([p[1] for p in path], [p[0] for p in path], "w")
    plt.savefig("output5.png")
    plt.close()
    print(f"score: {R[x.shape[0], y.shape[0]] / len(path)} len_path: {len(path)}")
    #########################################################################################################

    ############ RESERVE FOR BATCHING ##############
    
    # # z = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    # z = torch.zeros(9, 1)
    # m, n = z.shape[0], z.shape[1]
    # D = z

    # threshold = 1. 
    # D = torch.cat((D, torch.ones_like(z)*threshold), dim=1)
    # D = D.reshape(2*m, n)
    # D = torch.cat((torch.ones(1, n, dtype=z.dtype)*threshold, D), dim=0)
    # D = torch.cat((D, torch.ones_like(D)*threshold), dim=0)
    # D = D.transpose(0, 1).reshape(2*n, 2*m+1).transpose(0, 1)
    # D = torch.cat((torch.ones(2*m + 1, 1, dtype=z.dtype)*threshold, D), dim=1)

    # import matplotlib.pyplot as plt
    # plt.matshow(D.numpy())
    # plt.savefig("tmp.png")
    # plt.close()


if __name__ == "__main__":
    main()