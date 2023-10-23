import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer


############################# UTILS #############################
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def compute_dtw_similarity(x, y):
    nx = F.normalize(x, dim=-1)
    ny = F.normalize(y, dim=-1)
    z = torch.mm(nx, ny.t())

    m, n = z.shape[0], z.shape[1]
    R = torch.ones((m+1, n+1)).to(x.device) * -float("inf")
    R[0,0] = 0.

    for i in range(1, m+1):
        for j in range(1, n+1):
            r0 = R[i-1, j-1] + 2*z[i-1, j-1]
            r1 = R[i-1, j] + z[i-1, j-1]
            r2 = R[i, j-1] + z[i-1, j-1]
            R[i, j] = max(r0, r1, r2) 

    return R[m, n] / (m + n)
#################################################################


class User:
    def __init__(self, session_id):
        self.session_id = session_id
        self.poll = {}

    def update(self, tid, choice):
        if choice == 0:
            self.poll[tid] = "s1"
        elif choice == 1:
            self.poll[tid] = "s2"
        elif choice == 2:
            self.poll[tid] = "s3"
        elif choice == 3:
            self.poll[tid] = "s4"
        else:
            raise ValueError
        

class Triplet:
    def __init__(self, tid):
        self.tid = tid
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0
        self.s4 = 0

    def update(self, choice):
        if choice == 0:
            self.s1 += 1
        elif choice == 1:
            self.s2 += 1
        elif choice == 2:
            self.s3 += 1
        elif choice == 3:
            self.s4 += 1
        else:
            raise ValueError
        
    def set_info(self, data):
        self.dataset = data["dataset"]
        self.qvid = data["qvid"]
        self.vid1 = data["vid1"]
        self.vid2 = data["vid2"]
        self.rel1 = data["rel1"]
        self.rel2 = data["rel2"]


def compute_inter_human_agreement(user_list, tid2triplet):
    user_score_list = []
    for user in user_list:
        triplet_score_list = []
        for tid, choice in user.poll.items():
            triplet = tid2triplet[tid]
            s1 = triplet.s1
            s2 = triplet.s2
            s3 = triplet.s3
            s4 = triplet.s4

            # if triplet.dataset == "moma-lrg":
            #     continue

            if s1 + s2 < 2:
                continue

            if choice == "s1":
                triplet_score_list.append(
                    ((s1 - 1) + 0.5*s3) / (s1 + s2 + s3 + s4)
                )
            elif choice == "s2":
                triplet_score_list.append(
                    ((s2 - 1) + 0.5*s3) / (s1 + s2 + s3 + s4)
                )
            elif choice == "s3":
                triplet_score_list.append(
                    (0.5*s1 + 0.5*s2 + (s3-1)) / (s1 + s2 + s3 + s4)
                )
            elif choice == "s4":
                '''
                If a human marks that neither of the candidates is relevant for a triplet, 
                the triplet is not counted in the human agreement score.
                '''
                continue
            else:
                raise ValueError

        triplet_score_list = np.array(triplet_score_list)
        user_score_list.append(triplet_score_list.mean())

    user_score_list = np.array(user_score_list)
    return user_score_list.mean(), user_score_list.std()


def eval_random_agreement(tid2triplet):
    score_list = []
    for tid, triplet in tid2triplet.items():
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4

        if s1 + s2 < 2:
            continue

        # random choice between s1 and s2
        choice = random.randint(0, 1)

        if choice == 0: # s1
            score_list.append((s1 + 0.5*s3) / (s1 + s2 + s3 + s4))
        elif choice == 1: # s2
            score_list.append((s2 + 0.5*s3) / (s1 + s2 + s3 + s4))
        else:
            raise ValueError

    score_list = np.array(score_list)
    return score_list.mean()


def eval_visual_agreement(tid2triplet, method):
    score_list = []
    for tid, triplet in tid2triplet.items():
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4

        if s1 + s2 < 2:
            continue
        
        if triplet.dataset == "moma-lrg":
            if method == "s3d":
                q = np.load(os.path.join("/data/dir_moma/feats/s3d", f"{triplet.qvid}.npy")).mean(axis=0)
                v1 = np.load(os.path.join("/data/dir_moma/feats/s3d", f"{triplet.vid1}.npy")).mean(axis=0)
                v2 = np.load(os.path.join("/data/dir_moma/feats/s3d", f"{triplet.vid2}.npy")).mean(axis=0)
            elif method == "frozen":
                q = np.load(os.path.join("/data/dir_moma/feats/frozen", f"{triplet.qvid}.npy"))[:,0,:].mean(axis=0)
                v1 = np.load(os.path.join("/data/dir_moma/feats/frozen", f"{triplet.vid1}.npy"))[:,0,:].mean(axis=0)
                v2 = np.load(os.path.join("/data/dir_moma/feats/frozen", f"{triplet.vid2}.npy"))[:,0,:].mean(axis=0)
        elif triplet.dataset == "activitynet-captions":
            q = np.load(os.path.join("/data/dir_activitynet/feats/imagenet", f"{triplet.qvid}.npy"))
            v1 = np.load(os.path.join("/data/dir_activitynet/feats/imagenet", f"{triplet.vid1}.npy"))
            v2 = np.load(os.path.join("/data/dir_activitynet/feats/imagenet", f"{triplet.vid2}.npy"))
        pred_1 = cosine_similarity(q, v1)
        pred_2 = cosine_similarity(q, v2)

        if pred_1 > pred_2:
            score_list.append((s1 + 0.5*s3) / (s1 + s2 + s3 + s4))
        elif pred_1 < pred_2:
            score_list.append((s2 + 0.5*s3) / (s1 + s2 + s3 + s4))
        else:
            print(f"pred1: {pred_1} pred2: {pred_2}")
            raise ValueError
        
    score_list = np.array(score_list)
    return score_list.mean(), score_list.std()


def eval_dtw_agreement(tid2triplet):
    score_list = []
    for tid, triplet in tid2triplet.items():
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4

        if s1 + s2 < 2:
            continue

        if triplet.rel1 > triplet.rel2:
            score_list.append((s1 + 0.5*s3) / (s1 + s2 + s3 + s4))
        elif triplet.rel1 < triplet.rel2:
            score_list.append((s2 + 0.5*s3) / (s1 + s2 + s3 + s4))
        else:
            raise ValueError

    score_list = np.array(score_list)
    return score_list.mean(), score_list.std()


def eval_normalized_dtw_agreement(tid2triplet):
    id2cemb = torch.load("/root/cvpr24_video_retrieval/anno/moma/id2cemb.pt")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    with open("/data/dir_activitynet/anno/train.json", "r") as f:
        train_anno = json.load(f)
    with open("/data/dir_activitynet/anno/val_1.json", "r") as f:
        val_anno = json.load(f)
    train_anno.update(val_anno)
    anno = train_anno

    score_list = []
    for tid, triplet in tid2triplet.items():
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4

        if s1 + s2 < 2:
            continue
        
        if triplet.dataset == "moma-lrg":
            q = id2cemb[triplet.qvid]
            v1 = id2cemb[triplet.vid1]
            v2 = id2cemb[triplet.vid2]
        elif triplet.dataset == "activitynet-captions":
            q = sbert.encode(anno[triplet.qvid]["sentences"])
            q = torch.from_numpy(q).float()
            v1 = sbert.encode(anno[triplet.vid1]["sentences"])
            v1 = torch.from_numpy(v1).float()
            v2 = sbert.encode(anno[triplet.vid2]["sentences"])
            v2 = torch.from_numpy(v2).float()
        pred_1 = compute_dtw_similarity(q, v1)
        pred_2 = compute_dtw_similarity(q, v2)

        if pred_1 > pred_2:
            score_list.append((s1 + 0.5*s3) / (s1 + s2 + s3 + s4))
        elif pred_1 < pred_2:
            score_list.append((s2 + 0.5*s3) / (s1 + s2 + s3 + s4))
        else:
            raise ValueError

    score_list = np.array(score_list)
    return score_list.mean(), score_list.std()


def dataset_statistics(tid2triplet):
    moma = {"s1": 0, "s2": 0, "s3": 0, "s4": 0}
    activitynet = {"s1": 0, "s2": 0, "s3": 0, "s4": 0}

    for tid, triplet in tid2triplet.items():
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4
        if triplet.dataset == "moma-lrg":
            moma["s1"] += s1
            moma["s2"] += s2
            moma["s3"] += s3
            moma["s4"] += s4
        elif triplet.dataset == "activitynet-captions":
            activitynet["s1"] += s1
            activitynet["s2"] += s2
            activitynet["s3"] += s3
            activitynet["s4"] += s4

    return moma, activitynet


def main():
    # preprocessing
    user_list, tid2triplet = [], {}
    for filename in os.listdir("poll"):
        if filename.startswith("personal_"):
            session_id = filename[9:-4]
            user = User(session_id=session_id)

            poll = pd.read_csv(os.path.join("poll", f"{session_id}.csv"))
            for idx, row in poll.iterrows():
                tid, choice = row["id"], row["choice"]
                if tid not in tid2triplet:
                    tid2triplet[tid] = Triplet(tid=tid)
                tid2triplet[tid].update(choice)
                user.update(tid, choice)

            user_list.append(user)

    questions = pd.read_csv("pilot.csv")
    for idx, row in questions.iterrows():
        tid = row["tid"]
        tid2triplet[tid].set_info(row)

    # inter-human agreement score
    mu, std = compute_inter_human_agreement(user_list, tid2triplet)
    print(f"inter-human agreement score: {mu} ± {std}")

    # random-human agreement score
    score_per_run = []
    for _ in range(10):
        score_per_run.append(eval_random_agreement(tid2triplet))
    score_per_run = np.array(score_per_run)
    mu, std = score_per_run.mean(), score_per_run.std()
    print(f"random-human agreement score: {mu} ± {std}")

    # s3d-human agreement score
    mu, std = eval_visual_agreement(tid2triplet, "s3d")
    print(f"s3d-human agreement score: {mu} ± {std}")
    
    # frozen-human agreement score
    mu, std = eval_visual_agreement(tid2triplet, "frozen")
    print(f"frozen-human agreement score: {mu} ± {std}")

    # dtw-human agreement score
    mu, std = eval_dtw_agreement(tid2triplet)
    print(f"dtw-human agreement score: {mu} ± {std}")

    # normalized_dtw-human agreement score
    mu, std = eval_normalized_dtw_agreement(tid2triplet)
    print(f"dtw(symmetric2)-human agreement score: {mu} ± {std}")

    # dataset statistics
    moma, activitynet = dataset_statistics(tid2triplet)
    print(f"[MOMA-LRG] s1: {moma['s1']} s2: {moma['s2']} s3: {moma['s3']} s4: {moma['s4']}")
    print(f"[ActivityNet-Captions] s1: {activitynet['s1']} s2: {activitynet['s2']} s3: {activitynet['s3']} s4: {activitynet['s4']}")


if __name__ == "__main__":
    main()