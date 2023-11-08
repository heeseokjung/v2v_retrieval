import os
import random
import torch
import numpy as np
import pandas as pd


############################# Algorithms #############################
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# numpy version
def compute_dtw_similarity(x, y):
    nx = x / np.linalg.norm(x, axis=-1, keepdims=True)
    ny = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.matmul(nx, ny.T)

    m, n = z.shape[0], z.shape[1]
    R = np.ones((m+1, n+1)) * -np.inf
    R[0, 0] = 0.

    for i in range(1, m+1):
        for j in range(1, n+1):
            r0 = R[i-1, j-1] + 2*z[i-1, j-1]
            r1 = R[i-1, j] + z[i-1, j-1]
            r2 = R[i, j-1] + z[i-1, j-1]
            R[i, j] = max(r0, r1, r2) 

    return R[m, n] / (m + n)
######################################################################


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

        self.dataset = None
        self.qvid = None
        self.vid1 = None
        self.vid2 = None
        self.rel1 = None
        self.rel2 = None

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
        
    def set(self, data):
        self.dataset = data["dataset"]
        self.qvid = data["qvid"]
        self.vid1 = data["vid1"]
        self.vid2 = data["vid2"]
        self.rel1 = data["rel1"]
        self.rel2 = data["rel2"]


def parse_poll(num_triplets, skip_anonymous):
    info_list, user_list = [], []
    triplet_list = [Triplet(tid=i) for i in range(num_triplets)]

    problem_set = pd.read_csv("problem_set.csv")
    for idx, row in problem_set.iterrows():
        tid = int(row["tid"])
        triplet_list[tid].set(row)

    for filename in os.listdir("poll"):
        if skip_anonymous and (not os.path.exists(f"info/personal_{filename}")):
            continue
        
        if os.path.exists(f"info/personal_{filename}"):
            info = pd.read_csv(f"info/personal_{filename}")
            info_list.append(info)

        session_id = filename.split(".")[0]
        user = User(session_id=session_id)

        df = pd.read_csv(os.path.join("poll", filename))
        for idx, row in df.iterrows():
            tid, choice = int(row["id"]), int(row["choice"])
            triplet_list[tid].update(choice)
            user.update(tid, choice)
        
        user_list.append(user)

    if info_list:
        cat_info = pd.concat(info_list, ignore_index=True)
        cat_info = cat_info.sort_values(by=["name"])
        cat_info = cat_info.to_csv("total_info.csv")

    return user_list, triplet_list


def compute_triplet_statistics(triplet_list):
    polls = []
    s1, s2, s3, s4 = [], [], [], []
    for triplet in triplet_list:
        s1.append(triplet.s1)
        s2.append(triplet.s2)
        s3.append(triplet.s3)
        s4.append(triplet.s4)
        polls.append(triplet.s1 + triplet.s2 + triplet.s3 + triplet.s4)

    polls = np.array(polls)
    s1 = np.array(s1).sum() / polls.sum()
    s2 = np.array(s2).sum() / polls.sum()
    s3 = np.array(s3).sum() / polls.sum()
    s4 = np.array(s4).sum() / polls.sum()

    print(f"Number of polls per triplet: {polls.mean()} ± {polls.std()}, [min: {polls.min()}, max: {polls.max()}]")
    print(f"Distribution of choices: s1 ({s1:.2f}) s2 ({s2:.2f}) s3 ({s3:.2f}) s4 ({s4:.2f})")


def metric_human(choice, s1, s2, s3, s4):
    if choice == "s1":
        return (s1 + 0.5*s3 - 1) / (s1 + s2 + s3 + s4 - 1)
    elif choice == "s2":
        return (s2 + 0.5*s3 - 1) / (s1 + s2 + s3 + s4 - 1)
    elif choice == "s3":
        return (0.5*s1 + 0.5*s2 + s3 - 1) / (s1 + s2 + s3 + s4 - 1)
    

def metric_algorithm(choice, s1, s2, s3, s4):
    if choice == "s1":
        return (s1 + 0.5*s3) / (s1 + s2 + s3 + s4)
    elif choice == "s2":
        return (s2 + 0.5*s3) / (s1 + s2 + s3 + s4)


def compute_inter_human_agreement(user_list, triplet_list):
    per_user_scores = []
    for user in user_list:
        per_triplet_scores = []
        for tid, choice in user.poll.items():
            triplet = triplet_list[tid]
            s1 = triplet.s1
            s2 = triplet.s2
            s3 = triplet.s3
            s4 = triplet.s4

            if s1 + s2 < 2:
                continue
            
            if choice == "s1":
                per_triplet_scores.append(metric_human(choice, s1, s2, s3, s4))
            elif choice == "s2":
                per_triplet_scores.append(metric_human(choice, s1, s2, s3, s4))
            elif choice == "s3":
                per_triplet_scores.append(metric_human(choice, s1, s2, s3, s4))
            elif choice == "s4":
                '''
                If a human marks that neither of the candidates is relevant for a triplet, 
                the triplet is not counted in the human agreement score.
                '''
            else:
                raise ValueError
        
        if per_triplet_scores:
            per_triplet_scores = np.array(per_triplet_scores)
            per_user_scores.append(per_triplet_scores.mean())

    per_user_scores = np.array(per_user_scores)
    return per_user_scores.mean(), per_user_scores.std()


def compute_random_human_agreement(triplet_list, n_iter=10):
    scores = []
    for _ in range(n_iter):
        scores_per_run = []
        for triplet in triplet_list:
            s1 = triplet.s1
            s2 = triplet.s2
            s3 = triplet.s3
            s4 = triplet.s4

            if s1 + s2 < 2:
                continue

            random_choice = random.randint(0, 1)

            if random_choice == 0:
                scores_per_run.append(metric_algorithm("s1", s1, s2, s3, s4))
            elif random_choice == 1:
                scores_per_run.append(metric_algorithm("s2", s1, s2, s3, s4))

        if scores_per_run:
            scores.append(np.array(scores_per_run).mean())
    
    scores = np.array(scores)
    return scores.mean(), scores.std()


def compute_visual_agreement(triplet_list):
    moma_path = "/data/dir_moma/feats/s3d"
    activitynet_path = "/data/dir_activitynet/feats/imagenet"

    scores = []
    for triplet in triplet_list:
        dataset = triplet.dataset
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4

        if s1 + s2 < 2:
            continue

        if dataset == "moma-lrg":
            q = np.load(os.path.join(moma_path, f"{triplet.qvid}.npy")).mean(axis=0)
            v1 = np.load(os.path.join(moma_path, f"{triplet.vid1}.npy")).mean(axis=0)
            v2 = np.load(os.path.join(moma_path, f"{triplet.vid2}.npy")).mean(axis=0)
        elif dataset == "activitynet-captions":
            q = np.load(os.path.join(activitynet_path, f"{triplet.qvid}.npy")).mean(axis=0)
            v1 = np.load(os.path.join(activitynet_path, f"{triplet.vid1}.npy")).mean(axis=0)
            v2 = np.load(os.path.join(activitynet_path, f"{triplet.vid2}.npy")).mean(axis=0)

        pred1 = cosine_similarity(q, v1)
        pred2 = cosine_similarity(q, v2)

        if pred1 > pred2:
            scores.append(metric_algorithm("s1", s1, s2, s3, s4))
        elif pred1 < pred2:
            scores.append(metric_algorithm("s2", s1, s2, s3, s4))
        else:
            continue
        
    scores = np.array(scores)
    return scores.mean(), scores.std()


def compute_dtw_agreement(triplet_list):
    scores = []
    for triplet in triplet_list:
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4

        if s1 + s2 < 2:
            continue

        if triplet.rel1 > triplet.rel2:
            scores.append(metric_algorithm("s1", s1, s2, s3, s4))
        elif triplet.rel1 < triplet.rel2:
            scores.append(metric_algorithm("s2", s1, s2, s3, s4))
        else:
            raise ValueError
        
    scores = np.array(scores)
    return scores.mean(), scores.std()


def compute_dtw_global_agreement(triplet_list):
    moma_id2cemb = torch.load("/root/cvpr24_video_retrieval/anno/moma/id2cemb.pt")
    activitynet_id2cemb = torch.load("/root/cvpr24_video_retrieval/anno/activitynet/id2cemb.pt")

    scores = []
    for triplet in triplet_list:
        s1 = triplet.s1
        s2 = triplet.s2
        s3 = triplet.s3
        s4 = triplet.s4

        if s1 + s2 < 2:
            continue

        if triplet.dataset == "moma-lrg":
            q = moma_id2cemb[triplet.qvid]
            v1 = moma_id2cemb[triplet.vid1]
            v2 = moma_id2cemb[triplet.vid2]
        elif triplet.dataset == "activitynet-captions":
            q = activitynet_id2cemb[triplet.qvid]
            v1 = activitynet_id2cemb[triplet.vid1]
            v2 = activitynet_id2cemb[triplet.vid2]
        else:
            raise ValueError
        
        rel1 = compute_dtw_similarity(q, v1)
        rel2 = compute_dtw_similarity(q, v2)

        if rel1 > rel2:
            scores.append(metric_algorithm("s1", s1, s2, s3, s4))
        elif rel1 < rel2:
            scores.append(metric_algorithm("s2", s1, s2, s3, s4))
        else:
            continue
        
    scores = np.array(scores)
    return scores.mean(), scores.std()


def main():
    # parsing polls
    user_list, triplet_list = parse_poll(num_triplets=1000, skip_anonymous=True)
    print(f"Number of annotators: {len(user_list)}")

    # compute statistics for triplets
    compute_triplet_statistics(triplet_list)

    # inter-human agreement
    mu, std = compute_inter_human_agreement(user_list, triplet_list)
    print(f"inter-human agreement score: {mu} ± {std}")

    # random-human agreement
    mu, std = compute_random_human_agreement(triplet_list)
    print(f"random-human agreement score: {mu} ± {std}")

    # visual-human agreement
    mu, std = compute_visual_agreement(triplet_list)
    print(f"visual-human agreement score: {mu} ± {std}")

    # dtw(global)-human agreement
    mu, std = compute_dtw_global_agreement(triplet_list)
    print(f"dtw(global)-human agreement score: {mu} ± {std}")

    # dtw(sliding window)-human agreement
    mu, std = compute_dtw_agreement(triplet_list)
    print(f"dtw(sliding window)-human agreement score: {mu} ± {std}")


if __name__ == "__main__":
    main()