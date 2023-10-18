import numpy as np
import torch
import matplotlib.pyplot as plt

from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def compute_dtw_score(x, y, sx, sy):
    nx = x / np.linalg.norm(x, axis=-1, keepdims=True)
    ny = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.matmul(nx, ny.T)

    m, n = z.shape[0], z.shape[1]
    R = np.ones((m+1, n+1))
    R[0,:], R[:,0] = -np.inf, -np.inf
    R[0,0] = 0

    eps, w = -0.1, 3

    for i in range(1, m+1):
        for j in range(1, n+1):
            if abs(i - j) > w:
                continue
            r0 = R[i-1, j-1] + z[i-1, j-1]
            r1 = R[i-1, j] + eps
            r2 = R[i, j-1] + eps
            R[i, j] = max(r0, r1, r2)

    # backtracking
    i, j, path = m, n, []
    matching = []
    size = 0
    while i >= 1 and j >= 1:
        path.append((i, j)) # if for D, (i-1, j-1)

        r0 = R[i-1, j-1] + z[i-1, j-1]
        r1 = R[i-1, j] + eps
        r2 = R[i, j-1] + eps
        rmax = max(r0, r1, r2)

        if rmax == r0:
            matching.append((sx[i-1], sy[j-1]))
            i, j = i - 1, j - 1
            size += 1
        elif rmax == r1:
            matching.append((sx[i-1], "__"))
            i = i - 1
        elif rmax == r2:
            matching.append(("__", sy[j-1]))
            j = j - 1
        else:
            raise ValueError
        
    ###############################################
    # plt.matshow(z)
    # plt.plot([p[1]-1 for p in path[::-1]], [p[0]-1 for p in path[::-1]], "w")
    # plt.savefig("out.png")
    # plt.close()

    for k in matching[::-1]:
        print(f"{k[0]}", end=" ")
    print()
    for k in matching[::-1]:
        print(f"{k[1]}", end=" ")
    print()
    ###############################################
        
    return R[m, n] / size
    

def main():
    moma = MOMA("/data/dir_moma")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    vid2seqembs = {}
    ids_act = moma.get_ids_act()
    
    # for act in tqdm(moma.get_anns_act(ids_act=ids_act), desc="[MOMA] preprocessing"):
    #     seq = []
    #     for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
    #         seq.append(sact.cname)
    #     with torch.no_grad():
    #         seq_emb = sbert.encode(seq)
    #     vid2seqembs[act.id] = seq_emb

    # torch.save(vid2seqembs, "vid2seqembs.pt")

    vid2seq = {}
    for act in tqdm(moma.get_anns_act(ids_act=ids_act), desc="[MOMA] preprocessing"):
        seq = []
        for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
            seq.append(sact.cid)
        vid2seq[act.id] = seq
    
    vid2seqembs = torch.load("vid2seqembs.pt")

    # for vi in tqdm(ids_act):
    #     for vj in ids_act:
    #         if vi == vj:
    #             continue
    #         x = vid2seqembs[vi]
    #         y = vid2seqembs[vj]

    #         # print("===========================================================================")
    #         score1 = compute_dtw_score(x, y)
    #         # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #         score2 = compute_dtw_score(y, x)
    #         # print(f"score1: {score1} score2: {score2}")
    #         if not np.allclose(score1, score2):
    #             print(abs(score1 - score2))
    #         # print(f"allclose: {np.allclose(score1, score2)}")
    #         # print(f"diff: {abs(score1 - score2)}")
    #         # break

    q = vid2seqembs["elijDDPWOVA"]
    v1 = vid2seqembs["uC13kU8_Tzs"]
    v2 = vid2seqembs["5x3YYy4OHOg"]

    print(f"rel1: {compute_dtw_score(q, v1, vid2seq['elijDDPWOVA'], vid2seq['uC13kU8_Tzs'])}")
    print(f"rel2: {compute_dtw_score(q, v2, vid2seq['elijDDPWOVA'], vid2seq['5x3YYy4OHOg'])}")

if __name__ == "__main__":
    main()