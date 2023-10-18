import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dtw import *
from momaapi import MOMA
from sentence_transformers import SentenceTransformer
    

def cosine_similarity_m(x, y):
    return -np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def main():
    ####################################################################
    if os.path.exists("vid2embs.pt"):
        vid2embs = torch.load("vid2embs.pt")

    else:
        moma = MOMA("/data/dir_moma")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

        vid2embs = {}
        ids_act = moma.get_ids_act() # all 1412 video ids
        
        for act in moma.get_anns_act(ids_act=ids_act):
            seq = []
            for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
                seq.append(sact.cname)
            with torch.no_grad():
                seq_emb = sbert.encode(seq) # n_captions x d
            vid2embs[act.id] = seq_emb

        torch.save(vid2embs, "vid2embs.pt")

    #####################################################################

    # x = vid2embs["24q1O3M9xiw"]
    # y = vid2embs["m0MtHFfhhFs"]
    # y = vid2embs["OXxcLxvdHBc"]
    # y = vid2embs["V_1cchOYMFw"]
    # y = vid2embs["Yg2oElKPCas"]
    # y = vid2embs["24q1O3M9xiw"]
    x = np.random.randn(12, 384)
    y = np.random.randn(1, 384)

    alignment = dtw(
        x, 
        y, 
        dist_method=cosine_similarity_m,
        step_pattern=symmetric2,
        keep_internals=True,
        distance_only=False,
    )
     
    print(f"score: {-alignment.distance / (x.shape[0] + y.shape[0])}")
    print(f"score: {-alignment.normalizedDistance}")


    x = torch.from_numpy(x).float()
    x = F.normalize(x, dim=-1)
    y = torch.from_numpy(y).float()
    y = F.normalize(y, dim=-1)
    z = torch.mm(x, y.t())
    plt.matshow(z.numpy())
    plt.plot(alignment.index2, alignment.index1, "w")
    plt.savefig("out.png")
    plt.close()


if __name__ == "__main__":
    main()