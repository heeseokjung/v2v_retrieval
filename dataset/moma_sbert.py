import numpy as np
import torch

from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    moma = MOMA("/data/dir_moma")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    id2cembs_train = {}
    ids_act = moma.get_ids_act()
    for act in tqdm(moma.get_anns_act(ids_act=ids_act)):
        seq = []
        for sact in moma.get_anns_sact(ids_sact=act.ids_sact):
            seq.append(sact.cname)
        seq = list(set(seq))
        with torch.no_grad():
            emb = sbert.encode(seq)
            emb = torch.from_numpy(emb).float()
        id2cembs_train[act.id] = emb

    torch.save(id2cembs_train, "anno/moma/id2cembs_train.pt")

if __name__ == "__main__":
    main()