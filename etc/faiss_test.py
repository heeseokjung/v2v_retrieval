import faiss
import numpy as np
from momaapi import MOMA
from sentence_transformers import SentenceTransformer
from dtw import *
import torch
import os
import random
import torch.nn.functional as F


def main():
    x = np.random.randn(100, 512).astype(np.float32)

    idx = faiss.IndexFlatIP(512)
    faiss.normalize_L2(x)
    idx.add(x)

    k = 5
    D, I = idx.search(x[12:15,:], k)

    print(D)
    print(I)



if __name__ == "__main__":
    main()