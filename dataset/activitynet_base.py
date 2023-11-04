import os
import json
import numpy as np
import torch

from torch.utils.data import Dataset


class ActivityNetCaptionsRetrievalBaseDataset(Dataset):
    def __init__(self, cfg, split):
        super().__init__()

        self.cfg = cfg
        self.args = cfg.DATASET.actvitynet
        self.split = split

        self._load_anno_data()
        self._set_pairwise_similarity()

    def _load_anno_data(self):
        with open(f"anno/activitynet/{self.split}.ndjson", "r") as f:
            self.anno = json.load(f)
        self.id2cemb = torch.load("anno/activitynet/id2cemb.pt")

    def _set_pairwise_similarity(self):
        type = self.cfg.RELEVANCE.type
        print(f"Load from pre-computed surrogate measure [{type}-{self.split}]")
        self.sm = np.load(f"anno/activitynet/sm_{type}_{self.split}.npy")
        self.sm = torch.from_numpy(self.sm).float()

    def load_video(self, vid):
        if self.cfg.MODEL.name == "ours":
            path = os.path.join(self.args.path, "feats", "s3d", f"{vid}.npy")
            feat = torch.from_numpy(np.load(path)).float() # n_clips x d
        else:
            path = os.path.join(self.args.path, "feats", "preprocessed", f"{vid}.npy")
            feat = torch.from_numpy(np.load(path)).float() # d,
        return feat

    def __len__(self):
        return len(self.anno)