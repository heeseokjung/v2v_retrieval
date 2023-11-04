import copy
import torch
import numpy as np
from dataset.activitynet_base import ActivityNetCaptionsRetrievalBaseDataset


class ActivityNetCaptionsRetrievalTrainDataset(ActivityNetCaptionsRetrievalBaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg, "train")

    def sample_pair(self, idx):
        similarties = self.sm[idx]
        _, sorted_idx = torch.sort(similarties, descending=True)
        mask = (sorted_idx != idx)
        sorted_idx = sorted_idx[mask]
        
        # oversampling
        if self.args.use_over_sampling and np.random.rand(1) < 0.5:
            pair_idx = sorted_idx[np.random.randint(self.args.tail_range)]
        else:
            pair_idx = sorted_idx[np.random.randint(len(sorted_idx))]

        return pair_idx, similarties[pair_idx]

    def __getitem__(self, idx):
        anchor = copy.deepcopy(self.anno[idx])
        anchor["video"] = self.load_video(anchor["video_id"])
        
        pair_idx, similarity = self.sample_pair(idx)
        pair = copy.deepcopy(self.anno[pair_idx])
        pair["video"] = self.load_video(pair["video_id"])
        
        return anchor, pair, similarity


class ActivityNetCaptionsRetrievalEvalDataset(ActivityNetCaptionsRetrievalBaseDataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

        self._prepare_batches()

    def _prepare_batches(self):
        self.batches = []
        for i, query in enumerate(self.anno):
            batch = {
                "query_video_id": query["video_id"], 
                "trg_video_ids": [x["video_id"] for x in self.anno if query["video_id"] != x["video_id"]],
            }

            similarities = torch.cat([self.sm[i][:i], self.sm[i][i+1:]])
            similarities = torch.clamp(similarities, min=0.)
            batch["similarities"] = similarities

            self.batches.append(batch)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.batches[idx]) # batch size is always 1 for eval
        query_video = self.load_video(batch["query_video_id"])
        batch["query_video"] = query_video

        return batch