import copy
import torch
import numpy as np
from dataset.moma_base import MOMARetrievalBaseDataset


class MOMARetrievalTrainDataset(MOMARetrievalBaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg, "train")
        
    def sample_pair(self, idx):
        scores = self.sm[idx]
        _, sorted_idx = torch.sort(scores, descending=True)
        sorted_idx = sorted_idx[1:] # exclude self
        
        if self.args.use_over_sampling and np.random.rand(1) < 0.5:
            pair_idx = sorted_idx[np.random.randint(self.args.tail_range)]
        else:
            pair_idx = sorted_idx[np.random.randint(len(sorted_idx))]
            
        return pair_idx, scores[pair_idx]
        
    def __getitem__(self, idx):
        anchor_anno = copy.deepcopy(self.anno[idx])
        anchor_anno["video"] = self.load_video(anchor_anno["video_id"])
        
        pair_idx, score = self.sample_pair(idx)
        pair_anno = copy.deepcopy(self.anno[pair_idx])
        pair_anno["video"] = self.load_video(pair_anno["video_id"])
        
        return anchor_anno, pair_anno, score
        
        
class MOMARetrievalEvalDataset(MOMARetrievalBaseDataset):
    def __init__(self, cfg, split):
        if split == "val": 
            super().__init__(cfg, "val")
        elif split == "test":
            super().__init__(cfg, "test")

        self._prepare_batches()
        
    def _prepare_batches(self):
        self.batches = []
        for i, query in enumerate(self.anno):
            batch = {
                "query_video_id": query["video_id"], # video id e.g. '-49z-lj8eYQ'
                "query_activity_name": query["activity_name"], # activity name e.g. "basketball game"
                "ref_video_ids": [x["video_id"] for x in self.anno if query["video_id"] != x["video_id"]],
                "ref_activity_names": [x["activity_name"] for x in self.anno if query["video_id"] != x["video_id"]],
            }

            sm = torch.cat([self.sm[i][:i], self.sm[i][i+1:]])
            # relevance_scores = torch.clamp(relevance_scores, min=0.)
            batch["sm"] = sm
            
            self.batches.append(batch)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.batches[idx]) # batch size is always 1 for eval
        
        query_video = self.load_video(batch["query_video_id"])
        batch["query_video"] = query_video

        if self.cfg.MODEL.VIDEO.name != "slot":
            ref_videos = [self.load_video(ref_vid) for ref_vid in batch["ref_video_ids"]]
            ref_videos = torch.stack(ref_videos, dim=0) # only when use pre-extracted features
            batch["ref_videos"] = ref_videos
        
        return batch