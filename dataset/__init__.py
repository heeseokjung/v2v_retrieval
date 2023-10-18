import torch
import torch.nn.functional as F

from dataset.moma import(
    MOMARetrievalTrainDataset,
    MOMARetrievalEvalDataset,
)
from dataset.ag import (
    ActionGenomeRetrievalEvalDataset
)


class MOMACollator(object):
    def __init__(self, is_train, model):
        self.is_train = is_train
        self.model = model

    def __call__(self, data_list):
        if self.is_train:
            anchor_video_ids, pair_video_ids = [], []
            anchor_activity_ids, pair_activiy_ids = [], []
            anchor_activity_names, pair_activity_names = [], []
            anchor_videos, pair_videos = [], []
            relevance_scores = []
            
            for anchor, pair, score in data_list:
                anchor_video_ids.append(anchor["video_id"])
                anchor_activity_ids.append(anchor["activity_id"])
                anchor_activity_names.append(anchor["activity_name"])
                anchor_videos.append(anchor["video"])
                
                pair_video_ids.append(pair["video_id"])
                pair_activiy_ids.append(pair["activity_id"])
                pair_activity_names.append(pair["activity_name"])
                pair_videos.append(pair["video"])

                relevance_scores.append(score)

            if self.model != "slot":
                anchor_videos = torch.stack(anchor_videos, dim=0)
                pair_videos = torch.stack(pair_videos, dim=0)

            batch = {
                "anchor_video_ids": anchor_video_ids,
                "anchor_activity_ids": anchor_activity_ids,
                "anchor_activity_names": anchor_activity_names,
                "anchor_videos": anchor_videos, 
                "pair_video_ids": pair_video_ids,
                "pair_activity_ids": pair_activiy_ids,
                "pair_activity_names": pair_activity_names,
                "pair_videos": pair_videos,
                "relevance_scores": torch.stack(relevance_scores),
            }

            return batch
        else:
            batch = data_list[0] # batch size is always 1 for val
            return batch
        

def ag_train_collate_fn(data_list):
    ...
    
    
def ag_test_collate_fn(data_list):
    batch = data_list[0] # batch size is always 1 for val.
    return batch
    
    
def get_train_dataset(cfg):
    if cfg.TRAIN.dataset == "moma":
        dataset = MOMARetrievalTrainDataset(cfg)
        collate_fn = MOMACollator(is_train=True, model=cfg.MODEL.VIDEO.name)
        return dataset, collate_fn
    elif cfg.TRAIN.dataset == "ag":
        raise NotImplementedError
    else:
        raise NotImplementedError
    

def get_eval_dataset(cfg):
    if cfg.EVAL.dataset == "moma":
        dataset = MOMARetrievalEvalDataset(cfg)
        collate_fn = MOMACollator(is_train=False, model=cfg.MODEL.VIDEO.name)
        return dataset, collate_fn
    elif cfg.EVAL.dataset == "ag":
        raise NotImplementedError
    else:
        raise NotImplementedError