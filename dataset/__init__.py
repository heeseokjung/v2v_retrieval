import torch

from dataset.moma import(
    MOMARetrievalTrainDataset,
    MOMARetrievalEvalDataset,
)


class MOMACollator(object):
    def __init__(self, is_train, model):
        self.is_train = is_train
        self.model = model

    def __call__(self, data_list):
        if self.is_train:
            anchor_video_ids, pair_video_ids = [], []
            anchor_cnames, pair_cnames = [], []
            anchor_videos, pair_videos = [], []
            sm_l = []
            
            for anchor, pair, sm in data_list:
                anchor_video_ids.append(anchor["video_id"])
                anchor_cnames.append(anchor["cname"])
                anchor_videos.append(anchor["video"])
                
                pair_video_ids.append(pair["video_id"])
                pair_cnames.append(pair["cname"])
                pair_videos.append(pair["video"])

                sm_l.append(sm)

            if self.model != "slot":
                anchor_videos = torch.stack(anchor_videos, dim=0)
                pair_videos = torch.stack(pair_videos, dim=0)

            batch = {
                "anchor_video_ids": anchor_video_ids,
                "anchor_cnames": anchor_cnames,
                "anchor_videos": anchor_videos, 
                "pair_video_ids": pair_video_ids,
                "pair_cnames": pair_cnames,
                "pair_videos": pair_videos,
                "sm": torch.stack(sm_l),
            }

            return batch
        else:
            batch = data_list[0] # batch size is always 1 for val
            return batch
        
    
def get_train_dataset(cfg):
    if cfg.TRAIN.dataset == "moma":
        dataset = MOMARetrievalTrainDataset(cfg)
        collate_fn = MOMACollator(is_train=True, model=cfg.MODEL.VIDEO.name)
        return dataset, collate_fn
    else:
        raise NotImplementedError
    

def get_eval_dataset(cfg):
    if cfg.EVAL.dataset == "moma":
        dataset = MOMARetrievalEvalDataset(cfg, cfg.EVAL.split)
        collate_fn = MOMACollator(is_train=False, model=cfg.MODEL.VIDEO.name)
        return dataset, collate_fn
    else:
        raise NotImplementedError