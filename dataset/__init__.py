import torch
from torch.nn.utils.rnn import pad_sequence

from dataset.moma import(
    MOMARetrievalTrainDataset,
    MOMARetrievalEvalDataset,
)


class MOMACollator(object):
    def __init__(self, is_train, model):
        self.is_train = is_train
        self.model = model

    def pad_videos(self, seq_list):
        seq_len = torch.tensor([seq.shape[0] for seq in seq_list])
        padded_seq = pad_sequence(seq_list, batch_first=True) # b x t x d
        pad_len = padded_seq.shape[1] - seq_len
        pad_mask = []
        for ns, np in zip(seq_len, pad_len):
            pad_mask.append(torch.cat([torch.zeros(ns).bool(), torch.ones(np).bool()]))
        pad_mask = torch.stack(pad_mask, dim=0)

        return padded_seq, pad_mask

    def __call__(self, data_list):
        if self.is_train:
            anchor_video_ids, pair_video_ids = [], []
            anchor_cnames, pair_cnames = [], []
            anchor_videos, pair_videos = [], []
            similarities = []
            
            for anchor, pair, s in data_list:
                anchor_video_ids.append(anchor["video_id"])
                anchor_cnames.append(anchor["cname"])
                anchor_videos.append(anchor["video"])
                
                pair_video_ids.append(pair["video_id"])
                pair_cnames.append(pair["cname"])
                pair_videos.append(pair["video"])

                similarities.append(s)

            batch = {
                "anchor_video_ids": anchor_video_ids,
                "anchor_cnames": anchor_cnames,
                "pair_video_ids": pair_video_ids,
                "pair_cnames": pair_cnames,
                "similarities": torch.stack(similarities),
            }

            # video tensors to batch
            if self.model == "ours":
                anchor_videos, anchor_pad_mask = self.pad_videos(anchor_videos)
                batch["anchor_videos"] = anchor_videos
                batch["anchor_pad_mask"] = anchor_pad_mask

                pair_videos, pair_pad_mask = self.pad_videos(pair_videos)
                batch["pair_videos"] = pair_videos
                batch["pair_pad_mask"] = pair_pad_mask
            else:
                anchor_videos = torch.stack(anchor_videos, dim=0) # b x d
                pair_videos = torch.stack(pair_videos, dim=0) # b x d

                batch["anchor_videos"] = anchor_videos
                batch["pair_videos"] = pair_videos

            return batch
        else:
            batch = data_list[0] # batch size is always 1 for val

            return batch
        
    
def get_train_dataset(cfg):
    if cfg.TRAIN.dataset == "moma":
        dataset = MOMARetrievalTrainDataset(cfg)
        collate_fn = MOMACollator(is_train=True, model=cfg.MODEL.name)
        return dataset, collate_fn
    else:
        raise NotImplementedError
    

def get_eval_dataset(cfg):
    if cfg.EVAL.dataset == "moma":
        dataset = MOMARetrievalEvalDataset(cfg, cfg.EVAL.split)
        collate_fn = MOMACollator(is_train=False, model=cfg.MODEL.name)
        return dataset, collate_fn
    else:
        raise NotImplementedError