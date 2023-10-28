import os
import ndjson
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from utils.similarity import (
    compute_cosine_mean_similarity,
    compute_smooth_chamfer_similarity,
    compute_dtw_similarity,
)
from utils.clip_sampler import (
    s3d_clip_sampler,
    frozen_clip_sampler,
)
from tqdm import tqdm


class MOMARetrievalBaseDataset(Dataset):
    def __init__(self, cfg, split):
        super().__init__()

        self.cfg = cfg
        self.args = cfg.DATASET.moma
        self.paradigm = self.args.paradigm
        self.split = split

        self._load_anno_data(split)
        self._set_pairwise_similarity()

        self.graph_cache = {}

    def _load_anno_data(self, split):
        with open(f"anno/moma/{split}.ndjson", "r") as f:
            self.anno = ndjson.load(f)
        self.id2cemb = torch.load("anno/moma/id2cemb.pt")

    def _set_pairwise_similarity(self):
        type = self.cfg.RELEVANCE.type
        if os.path.exists(f"anno/moma/sm_{type}_{self.split}.pt"):
            print(f"Load from pre-computed surrogate measure [{type}]")
            self.sm = torch.load(f"anno/moma/sm_{type}_{self.split}.pt")
            # self.sm = np.load(f"anno/moma/sm_{type}_{self.split}.npy")
            # self.sm = torch.from_numpy(self.sm).float()
        else:
            self.sm = np.zeros((len(self.anno), len(self.anno))).astype(np.float32)
            check = np.zeros((len(self.anno), len(self.anno))).astype(bool)
            for i in tqdm(range(len(self.anno)), desc=f"Compute pair-wise surrogate measure [{type}]"):
                for j in range(len(self.anno)):
                    if check[j][i]:
                        self.sm[i][j] = self.sm[j][i]
                        continue
                    if type == "mean":
                        c_i = self.id2cemb[self.anno[i]["video_id"]]
                        c_j = self.id2cemb[self.anno[j]["video_id"]]
                        self.sm[i][j] = compute_cosine_mean_similarity(c_i, c_j)
                        check[i][j] = True
                    elif type == "smooth-chamfer":
                        c_i = self.id2cemb[self.anno[i]["video_id"]]
                        c_j = self.id2cemb[self.anno[j]["video_id"]]
                        alpha = self.RELEVANCE.smooth_chamfer.alpha
                        self.sm[i][j] = compute_smooth_chamfer_similarity(c_i, c_j, alpha)
                        check[i][j] = True
                    elif type == "dtw":
                        c_i = self.id2cemb[self.anno[i]["video_id"]]
                        c_j = self.id2cemb[self.anno[j]["video_id"]]
                        self.sm[i][j] = compute_dtw_similarity(c_i, c_j)
                        check[i][j] = True
                    else:
                        raise NotImplementedError
            
            np.save(f"anno/moma/sm_{type}_{self.split}.npy", self.sm)
            self.sm = torch.from_numpy(self.sm).float()
                
    def transform_s3d(self, snippet):
        ''' stack & noralization '''
        snippet = np.concatenate(snippet, axis=-1)
        snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
        snippet = snippet.mul_(2.).sub_(255).div(255)
        return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
    
    def transform_frozen(self, snippet):
        snippet = torch.from_numpy(snippet)
        snippet = snippet.permute(0, 3, 1, 2) # n_frames x 3 x h x w
        snippet = snippet / 255.
        snippet = transforms.Resize(size=(224, 224))(snippet)
        return snippet

    def load_video(self, vid):
        if self.cfg.MODEL.VIDEO.name == "s3d":
            s3d_args = self.cfg.MODEL.VIDEO.S3D
            if s3d_args.freeze:
                path = os.path.join(self.args.path, "preprocessed", "s3d", f"{vid}.npy")
                feat = torch.from_numpy(np.load(path)) # d,
                return feat
            else:
                path = os.path.join(self.args.path, "videos", "raw", f"{vid}.mp4")
                clip_duration = s3d_args.clip_duration
                frames_per_clip = s3d_args.frames_per_clip
                stride = s3d_args.stride

                video = []
                sampled_clips = s3d_clip_sampler(path, clip_duration, frames_per_clip, stride)
                for clip in sampled_clips:
                    # clip: list of n_frames x h x w x 3
                    clip = self.transform_s3d(clip) # 1 x 3 x n_frames x h x w
                    video.append(clip)

                return torch.cat(video, dim=0) # n_clips x 3 x n_frames x h x w
        elif self.cfg.MODEL.VIDEO.name == "frozen":
            frozen_args = self.cfg.MODEL.VIDEO.FROZEN
            if frozen_args.freeze:
                path = os.path.join(self.args.path, "preprocessed", "frozen", f"{vid}.npy")
                feat = torch.from_numpy(np.load(path)) # d,
                return feat
            else:
                path = os.path.join(self.args.path, "videos", "raw", f"{vid}.mp4")
                clip_duration = self.frozen_args.clip_duration
                num_frames = self.frozen_args.num_frames

                video = []
                sampled_clips = frozen_clip_sampler(path, clip_duration, num_frames)
                for clip in sampled_clips:
                    # clip: n_frames(=4) x h x w x 3
                    clip = self.transform_frozen(clip) # n_frames(=4) x 3 x 224 x 224
                    video.append(clip)

                return torch.stack(video, dim=0) # n_clips x n_frames(=4) x 3 x 224 x 224
        elif self.cfg.MODEL.VIDEO.name == "slot":
            path = os.path.join(self.args.path, "feats", "frozen", f"{vid}.npy")
            feat = torch.from_numpy(np.load(path)).float() # n_clips x n_patches (785) x d
            return feat[:,0,:] # only CLS token
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.anno)