import os
import ndjson
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class nDCGMetric:
    def __init__(self, topK):
        self.topK = topK
        self.score = {f"nDCG@{k}": [] for k in self.topK}
            
    def update(self, pred, proxy):
        _, pred_idx = torch.topk(pred, max(self.topK))
        _, opt_idx = torch.topk(proxy, max(self.topK))

        # proxy = torch.clamp(proxy, min=0.5)
        # proxy = (proxy - proxy.min()) / (proxy.max() - proxy.min())

        for k in self.topK:
            pred_rel = proxy[pred_idx[:k]]
            opt_rel = proxy[opt_idx[:k]]
            
            dcg = ((2**pred_rel - 1) / torch.log2(torch.arange(2, k+2)).to("cuda")).sum()
            idcg = ((2**opt_rel - 1) / torch.log2(torch.arange(2, k+2)).to("cuda")).sum()
            
            self.score[f"nDCG@{k}"].append(dcg / idcg)
        
    def compute(self):
        return {
            k: torch.tensor(v).mean() for k, v in self.score.items()
        }
        
    def reset(self):
        self.score = {f"nDCG@{k}": [] for k in self.topK}
        

class MSEError:
    def __init__(self):
        self.mse_log = []
        
    def update(self, pred_similarities, proxy_similarities):
        diff = pred_similarities - proxy_similarities
        self.mse_log.append((diff ** 2).mean())
        
    def compute(self):
        return torch.tensor(self.mse_log).mean()
        
    def reset(self):
        self.mse_log = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val") 
    args = parser.parse_args()
    split = args.split

    ndcg_metric = nDCGMetric([5, 10, 20, 40])
    mse_error = MSEError()

    with open(f"anno/howto100m/{split}.ndjson", "r") as f:
        anno = ndjson.load(f)
    video_ids = [x["video_id"] for x in anno]

    sm = np.load(f"anno/howto100m/sm_dtw_{split}.npy")
    sm = torch.from_numpy(sm).float().to("cuda")

    id2vemb = {}
    path = "/data/dir_howto100m/feats/s3d"
    for vid in video_ids:
        emb = np.load(os.path.join(path, f"{vid}.npy"))
        emb = emb.mean(axis=0)
        emb = torch.from_numpy(emb).float().to("cuda")
        id2vemb[vid] = emb

    for i, qvid in enumerate(tqdm(video_ids, desc="[EVAL S3D w/o FT]")):
        query_emb = id2vemb[qvid] # d,
        trg_video_ids = [vid for vid in video_ids if qvid != vid]
        trg_embs = [id2vemb[vid] for vid in trg_video_ids]
        trg_embs = torch.stack(trg_embs, dim=0) # (N-1) x d
        similarities = torch.cat([sm[i][:i], sm[i][i+1:]])
        similarities = torch.clamp(similarities, min=0.)

        pred = F.cosine_similarity(query_emb, trg_embs)

        ndcg_metric.update(pred, similarities)
        mse_error.update(pred, similarities)

    score = ndcg_metric.compute()
    score["mse_error"] = mse_error.compute()

    print(f"< HowTo100M - S3D w/o no fine-tuning >")
    for k, v in score.items():
        print(f"[{k}]: {v}")


if __name__ == "__main__":
    main()