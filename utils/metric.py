import torch
import torch.nn.functional as F


class nDCGMetric:
    def __init__(self, topK):
        self.topK = topK
        self.score = {f"nDCG@{k}": [] for k in self.topK}
            
    def update(self, pred, proxy):
        _, pred_idx = torch.topk(pred, max(self.topK))
        _, opt_idx = torch.topk(proxy, max(self.topK))
        
        ############## nDCG normalize ################
        # proxy = (proxy - 0.2) /  0.2
        ##############################################

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