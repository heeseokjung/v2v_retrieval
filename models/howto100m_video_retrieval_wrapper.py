import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

from utils.similarity import (
    compute_cosine_mean_similarity,
    compute_smooth_chamfer_similarity,
    smooth_chamfer_train,
)

from utils.loss import compute_mse_loss
from tslearn.metrics import SoftDTWLossPyTorch
from utils.metric import nDCGMetric
from utils.metric import MSEError
from tqdm import tqdm


def cdist(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1. - torch.bmm(x, y.transpose(-2, -1))


def compute_ndcg_single(pred, sm):
    topk = [5, 10, 20, 40]
    _, pred_idx = torch.topk(pred, max(topk))
    _, opt_idx = torch.topk(sm, max(topk))

    scores = {}
    for k in topk:
        pred_rel = sm[pred_idx[:k]]
        opt_rel = sm[opt_idx[:k]]
        dcg = ((2**pred_rel - 1) / torch.log2(torch.arange(2, k+2)).to("cuda")).sum()
        idcg = ((2**opt_rel - 1) / torch.log2(torch.arange(2, k+2)).to("cuda")).sum()
        scores[f"nDCG@{k}"] = dcg / idcg

    return scores


def compute_mse_single(pred, sm):
    diff = pred - sm
    return (diff ** 2).mean()


def visualize_result(
        path,
        model, 
        pred,
        sm, 
        query_vid,
        query_cname,
        trg_cnames, 
        topk=None
    ):

    ndcg = compute_ndcg_single(pred, sm)
    mse = compute_mse_single(pred, sm)

    if topk is not None:
        pred = pred[:topk]
        sm = sm[:topk]
        trg_cnames[:topk]

    _, sorted_idx = torch.sort(sm, descending=True)
    sm_y = sm[sorted_idx].detach().cpu().numpy()
    pred_y = pred[sorted_idx].detach().cpu().numpy()
    x_label = [trg_cnames[idx] for idx in sorted_idx]

    plt.figure(figsize=(30, 5))
    plt.title((
        f"{query_vid} [{query_cname}] "
        f"nDCG@5: {ndcg['nDCG@5']} "
        f"nDCG@10: {ndcg['nDCG@10']} "
        f"nDCG@20: {ndcg['nDCG@20']} "
        f"nDCG@40: {ndcg['nDCG@40']} "
        f"mse: {mse}"
    ))
    plt.xticks(np.arange(len(x_label)), label=x_label, rotation=90)
    plt.gca().set_xticklabels(x_label)
    plt.xlim([0, len(trg_cnames)])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.tight_layout()

    plt.plot(np.arange(len(sm)), sm_y, color="red")
    plt.plot(np.arange(len(pred)), pred_y, color="black")
    plt.savefig(os.path.join(path, model,  f"{query_vid}.png"))
    plt.close()


class HowTo100MVideoRetrievalWrapper(pl.LightningModule):
    def __init__(self, cfg, video_encoder):
        super().__init__()
        
        self.cfg = cfg
        self.video_encoder = video_encoder

        # loss functions
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.sdtw_loss = SoftDTWLossPyTorch(gamma=0.1, normalize=True, dist_func=cdist)
        
        # used in eval step
        self.id2vemb = {}
        self.eval_query_vids = []
        self.eval_query_cnames = []
        self.eval_trg_vids = []
        self.eval_trg_cnames = []
        self.eval_similarities = []
        self.ndcg_metric = nDCGMetric([5, 10, 20, 40])
        self.mse_error = MSEError()

        # tmp for smooth-chamfer
        self.id2cemb = torch.load("anno/howto100m/id2cemb.pt")
    
    def forward(self, x, pad_mask=None):
        if self.cfg.MODEL.name == "ours":
            return self.video_encoder(x, pad_mask) # b x k x d
        else:
            return self.video_encoder(x) # b x d
    
    def training_step(self, batch, batch_idx):
        # get anchor data
        anchor_video_ids = batch["anchor_video_ids"]
        anchor_c1names = batch["anchor_c1names"]
        anchor_c2names = batch["anchor_c2names"] 
        anchor_videos = batch["anchor_videos"]
        if "anchor_pad_mask" in batch:
            anchor_pad_mask = batch["anchor_pad_mask"]
        
        # get pair data
        pair_video_ids = batch["pair_video_ids"]
        pair_c1names = batch["pair_c1names"]
        pair_c2names = batch["pair_c2names"] 
        pair_videos = batch["pair_videos"]
        if "pair_pad_mask" in batch:
            pair_pad_mask = batch["pair_pad_mask"]
        
        # semantic similarities (surrogate measure)
        similarities = batch["similarities"]

        if self.cfg.MODEL.name == "ours": # proposed
            anchor_embs = self(anchor_videos, anchor_pad_mask) # b x k x d
            pair_embs = self(pair_videos, pair_pad_mask) # b x k x d

            if self.cfg.RELEVANCE.type == "cosine":
                anchor_embs = F.normalize(anchor_embs, dim=-1)
                pair_embs = F.normalize(pair_embs, dim=-1)
                pred_sim = torch.bmm(anchor_embs, pair_embs.transpose(1, 2)) # b x k x k
                pred = pred_sim.mean(dim=(1, 2)) # b,
            elif self.cfg.RELEVANCE.type == "dtw":
                k = anchor_embs.shape[1]
                pred = 1. - (self.sdtw_loss(anchor_embs, pair_embs) / 2*k) 
            else:
                raise NotImplementedError
            
            # surrogate predict loss
            sm_loss = self.mse_loss(pred, similarities)

            # smooth-chamfer similarity loss
            sc_loss = []
            alpha = self.cfg.RELEVANCE.smooth_chamfer.alpha
            for vid, emb in zip(anchor_video_ids, anchor_embs):
                cemb = torch.from_numpy(self.id2cemb[vid]).float().to(emb.device)
                sc_loss.append(-1. * compute_smooth_chamfer_similarity(emb, cemb, alpha))
            for vid, emb in zip(pair_video_ids, pair_embs):
                cemb = torch.from_numpy(self.id2cemb[vid]).float().to(emb.device)
                sc_loss.append(-1. * compute_smooth_chamfer_similarity(emb, cemb, alpha))
            sc_loss = torch.stack(sc_loss)
            sc_loss = sc_loss.mean()

            # total loss
            gamma = self.cfg.TRAIN.loss.gamma
            # loss = sm_loss
            loss = sm_loss + gamma*sc_loss

            self.log(
                "train/loss",
                loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            ) 

            self.log(
                "train/sm_loss",
                sm_loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            ) 

            self.log(
                "train/sc_loss",
                gamma*sc_loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            ) 
        else: # baseline
           anchor_embs = self(anchor_videos) # b x d
           pair_embs = self(pair_videos) # b x d

           anchor_embs = F.normalize(anchor_embs, dim=-1) 
           pair_embs = F.normalize(pair_embs, dim=-1) 
           pred = torch.mm(anchor_embs, pair_embs.t()).diagonal() # b,

           loss = self.mse_loss(pred, similarities)

           self.log(
                "train/loss",
                loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            ) 

        return loss
    
    # for caching embeddings
    def val_shared_step(self, vid, x):
        if vid in self.id2vemb:
            return self.id2vemb[vid]
        else:
            x = x.unsqueeze(dim=0) # 1 x n x d or 1 x d
            emb = self(x, pad_mask=None).squeeze(dim=0) # k x d or d
            self.id2vemb[vid] = emb
            return emb

    def validation_step(self, batch, batch_idx):
        # get query data
        query_video_id = batch["query_video_id"]
        query_c1name = batch["query_c1name"]
        query_c2name = batch["query_c2name"]
        query_video = batch["query_video"]
        
        # get target data
        trg_video_ids = batch["trg_video_ids"]
        trg_c1names = batch["trg_c1names"] 
        trg_c2names = batch["trg_c2names"]

        # semantic similarities (surrogate measure)
        similarities = batch["similarities"]
        
        # query_emb = self.val_shared_step(query_video_id, query_video) # save in cache
        if self.cfg.MODEL.name == "ours":
            query_video = query_video.unsqueeze(dim=0) # 1 x n x d or 1 x d
            emb = self(query_video, pad_mask=None).squeeze(dim=0) # k x d or d
            emb = emb.detach().cpu().numpy()
            if self.current_epoch % 10 == 0 or (self.current_epoch == self.trainer.max_epochs - 1):
                os.makedirs(os.path.join(self.cfg.DATASET.howto100m.path, "out", str(self.current_epoch)), exist_ok=True)
                np.save(
                    os.path.join(self.cfg.DATASET.howto100m.path, "out", str(self.current_epoch), f"{query_video_id}.npy"),
                    emb,
                )

        self.eval_query_vids.append(query_video_id)
        self.eval_query_cnames.append(query_c1name)
        self.eval_trg_vids.append(trg_video_ids)
        self.eval_trg_cnames.append(trg_c1names)
        self.eval_similarities.append(similarities)
        
    def validation_epoch_end(self, validation_step_outputs):
        for query_vid, query_cname, trg_vids, trg_cnames, similarities in zip(
            self.eval_query_vids,
            self.eval_query_cnames,
            self.eval_trg_vids,
            self.eval_trg_cnames,
            self.eval_similarities,
        ):
            if self.cfg.MODEL.name == "ours": # proposed
                ...
            else: # baseline
                query_emb = self.id2vemb[query_vid]
                query_emb = F.normalize(query_emb, dim=-1)
                trg_embs = [self.id2vemb[vid] for vid in trg_vids]
                trg_embs = torch.stack(trg_embs, dim=0) # b x k x d or b x d
                trg_embs = F.normalize(trg_embs, dim=-1)
                pred = torch.matmul(query_emb, trg_embs.t())

            # visualize_result(
            #     path="/root/visualize_results",
            #     model=self.cfg.MODEL.name,
            #     pred=pred,
            #     sm=similarities,
            #     query_vid=query_vid,
            #     query_cname=query_cname,
            #     trg_cnames=trg_cnames, 
            # )

        #     self.ndcg_metric.update(pred, similarities)
        #     self.mse_error.update(pred, similarities)

        # score = self.ndcg_metric.compute()
        # score["mse_error"] = self.mse_error.compute()
        
        # for k, v in score.items():
        #     self.log(
        #         f"val/{k}",
        #         v,
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True,
        #         sync_dist=True,
        #     )
            
        # self.print(f"Test score: {score}")
        
        self.id2vemb = {}
        self.eval_query_vids = []
        self.eval_query_cnames = []
        self.eval_trg_vids = []
        self.eval_trg_cnames = []
        self.eval_similarities = []
        self.ndcg_metric.reset()
        self.mse_error.reset()
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        
    def test_epoch_end(self, test_step_ouptuts):
        return self.validation_epoch_end(test_step_ouptuts)
        
    def configure_optimizers(self):
        if self.cfg.TRAIN.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.cfg.TRAIN.lr
            )
            
        return optimizer