import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.similarity import (
    compute_cosine_mean_similarity,
    compute_smooth_chamfer_similarity,
    smooth_chamfer_train,
)

from utils.loss import compute_mse_loss
from tslearn.metrics import SoftDTWLossPyTorch
from utils.metric import nDCGMetric
from utils.metric import MSEError


class VideoRetrievalWrapper(pl.LightningModule):
    def __init__(self, cfg, video_encoder, slot_encoder):
        super().__init__()
        
        self.cfg = cfg
        
        self.video_encoder = video_encoder
        self.slot_encoder = slot_encoder
        self.layer_norm = nn.LayerNorm(cfg.MODEL.SLOT.slot_size)
        
        self.id2emb = {}
        self.eval_query_vids = []
        self.eval_ref_vids = []
        self.eval_sm_l = []
        self.ndcg_metric = nDCGMetric([5, 10, 20, 40])
        self.mse_error = MSEError()

        self.id2cemb = torch.load("anno/moma/id2cemb.pt")

        self.soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)
    
    def forward(self, data):
        if self.cfg.MODEL.VIDEO.name == "slot":
            # data: n_clips x d (CLS 토큰들의 시퀀스)
            x = self.video_encoder(data) # 트랜스포머 인코더 한번 태우는거
            slots = self.slot_encoder(x) # n_slot x d
            return slots
        else:
            return self.video_encoder(data) # b x d
    
    def training_step(self, batch, batch_idx):
        anchor_video_ids = batch["anchor_video_ids"]
        # anchor_cnames = batch["anchor_cnames"] # dataset dependent field
        anchor_videos = batch["anchor_videos"]
        
        pair_video_ids = batch["pair_video_ids"]
        # pair_cnames = batch["pair_cnames"] # dataset dependent field
        pair_videos = batch["pair_videos"]
        
        sm = batch["sm"]

        if self.cfg.MODEL.VIDEO.name == "slot":
            anchor_video_embs = []
            for video in anchor_videos:
                emb = self(video)
                anchor_video_embs.append(emb)

            pair_video_embs = []
            for video in pair_videos:
                emb = self(video)
                pair_video_embs.append(emb)

            anchor_video_embs = torch.stack(anchor_video_embs, dim=0)
            pair_video_embs = torch.stack(pair_video_embs, dim=0)

            pred, vt_align_loss, cov = [], [], []
            for anchor_vid, anchor_emb, pair_vid, pair_emb in zip(
                anchor_video_ids, anchor_video_embs, pair_video_ids, pair_video_embs,
            ):
                if self.cfg.RELEVANCE.type == "mean":
                    pred.append(compute_cosine_mean_similarity(anchor_emb, pair_emb))
                elif self.cfg.RELEVANCE.type == "smooth-chamfer":
                    alpha = self.cfg.RELEVANCE.smooth_chamfer.alpha
                    pred.append(compute_smooth_chamfer_similarity(anchor_emb, pair_emb, alpha))
                elif self.cfg.RELEVANCE.type == "dtw":
                    pred.append(compute_cosine_mean_similarity(anchor_emb, pair_emb))

                # for Smooth-Chamfer loss (L2)
                alpha = self.cfg.RELEVANCE.smooth_chamfer.alpha
                vt_align_loss.append(
                    -1. * smooth_chamfer_train(anchor_emb, self.id2cemb[anchor_vid].to("cuda"), alpha)
                )
                vt_align_loss.append(
                    -1. * smooth_chamfer_train(pair_emb, self.id2cemb[pair_vid].to("cuda"), alpha)
                )

                # for covariance regularizer
                anchor_emb = F.normalize(anchor_emb, dim=-1)
                cov.append(torch.mm(anchor_emb, anchor_emb.t()))
                pair_emb = F.normalize(pair_emb, dim=-1)
                cov.append(torch.mm(pair_emb, pair_emb.t()))

            # predict surrogate measure (L1)
            pred = torch.stack(pred)
            mse_loss = compute_mse_loss(pred, sm)

            # variance regularizer (L3)
            cov = torch.stack(cov, dim=0) # b x k x k
            target_var = torch.eye(cov.shape[1]).unsqueeze(dim=0).expand(cov.shape[0], cov.shape[1], cov.shape[2]).to(cov.device)
            cov_reg = compute_mse_loss(cov, target_var)

            # video-text align loss
            vt_align_loss = torch.stack(vt_align_loss).mean()

            # soft-DTW loss
            soft_dtw_loss = self.soft_dtw_loss(anchor_video_embs, pair_video_embs).mean()

            loss = mse_loss + 0.05*cov_reg + 0.1*vt_align_loss + 0.0001 * soft_dtw_loss
            
            # vt_algin_loss = -1. * F.relu(torch.stack(vt_algin_loss).mean())
            # mse_loss = compute_mse_loss(pred, relevance_scores)
            # loss = mse_loss + 0.01*vt_algin_loss
            # loss = compute_mse_loss(pred, relevance_scores)

            self.log(
                "train/loss",
                loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            ) 

            self.log(
                "train/mse_loss",
                mse_loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            self.log(
                "train/vt_align_loss",
                0.1*vt_align_loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            self.log(
                "train/cov_reg",
                0.05*cov_reg,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            self.log(
                "train/soft_dtw",
                0.0001*soft_dtw_loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        else:
           anchor_video_embs = self(anchor_videos)
           pair_video_embs = self(pair_videos)

           anchor_video_embs = F.normalize(anchor_video_embs, dim=-1) # b x d
           pair_video_embs = F.normalize(pair_video_embs, dim=-1) # b x d
           pred = torch.mm(anchor_video_embs, pair_video_embs.t()).diagonal()

           loss = compute_mse_loss(pred, sm)

        return loss
    
    # for caching embeddings
    def val_shared_step(self, vid, data):
        if vid in self.id2emb:
            return self.id2emb[vid]
        else:
            emb = self(data)
            # emb = emb.to("cpu")
            self.id2emb[vid] = emb
            return emb

    def validation_step(self, batch, batch_idx):
        query_video_id = batch["query_video_id"]
        # query_cname = batch["query_cname"] # dataset dependent field
        query_video = batch["query_video"]

        ref_video_ids = batch["ref_video_ids"]
        # ref_cnames = batch["ref_cnames"] # dataset dependent field
        if "ref_videos" in batch:
            ref_videos = batch["ref_videos"]

        sm = batch["sm"]

        if self.cfg.MODEL.VIDEO.name == "slot":
            self.eval_query_vids.append(query_video_id)
            self.eval_ref_vids.append(ref_video_ids)
            self.eval_sm_l.append(sm)
            query_video_emb = self.val_shared_step(query_video_id, query_video)  
        else:
            query_video_emb = self.val_shared_step(query_video_id, query_video)
            ref_video_embs = []
            for vid, data in zip(ref_video_ids, ref_videos):
                ref_video_embs.append(self.val_shared_step(vid, data))
            ref_video_embs = torch.stack(ref_video_embs, dim=0)

            query_video_emb = F.normalize(query_video_emb, dim=-1)
            ref_video_embs = F.normalize(ref_video_embs, dim=-1)
            pred = torch.matmul(query_video_emb, ref_video_embs.t())

            self.ndcg_metric.update(pred, sm)
            self.mse_error.update(pred, sm)
        
    def validation_epoch_end(self, validation_step_outputs):
        if self.cfg.MODEL.VIDEO.name == "slot":
            for query_vid, ref_vids, sm in zip(
                self.eval_query_vids, self.eval_ref_vids, self.eval_sm_l
            ):
                pred = []
                query_video_emb = self.id2emb[query_vid]
                ref_video_embs = [self.id2emb[vid] for vid in ref_vids]
                for ref_emb in ref_video_embs:
                    if self.cfg.RELEVANCE.type == "mean":
                        pred.append(compute_cosine_mean_similarity(query_video_emb, ref_emb))
                    elif self.cfg.RELEVANCE.type == "smooth-chamfer":
                        alpha = self.RELEVANCE.smooth_chamfer.alpha
                        pred.append(compute_smooth_chamfer_similarity(query_video_emb, ref_emb, alpha))
                    elif self.cfg.RELEVANCE.type == "dtw":
                        pred.append(compute_cosine_mean_similarity(query_video_emb, ref_emb))
                pred = torch.stack(pred)

                self.ndcg_metric.update(pred, sm)
                self.mse_error.update(pred, sm)

        score = self.ndcg_metric.compute()
        score["mse_error"] = self.mse_error.compute()
        
        for k, v in score.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            
        self.print(f"Test score: {score}")
        
        self.id2emb = {}
        self.eval_query_vids = []
        self.eval_ref_vids = []
        self.eval_sm_l = []
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