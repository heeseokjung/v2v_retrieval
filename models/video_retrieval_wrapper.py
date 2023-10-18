import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.similarity import (
    cosine_mean_similarity,
    smooth_chamfer_similarity,
    smooth_chamfer_train,
)

from utils.loss import compute_mse_loss
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
        self.eval_relevance_scores = []
        self.ndcg_metric = nDCGMetric([5, 10, 20, 40])
        self.mse_error = MSEError()

        self.id2cembs_train = torch.load("dataset/anno/moma/id2cembs_train.pt")
    
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
        anchor_activity_ids = batch["anchor_activity_ids"]
        anchor_activity_names = batch["anchor_activity_names"]
        anchor_videos = batch["anchor_videos"]
        
        pair_video_ids = batch["pair_video_ids"]
        pair_activity_ids = batch["pair_activity_ids"]
        pair_activity_names = batch["pair_activity_names"]
        pair_videos = batch["pair_videos"]
        
        relevance_scores = batch["relevance_scores"]

        if self.cfg.MODEL.VIDEO.name == "slot":
            anchor_video_embs = []
            for video in anchor_videos:
                anchor_video_embs.append(self(video))

            pair_video_embs = []
            for video in pair_videos:
                pair_video_embs.append(self(video))

            pred, vt_algin_loss = [], []
            for anchor_vid, anchor_emb, pair_vid, pair_emb in zip(
                anchor_video_ids, anchor_video_embs, pair_video_ids, pair_video_embs
            ):
                if self.cfg.RELEVANCE.type == "mean":
                    pred.append(cosine_mean_similarity(anchor_emb, pair_emb))
                elif self.cfg.RELEVANCE.type == "smooth-chamfer":
                    alpha = self.cfg.RELEVANCE.smooth_chamfer.alpha
                    pred.append(smooth_chamfer_similarity(anchor_emb, pair_emb, alpha))
                elif self.cfg.RELEVANCE.type == "dtw":
                    pred.append(cosine_mean_similarity(anchor_emb, pair_emb))

                # alpha = self.cfg.RELEVANCE.smooth_chamfer.alpha
                # vt_algin_loss.append(
                #     smooth_chamfer_train(anchor_emb, self.id2cembs_train[anchor_vid].to("cuda"), alpha)
                # )
                # vt_algin_loss.append(
                #     smooth_chamfer_train(pair_emb, self.id2cembs_train[pair_vid].to("cuda"), alpha)
                # )

            pred = torch.stack(pred)
            # vt_algin_loss = -1. * F.relu(torch.stack(vt_algin_loss).mean())
            # mse_loss = compute_mse_loss(pred, relevance_scores)
            # loss = mse_loss + 0.01*vt_algin_loss
            loss = compute_mse_loss(pred, relevance_scores)

            self.log(
                "train/loss",
                loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            ) 

            # self.log(
            #     "train/mse_loss",
            #     mse_loss,
            #     on_step=True,
            #     prog_bar=True,
            #     logger=True,
            #     sync_dist=True,
            # )

            # self.log(
            #     "train/vt_align_loss",
            #     0.5*vt_algin_loss,
            #     on_step=True,
            #     prog_bar=True,
            #     logger=True,
            #     sync_dist=True,
            # )
        else:
           anchor_video_embs = self(anchor_videos)
           pair_video_embs = self(pair_videos)

           anchor_video_embs = F.normalize(anchor_video_embs, dim=-1) # b x d
           pair_video_embs = F.normalize(pair_video_embs, dim=-1) # b x d
           pred = torch.mm(anchor_video_embs, pair_video_embs.t()).diagonal()

           loss = compute_mse_loss(pred, relevance_scores)
           
        return loss
    
    # for caching embeddings
    def val_shared_step(self, vid, data):
        if vid in self.id2emb:
            return self.id2emb[vid]
        else:
            emb = self(data) # d or n_slot x d
            # emb = emb.to("cpu")
            self.id2emb[vid] = emb
            return emb

    def validation_step(self, batch, batch_idx):
        query_video_id = batch["query_video_id"]
        query_activity_name = batch["query_activity_name"]
        query_video = batch["query_video"]

        ref_video_ids = batch["ref_video_ids"]
        ref_activity_names = batch["ref_activity_names"]
        if "ref_videos" in batch:
            ref_videos = batch["ref_videos"]

        relevance_scores = batch["relevance_scores"]

        if self.cfg.MODEL.VIDEO.name == "slot":
            self.eval_query_vids.append(query_video_id)
            self.eval_ref_vids.append(ref_video_ids)
            self.eval_relevance_scores.append(relevance_scores)
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

            self.ndcg_metric.update(pred, relevance_scores)
            self.mse_error.update(pred, relevance_scores)
        
    def validation_epoch_end(self, validation_step_outputs):
        if self.cfg.MODEL.VIDEO.name == "slot":
            for query_vid, ref_vids, relevance_scores in zip(
                self.eval_query_vids, self.eval_ref_vids, self.eval_relevance_scores
            ):
                pred = []
                query_video_emb = self.id2emb[query_vid]
                ref_video_embs = [self.id2emb[vid] for vid in ref_vids]
                for ref_emb in ref_video_embs:
                    if self.cfg.RELEVANCE.type == "mean":
                        pred.append(cosine_mean_similarity(query_video_emb, ref_emb))
                    elif self.cfg.RELEVANCE.type == "smooth-chamfer":
                        alpha = self.RELEVANCE.smooth_chamfer.alpha
                        pred.append(smooth_chamfer_similarity(query_video_emb, ref_emb, alpha))
                    elif self.cfg.RELEVANCE.type == "dtw":
                        pred.append(cosine_mean_similarity(query_video_emb, ref_emb))
                pred = torch.stack(pred)

                self.ndcg_metric.update(pred, relevance_scores)
                self.mse_error.update(pred, relevance_scores)

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
        self.eval_relevance_scores = []
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
        