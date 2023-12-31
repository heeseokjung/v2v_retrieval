import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.attr_dict import AttrDict
from utils.logger import set_up_logger

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from dataset import (
    get_train_dataset,
    get_eval_dataset,
)
from torch.utils.data import DataLoader

# Video encoder models
from models.s3d import (
    S3D,
    S3DProjector,
)
from models.frozen import (
    FrozenInTime,
    FrozenInTimeProjector,
)
from utils.frozen_utils import state_dict_data_parallel_fix
from models.slot import TemporalSlotAttentionVideoEncoder

from models.moma_video_retrieval_wrapper import MOMAVideoRetrievalWrapper
from models.howto100m_video_retrieval_wrapper import HowTo100MVideoRetrievalWrapper
from models.activitynet_video_retrieval_wrapper import ActivityNetCaptionsVideoRetrievalWrapper


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def init_config(path="config/default.yaml"):
    with open(path, "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)
        cfg = AttrDict(cfg)
        
    if "SEED" in cfg:
        seed_everything(cfg.SEED)
        
    torch.set_float32_matmul_precision("high")
    # torch.multiprocessing.set_start_method('spawn')
    warnings.filterwarnings("ignore", category=PossibleUserWarning)
    
    logger = set_up_logger()
        
    return cfg, logger
    
    
def init_dataloaders(cfg):
    # train dataloader
    train_dataset, train_collate_fn = get_train_dataset(cfg)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.batch_size,
        shuffle=True,
        num_workers=cfg.TRAIN.num_workers, 
        pin_memory=cfg.TRAIN.pin_memory,
        persistent_workers=cfg.TRAIN.persistent_workers,
        collate_fn=train_collate_fn,
    )
    
    # validation dataloader 
    val_dataset, val_collate_fn = get_eval_dataset(cfg)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1, # always 1 for evaluation
        shuffle=False,
        num_workers=cfg.EVAL.num_workers, 
        pin_memory=cfg.EVAL.pin_memory,
        persistent_workers=cfg.TRAIN.persistent_workers,
        collate_fn=val_collate_fn,
    )

    return train_dataloader, val_dataloader
    # return None, val_dataloader


def init_model(cfg):
    # Load video encoder model
    if cfg.MODEL.name == "s3d":
        s3d_args = cfg.MODEL.S3D
        s3d_model, s3d_projector = S3D(cfg), S3DProjector(cfg)

        if s3d_args.use_kinetics_pretrained:
            path = os.path.join(
                cfg.PATH.CKPT_PATH, "S3D", "S3D_kinetics400.pt"
            )
            assert os.path.isfile(path)

            weight_dict = torch.load(path)
            model_dict = s3d_model.state_dict()

            for name, param in weight_dict.items():
                if "module" in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    assert param.size() == model_dict[name].size()
                    model_dict[name].copy_(param)

        if s3d_args.freeze:
            video_encoder = s3d_projector
        else:
            video_encoder = nn.Sequential(
                s3d_model,
                s3d_projector,
            )
    elif cfg.MODEL.name == "frozen":
        frozen_args = cfg.MODEL.FROZEN
        frozen_model, frozen_projector = FrozenInTime(cfg), FrozenInTimeProjector(cfg)

        if frozen_args.use_cc_web_pretrained:
            path = os.path.join(
                cfg.PATH.CKPT_PATH, "FROZEN", "cc-webvid2m-4f_stformer_b_16_224.pth.tar"
            )
            sys.path.append("utils")
            checkpoint = torch.load(path)
            state_dict = checkpoint["state_dict"]
            new_state_dict = state_dict_data_parallel_fix(
                state_dict, frozen_model.state_dict()
            )
            # since we use 4 frames, _inflate_positional_embeds() not needed
            frozen_model.load_state_dict(new_state_dict, strict=False)

        if frozen_args.freeze:
            video_encoder = frozen_projector
        else:
            video_encoder = nn.Sequential(
                frozen_model,
                frozen_projector,
            )
    elif cfg.MODEL.name == "ours":
        video_encoder = TemporalSlotAttentionVideoEncoder(cfg) 
    else:
        raise NotImplementedError

    # Load video retrieval wrapper model (different across datasets)
    if cfg.DATASET.name == "moma":
        if "LOAD_FROM" in cfg.MODEL and len(cfg.MODEL.LOAD_FROM) > 0:
            model = MOMAVideoRetrievalWrapper.load_from_checkpoint(
                cfg=cfg,
                video_encoder=video_encoder,
                checkpoint_path=os.path.join(cfg.PATH.CKPT_PATH, cfg.MODEL.LOAD_FROM, "model-v1.ckpt"),
                strict=True,
            )
        else:
            model = MOMAVideoRetrievalWrapper(
                cfg=cfg,
                video_encoder=video_encoder,
            )
    elif cfg.DATASET.name == "howto100m":
        if "LOAD_FROM" in cfg.MODEL and len(cfg.MODEL.LOAD_FROM) > 0:
            model = HowTo100MVideoRetrievalWrapper.load_from_checkpoint(
                cfg=cfg,
                video_encoder=video_encoder,
                checkpoint_path=os.path.join(cfg.PATH.CKPT_PATH, cfg.MODEL.LOAD_FROM, "model-v1.ckpt"),
                strict=True,
            )
        else:
            model = HowTo100MVideoRetrievalWrapper(
                cfg=cfg,
                video_encoder=video_encoder,
            )
    elif cfg.DATASET.name == "activitynet":
        if "LOAD_FROM" in cfg.MODEL and len(cfg.MODEL.LOAD_FROM) > 0:
            model = ActivityNetCaptionsVideoRetrievalWrapper.load_from_checkpoint(
                cfg=cfg,
                video_encoder=video_encoder,
                checkpoint_path=os.path.join(cfg.PATH.CKPT_PATH, cfg.MODEL.LOAD_FROM, "model-v1.ckpt"),
                strict=True,
            )
        else:
            model = ActivityNetCaptionsVideoRetrievalWrapper(
                cfg=cfg,
                video_encoder=video_encoder,
            )
    else:
        raise NotImplementedError
    
    return model


def init_trainer(cfg):
    os.makedirs(cfg.PATH.LOG_PATH, exist_ok=True)
    if cfg.TRAINER.logger == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(cfg.PATH.LOG_PATH)
    elif cfg.TRAINER.logger == "wandb":
        logger = pl.loggers.WandbLogger(project="cvpr24_video_retrieval")
    
    # when use ddp (HowTo100M, ActivityNet-Captions)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0, 1, 2],
        strategy="ddp",
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        max_epochs=cfg.TRAIN.num_epochs,
    )

    # do not use ddp (MOMA-LRG)
    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     devices=[4],
    #     logger=logger,
    #     log_every_n_steps=1,
    #     check_val_every_n_epoch=10,
    #     num_sanity_val_steps=0,
    #     max_epochs=cfg.TRAIN.num_epochs,
    # )
    
    return trainer