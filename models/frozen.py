import torch
import torch.nn as nn

from models.video_transformer import SpaceTimeTransformer


class FrozenInTimeProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.args = cfg.MODEL.FROZEN.projector

        self.projector = nn.Sequential(
            nn.Linear(self.args.in_dim, self.args.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.args.dropout),
            nn.Linear(self.args.hidden_dim, self.args.out_dim),
        )

    def forward(self, x):
        return self.projector(x)


class FrozenInTime(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.args = cfg.MODEL.FROZEN

        self.video_model = SpaceTimeTransformer(
            num_frames=self.args.num_frames,
            time_init=self.args.time_init,
            attention_style=self.args.attention_style,
        )

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(dim=0) # b x num_frames x c x w x h

        embeddings = self.video_model(x) # b x num_patches x d
        
        if self.args.emb_type == "flat":
            return embeddings[:,0,:]
        elif self.args.emb_type == "set":
            return embeddings
        else:
            raise ValueError