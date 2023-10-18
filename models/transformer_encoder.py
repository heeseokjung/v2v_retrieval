import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.args = cfg.MODEL.VIDEO.TRANSFORMER

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.args.d_model,
            nhead=self.args.nhead,
            dim_feedforward=self.args.dim_feedforward,
            activation=nn.GELU(),
            dropout=self.args.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.args.num_layers,
        )

    def forward(self, x):
        x = x.unsqueeze(dim=0) # n x d -> 1 x n x d (for batch computation)
        x = self.encoder(x) # 1 x n x d
        x = x.squeeze(dim=0)
        return x
