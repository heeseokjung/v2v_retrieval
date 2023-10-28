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

        self.pos_emb = nn.Embedding(self.args.max_len, self.args.d_model)
        self.register_buff("pos_ids", torch.arange(self.args.max_len, dtype=torch.long))

    def forward(self, x, pad_mask):
        pos_emb = self.pos_emb(self.pos_ids[:x.shape[1]])
        x = x + pos_emb[None, :]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        
        return x