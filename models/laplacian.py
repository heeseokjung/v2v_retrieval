import torch.nn as nn


class GraphLaplacianEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
        self.cfg = cfg
        self.args = cfg.MODEL.GRAPH.LAPLACIAN

        self.spatial_pos_emb = nn.Linear(
            self.args.pos_enc_dim, self.args.in_dim
        )
        self.temporal_pos_emb = nn.Embedding(
            self.args.max_len, self.args.in_dim
        )
        self.cls_linear = nn.Linear(
            self.args.in_dim, self.args.out_dim
        )

        self.transformer_encoder = TransformerEncoder()

    def forward(self, x, pos_enc, cls, ts_l):
        k = self.args.pos_enc_dim
        if pos_enc.shape[-1] > k:
            pos_enc = pos_enc[:,1:k+1]
        pos_enc = self.spatial_pos_emb(pos_enc) # n x k -> n x d
        time_enc = self.temporal_pos_emb(ts_l) # n x d
        x = x + pos_enc + time_enc # n x d
        ####### add transformer encoder #######
        x = self.transformer_encoder(x)
        #######################################
        cls = self.cls_linear(cls)
        
        return x, cls
    

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1024,
            activation=nn.GELU(),
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=4
        )

    def forward(self, x):
        x = x.unsqueeze(dim=0) # n x d -> 1 x n x d (for batch computation)
        x = self.encoder(x) # 1 x n x d
        x = x.squeeze()
        return x