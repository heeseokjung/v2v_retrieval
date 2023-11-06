'''
    Most of the codes are adopted from:
    https://github.com/evelinehong/slot-attention-pytorch
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SlotAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_slots = args.num_slots
        self.iters = args.iters
        self.eps = float(args.eps)
        self.scale = args.dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, args.dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, args.dim))

        self.to_q = nn.Linear(args.dim, args.dim)
        self.to_k = nn.Linear(args.dim, args.dim)
        self.to_v = nn.Linear(args.dim, args.dim)

        self.gru = nn.GRUCell(args.dim, args.dim)

        hidden_dim = max(args.dim, args.hidden_dim)

        self.fc1 = nn.Linear(args.dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, args.dim)

        self.norm_input  = nn.LayerNorm(args.dim)
        self.norm_slots  = nn.LayerNorm(args.dim)
        self.norm_pre_ff = nn.LayerNorm(args.dim)

        self.pos_emb = nn.Embedding(args.num_slots, args.dim)
        self.register_buffer("pos_ids", torch.arange(args.num_slots, dtype=torch.long))

        # self.count = 0

    def forward(self, inputs, pad_mask, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        ########################## visualization ##########################
        # inputs_v = inputs[:].squeeze(dim=0)
        # inputs_v = F.normalize(inputs_v, dim=-1)
        # tsm = torch.mm(inputs_v, inputs_v.t()).detach().cpu().numpy()
        # sns.heatmap(tsm)
        # plt.savefig(f"/root/slot_vis/tsm/{self.count}.png")
        # plt.close()
        ###################################################################

        for i in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale # b x k x n
            attn = dots.softmax(dim=1) + self.eps # b x k x n
            attn = attn / attn.sum(dim=-1, keepdim=True) # b x k x n
            if pad_mask is not None:
                attn = attn.masked_fill(pad_mask[:,None,:], 0) # mask out padding

            ########################## visualization ##########################
            # attn_v = attn[:].detach().cpu().numpy()
            # attn_v = attn_v.squeeze(axis=0)
            # sns.heatmap(attn_v)
            # plt.savefig(f"/root/slot_vis/slot/{self.count}_iter{i}.png")
            # plt.close()
            ###################################################################

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        # self.count += 1

        return slots + self.pos_emb(self.pos_ids)


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.linear = nn.Linear(args.in_dim, args.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            activation=nn.GELU(),
            dropout=args.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=args.num_layers,
        )

        self.in_pos_emb = nn.Embedding(args.max_len, args.d_model)
        self.out_pos_emb = nn.Embedding(args.max_len, args.d_model)
        self.register_buffer("pos_ids", torch.arange(args.max_len, dtype=torch.long))

    def forward(self, x, pad_mask):
        x = self.linear(x)
        # in_pos_emb = self.in_pos_emb(self.pos_ids[:x.shape[1]])
        # x = x + in_pos_emb[None, :]
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        # out_pos_emb = self.out_pos_emb(self.pos_ids[:x.shape[1]])
        # return x + out_pos_emb[None, :]
        return x
    

class SlotDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.linear = nn.Linear(args.slot_dim, args.out_dim)
        # TODO:
        # ConvTranspose1d -> reconstruction loss

    def forward(self, x):
        return self.linear(x)
    

class TemporalSlotAttentionVideoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.args = cfg.MODEL.SLOT

        self.encoder = TransformerEncoder(self.args.encoder)
        self.slot_attention = SlotAttention(self.args.slot_attn)
        self.decoder = SlotDecoder(self.args.decoder)

    def forward(self, x, pad_mask):
        x = self.encoder(x, pad_mask)
        slots = self.slot_attention(x, pad_mask)
        out = self.decoder(slots)
        return out