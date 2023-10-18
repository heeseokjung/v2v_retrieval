import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter_add


'''
    Most of the codes are adopted from:
    https://github.com/wondergo2017/DIDA
    Dynamic Graph Neural Networks Under Spatio-Temporal Distribution Shift (NeurIPS '22)
'''


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''

    def __init__(self, n_hid, max_len=600):  # original max_len=240
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class DGNNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_heads,
        norm=True,
        dropout=0,
        skip=False,
        agg="add",
        use_RTE=True,
        use_fmask=False,
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.norm = norm
        self.skip = skip
        self.use_RTE = use_RTE
        self.use_fmask = use_fmask
        self.in_dim, self.hidden_dim = in_dim, hidden_dim
        self.agg = agg
        
        self.msg_linear = nn.Linear(in_dim*2, in_dim)
        
        self.q_linear = nn.Linear(in_dim, hidden_dim) # W_q
        self.k_linear = nn.Linear(in_dim, hidden_dim) # W_k
        self.v_linear = nn.Linear(in_dim, hidden_dim) # W_v
        
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.update_skip = nn.Parameter(torch.ones(1))
        self.update_drop = nn.Dropout(dropout)
        
        self.time_emb = RelTemporalEncoding(hidden_dim)
        self.cs_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.GELU(),
            nn.Linear(2*hidden_dim, hidden_dim),
        )
        
    def ffn(self, x):
        if self.norm:
            emb = self.cs_mlp(self.update_norm(x)) # MLP + LayerNorm
        else:
            emb = self.cs_mlp(x) # MLP
            
        emb = self.update_drop(emb)
        
        if self.skip:
            alpha = torch.sigmoid(self.update_skip)
            emb = (1 - alpha)*x + alpha*emb # skip connection
        else:
            emb = x + emb
            
        return emb
        
    def collect_neighbors(self, edge_index_list, t_list, x_list, edge_attr):
        time_span = len(x_list)
        for tgt_idx in range(time_span):
            topos = []
            x_tgt = x_list[tgt_idx]
            for src_idx in range(max(0, tgt_idx-2), tgt_idx+1):
                ei_src = edge_index_list[src_idx]
                ei_tgt = ei_src[1,:]
                # x_tgt_e = x_list[tgt_idx].index_select(dim=0, index=ei_src[1,:])
                # x_src_e = x_list[src_idx].index_select(dim=0, index=ei_src[0,:])
                # attr_e = edge_attr.index_select(dim=0, index=ei_src[2,:])
                x_tgt_e = x_list[tgt_idx][ei_src[1,:], :]
                x_src_e = x_list[src_idx][ei_src[0,:], :]
                attr_e = edge_attr[ei_src[2,:], :]
                topo = [x_tgt_e, x_src_e, attr_e, t_list[tgt_idx], t_list[src_idx], ei_tgt]
                topos.append(topo)
            yield x_tgt, topos
        
    def DAttn(self, x_tgt, x_src, attr_e, t_tgt, t_src):
        # to sparse
        target_node_vec = x_tgt
        source_msg_vec = torch.cat([x_src, attr_e], dim=-1)
        source_msg_vec = self.msg_linear(source_msg_vec)
        
        if self.use_RTE: # CAW style
            target_node_vec = self.time_emb(
                target_node_vec, torch.LongTensor([t_tgt-t_tgt]).to("cuda")
            )
            source_msg_vec = self.time_emb(
                source_msg_vec, torch.LongTensor([t_tgt-t_src]).to("cuda")
            )
            
        q = self.q_linear(target_node_vec).view(-1, self.n_heads, self.d_k) # e x h x d/h
        k = self.k_linear(source_msg_vec).view(-1, self.n_heads, self.d_k) # e x h x d/h 
        v = self.v_linear(source_msg_vec).view(-1, self.n_heads, self.d_k) # e x h x d/h
        
        res_attn = (q * k).sum(dim=-1) / self.sqrt_dk # e x h
        res_msg = v                                   # e x h x d/h
        
        return res_attn, res_msg
        
    def DAttnMulti(self, x_tgt, topos):
        res_atts = []
        res_msgs = []
        ei_tgts = []
        
        for x_tgt_e, x_src_e, attr_e, t_tgt, t_src, ei_tgt in topos:
            att, msg = self.DAttn(x_tgt_e, x_src_e, attr_e, t_tgt, t_src)
            res_atts.append(att)
            res_msgs.append(msg)
            ei_tgts.append(ei_tgt)
            
        res_att = torch.cat(res_atts, dim=0) # E x h
        res_msg = torch.cat(res_msgs, dim=0) # E x h x d/h
        # ei_tgt = torch.cat(ei_tgts)
        ei_tgt = np.concatenate(ei_tgts)
        ei_tgt = torch.from_numpy(ei_tgt).to("cuda")
        
        
        res_att = softmax(res_att, ei_tgt) # E x h
        res = res_msg * res_att.view(-1, self.n_heads, 1) # E x h x d/h
        res = res.view(-1, self.hidden_dim) # E x d
        
        emb = scatter(res, ei_tgt, dim=0, dim_size=x_tgt.shape[0], reduce=self.agg) # V x d
        
        if self.use_fmask:
            ... # raise NotImplementedError
        
        emb = self.ffn(emb + x_tgt)
        
        return emb
        
    def forward(self, edge_index_list, t_list, x_list, edge_attr):
        xs = []
        for x_tgt, topos in self.collect_neighbors(edge_index_list, t_list, x_list, edge_attr):
            x = self.DAttnMulti(x_tgt, topos)
            xs.append(x)

        return xs


class DGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.args = cfg.MODEL.DIDA
        
        n_layers = self.args.n_layers
        n_heads = self.args.n_heads
        norm = self.args.norm
        
        in_dim = self.args.in_dim
        hidden_dim = self.args.hidden_dim
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            DGNNLayer(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                norm=norm,
                dropout=self.args.dropout,
                skip=self.args.skip,
                agg=self.args.agg,
                use_RTE=self.args.use_RTE,
                use_fmask=self.args.use_fmask,
            ) for _ in range(n_layers)
        )
        self.act = F.relu
        
    def forward(self, edge_index_list, t_list, x_list, edge_attr):
        x_list = [self.linear(x) for x in x_list]
        
        for i, layer in enumerate(self.layers):
            x_list = layer(edge_index_list, t_list, x_list, edge_attr)
            if i != len(self.layers)-1:
                x_list = [self.act(x) for x in x_list]
                
        return x_list
    
    
    
def main():
    import yaml
    from utils.attr_dict import AttrDict
    
    cfg_path = "/root/cvpr24_video_retrieval/config/default.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)
        cfg = AttrDict(cfg)
    
    model = DGNN(cfg=cfg)
    
    data_path = "/data/dir_moma/graphs/HpmtecgPGBo.pt"
    data = torch.load(data_path)
    
    '''
        data["x"]: node feature
        data["msg"]: edge feature
        data["edge_index_list"]: edge index list
        data["t_list"]: timestamp list
    '''
    
    x = data["x"]
    edge_index_list = data["edge_index_list"]
    t_list = data["t_list"]
    edge_attr = data["edge_attr"]
    x_list = [x for _ in range(len(edge_index_list))]
    
    print(f"x: {x.shape} x_list: {len(x_list)}")
    
    x_list = model(edge_index_list, t_list, x_list, edge_attr)
    
    print(f"output: {len(x_list)}")
    
    
if __name__ == "__main__":
    main()