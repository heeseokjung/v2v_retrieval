import torch
import torch.nn as nn
import torch.nn.functional as F
    

class STGCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.args = cfg.MODEL.GRAPH.STGCN
        
        self.linear1 = nn.Linear(self.args.in_dim, self.args.hidden_dim)
        self.linear2 = nn.Linear(self.args.hidden_dim, self.args.out_dim)
        self.gru = nn.GRUCell(self.args.out_dim, self.args.out_dim)
        
    def forward(self, x, adj_list, t_list):
        x = self.linear1(x)
        x = F.relu(x)
        hx = x
        
        xs = [self.linear2(torch.mm(adj_list[0], x))]
        for i in range(1, len(t_list)):
            delta = torch.abs(t_list[:i] - t_list[i])
            t_w = 1. / delta

            adj = t_w[:, None, None] * adj_list[:i, :, :]
            adj = adj.sum(dim=0) + adj_list[i, :, :]
            adj = torch.softmax(adj, dim=1)

            xs.append(self.linear2(torch.mm(adj, x)))
    
        for i in range(len(t_list)):
            hx = self.gru(xs[i], hx)

        return hx # V x d