import torch
import torch.nn as nn
import torch.nn.functional as F

############ visualization ###########
import matplotlib.pyplot as plt
import seaborn as sns
######################################


"""
This is the PyTorch ported version of official implementation in:
https://github.com/google-research/google-research/tree/master/slot_attention
"""


class SlotAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.args = cfg.MODEL.SLOT

        self.norm_inputs = nn.LayerNorm(self.args.input_size)
        self.norm_slots = nn.LayerNorm(self.args.slot_size)
        self.norm_mlp = nn.LayerNorm(self.args.slot_size)

        # Parameters for Gaussian init (shared by all slots).
        if self.args.init_type == "gaussian":
            self.slots_mu = nn.Parameter(
                torch.randn(1, self.args.slot_size),
                requires_grad=True,
            )
            self.slots_log_sigma = nn.Parameter(
                torch.randn(1, self.args.slot_size),
                requires_grad=True,
            )
        elif self.args.init_type == "fixed":
            self.slots = nn.Parameter(
                torch.randn(self.args.num_slots, self.args.slot_size),
                requires_grad=True,
            )
        else:
            raise NotImplementedError

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.args.slot_size, self.args.slot_size, bias=False)
        self.project_k = nn.Linear(self.args.input_size, self.args.slot_size, bias=False)
        self.project_v = nn.Linear(self.args.input_size, self.args.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.args.slot_size, self.args.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.args.slot_size, self.args.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.args.mlp_hidden_size, self.args.slot_size),
        )

        # self.count = 0

    def forward(self, inputs):
        # batch size not provided (n can differs between videos)
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs) # n x d
        v = self.project_v(inputs) # n x d

        # Initialize slots
        if self.args.init_type == "gaussian":
            # slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
            #     self.args.num_slots, self.args.slot_size
            # ).to("cuda")
            slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
                self.args.num_slots, self.args.slot_size
            ).to("cuda")
        elif self.args.init_type == "fixed":
            slots = self.slots
        else:
            raise NotImplementedError
        
        # Multiple rounds of attention
        vars = []
        for i in range(self.args.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention
            q = self.project_q(slots) # n_slot x d
            q = q * (self.args.slot_size ** -0.5) # Normalization
            attn_logits = torch.mm(k, q.t()) # n x n_slot
            attn = F.softmax(attn_logits, dim=-1) # n x n_slot

            # Weighted mean
            attn = attn + self.args.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True) # n x n_slot
            vars.append(attn.var()) # for var hinge loss 
            updates = torch.mm(attn.t(), v) # n_slots x d

            ############# visualization ############
            # tmp_inputs = inputs[:]
            # tmp_inputs = F.normalize(tmp_inputs, dim=-1)
            # tsm = torch.mm(tmp_inputs, tmp_inputs.t())
            # tsm = tsm.detach().cpu().numpy()
            # sns.heatmap(tsm)
            # plt.savefig(f"/root/slot_vis/tsm/{self.count}.png")
            # plt.close()

            # tmp_attn = attn[:].t().detach().cpu().numpy()
            # sns.heatmap(tmp_attn)
            # plt.savefig(f"/root/slot_vis/slot/{self.count}_iter{i}.png")
            # plt.close()
            ########################################

            # Slot update
            slots = self.gru(updates, slots_prev)
            slots = slots + self.mlp(self.norm_mlp(slots))

        # self.count += 1

        return slots, torch.stack(vars).mean() 