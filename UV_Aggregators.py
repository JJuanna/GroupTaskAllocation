import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention
from Group_Embedding import Group_Embedding


class UV_Aggregator(nn.Module):
    """
    task aggregator: for aggregating embeddings of groups.
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)
        self.Group_Embedding = Group_Embedding(self.embed_dim)

    def forward(self, nodes, history_uv, history_r):
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]
            e_uv = torch.empty(len(history), self.embed_dim).to(self.device)
            for j in range(0, len(history)):
                x = self.Group_Embedding(self.u2e.weight[history[j]])
                e_uv[j] = x
            uv_rep = self.v2e.weight[nodes[i]]
            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))
            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats
