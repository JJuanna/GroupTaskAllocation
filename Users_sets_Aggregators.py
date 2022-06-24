import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention
from Group_Embedding import  Group_Embedding



class Users_sets_Aggregator(nn.Module):
    """
    Group aggregator: for aggregating embeddings of a Group.
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(Users_sets_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_r3 = nn.Linear(self.embed_dim * 6, self.embed_dim)
        self.att = Attention(self.embed_dim)
        self.Group_Embedding = Group_Embedding(self.embed_dim)

    def forward(self, nodes, history_u, history_r, social_adj_lists):
        embed_matrix = torch.empty(len(history_u), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(history_u)):
            history = history_u[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]
            e_uv = self.v2e.weight[history]
            members_embed = self.u2e.weight[nodes[i]]
            members_embed = self.Group_Embedding(members_embed)
            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))
            att_w = self.att(o_history, members_embed, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats