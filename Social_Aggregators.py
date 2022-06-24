import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.functional as F
from Attention import Attention
from Group_Embedding import  Group_Embedding



class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.Group_embedding = Group_Embedding(self.embed_dim)
        self.w_r3 = nn.Linear(self.embed_dim * 6, self.embed_dim)
        self.att_linear1 = nn.Linear(self.embed_dim, 16)
        self.att_linear2 = nn.Linear(16, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(list(tmp_adj))
            # group的维度n
            members_embed = self.u2e.weight[nodes[i]]
            user_p = self.Group_embedding(members_embed)
            embed_matrix[i] = user_p
        to_feats = embed_matrix
        return to_feats
