import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        # num_neighs：item的length
        # 将tensor在指定的维度上进行重复
        # print(node1.size())>>[20, 64]
        # print(num_neighs)>>20
        uv_reps = u_rep.repeat(num_neighs, 1)
        # print(uv_reps.size())>>[20,64]
        # print(u_rep.size())>>[64]
        # [n,128]
        x = torch.cat((node1, uv_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        # print(x)
        # dim = 0,每一列和为1
        att = F.softmax(x, dim=0)
        # print(att)
        return att
