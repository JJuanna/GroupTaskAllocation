import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.functional as F
class Group_Embedding(nn.Module):
    def __init__(self, embedding_dims):
        super(Group_Embedding, self).__init__()
        self.embed_dim = embedding_dims
        self.att_linear1 = nn.Linear(self.embed_dim, 16)
        self.att_linear2 = nn.Linear(16, 1)

    def forward(self, members_embed):
        x = F.relu(self.att_linear1(members_embed))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att_linear2(x))
        mask = self.padding_mask(members_embed, 0)
        mask = mask[:, :, 0:1][0]
        x = torch.mul(mask, x)
        # print(x)
        att = F.softmax(x, dim=0)
        att_members = torch.mm(members_embed.t(), att)
        user_p = att_members.t()
        return user_p


    def padding_mask(self, members_embed, pad_idx):
        return (members_embed != pad_idx).unsqueeze(0)