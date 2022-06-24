import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from Group_Embedding import  Group_Embedding


class Users_sets_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_u_lists, history_r_lists, social_adj_lists, aggregator, cuda="cpu", uv=True):
        super(Users_sets_Encoder, self).__init__()
        # features is u2e
        self.features = features
        self.uv = uv
        self.history_uv_lists = history_u_lists
        self.history_r_lists = history_r_lists
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #
        self.att_linear1 = nn.Linear(self.embed_dim, 16)
        self.att_linear2 = nn.Linear(16, 1)
        self.w_r1 = nn.Linear(6*self.embed_dim,self.embed_dim)
        self.Group_Embedding = Group_Embedding(self.embed_dim)

    def forward(self, nodes):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        tmp_history_u = []
        tmp_history_r = []
        for node in nodes:
            if self.history_uv_lists[tuple(node.tolist())] == []:
                tmp_history_u.append([0])
                tmp_history_r.append([0])
            else:
                tmp_history_u.append(self.history_uv_lists[tuple(node.tolist())])
                tmp_history_r.append(self.history_r_lists[tuple(node.tolist())])
        neigh_feats = self.aggregator.forward(nodes, tmp_history_u, tmp_history_r, self.social_adj_lists)  # user-item network
        for k in range(0, len(nodes)):
            members_embed = self.features.weight[nodes[k]]
            user_p = self.Group_Embedding(members_embed)
            embed_matrix[k] = user_p

        self_feats = embed_matrix
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))




        return combined
