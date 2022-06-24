import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, history_u_lists, history_v_lists, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        self.history_u_lists = history_u_lists
        self.history_v_lists = history_v_lists
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):
        to_neigh = []
        for i in range(0, len(nodes)):
            nodes_neighbor = []
            for task in self.history_u_lists[tuple(nodes[i].tolist())]:
                nodes_neighbor.extend(self.history_v_lists[task])
            to_neigh.append(nodes_neighbor)

        neigh_feats = self.aggregator.forward(nodes, to_neigh)  # user-user network
        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined