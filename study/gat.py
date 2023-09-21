import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2*out_features, 1)

    def forward(self, X, adjacency_matrix):
        h = self.W(X)
        N = h.size(0)
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*h.size(1))
        e = F.leaky_relu(self.a(a_input).squeeze(2), negative_slope=0.2)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adjacency_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads):
        super().__init__()
        self.attentions = [GraphAttentionLayer(in_features, hidden_features) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden_features*num_heads, out_features)

    def forward(self, X, adjacency_matrix):
        x = torch.cat([att(X, adjacency_matrix) for att in self.attentions], dim=1)
        x = self.out_att(x, adjacency_matrix)
        return F.log_softmax(x, dim=1)
