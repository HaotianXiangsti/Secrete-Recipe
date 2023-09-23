import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ResGatedGraphConv

class GCNModel(nn.Module):
    def __init__(self, time_dim, device, input_shape, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = ResGatedGraphConv(input_dim, hidden_dim)
        self.conv2 = ResGatedGraphConv(hidden_dim, output_dim)
        self.device = device
        self.time_dim = time_dim
        self.hidden = nn.Linear(input_shape, time_dim)

    def pos_encoding(self, t, emb_dim, n=10000):
        pos_enc = torch.zeros(t.shape[0], emb_dim).to(self.device)
        inv_freq = 1.0 / (
                n
                ** (torch.arange(0, emb_dim, 2, device=self.device).float() / emb_dim)
        )
        pos_enc_a = torch.sin(t.repeat(1, emb_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, emb_dim // 2) * inv_freq)
        pos_enc[:,0::2] = pos_enc_a
        pos_enc[:,1::2] = pos_enc_b
        return pos_enc

    def forward(self, data, edge_index, t):

        x, edge_index = data, edge_index

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        t = t.unsqueeze(-2)
        t = t.repeat(1,x.shape[1],1)

        # 第一层GCN

        x = self.hidden(x)
        x = x + t
        x = self.conv1(x, edge_index)
        x = x.relu()

        # 第二层GCN
        x = self.conv2(x, edge_index)

        return x