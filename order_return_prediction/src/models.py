import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HANConv, GATConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self, out_channels, metadata):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 64)
        self.conv2 = SAGEConv((-1, -1), 128)
        self.conv3 = SAGEConv((-1, -1), 256)
        self.linear = nn.Linear(256, out_channels)
        self.model = to_hetero(self, metadata, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict).relu()
        x = self.conv2(x, edge_index_dict)
        x = self.conv3(x, edge_index_dict).relu()
        x = self.linear(x)
        return {"order": F.softmax(x, dim=-1)}

class HAN(nn.Module):
    def __init__(self, dim_in, dim_out, metadata, dim_h=128, heads=8):
        super().__init__()
        self.han = HANConv(dim_in, dim_h, heads=heads, dropout=0.6, metadata=metadata)
        self.linear = nn.Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        return {"order": self.linear(out["order"])}

class GAT(torch.nn.Module):
    def __init__(self, dim_h, dim_out, metadata):
        super().__init__()
        self.conv = GATConv((-1, -1), dim_h, add_self_loops=False)
        self.linear = nn.Linear(dim_h, dim_out)
        self.model = to_hetero(self, metadata, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        h = self.conv(x_dict, edge_index_dict).relu()
        return {"order": self.linear(h)}
