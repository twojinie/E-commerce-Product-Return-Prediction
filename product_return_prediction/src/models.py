import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HANConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self, out_channels, metadata, conv_type="sage"):
        super().__init__()
        if conv_type=="sage":
            self.conv1 = SAGEConv((-1,-1),32); self.conv2 = SAGEConv((-1,-1),32)
        else:
            self.conv1 = GATConv((-1,-1),16,heads=2,concat=True,dropout=0.3)
            self.conv2 = GATConv(32,16,heads=1,concat=True,dropout=0.3)
        self.linear = nn.Linear(32 if conv_type=="sage" else 16, out_channels)
        self.model = to_hetero(self, metadata, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict).relu()
        x = self.conv2(x, edge_index_dict)
        x = self.linear(x)
        return {"product_order": torch.sigmoid(x)}

class HAN(nn.Module):
    def __init__(self, dim_in, metadata, out_channels=1, dim_h=128):
        super().__init__()
        self.han = HANConv(dim_in, dim_h, heads=8, dropout=0.6, metadata=metadata)
        self.fc1 = nn.Linear(dim_h,128); self.fc2 = nn.Linear(128,64); self.fc3 = nn.Linear(64,out_channels)
        self.bn1 = nn.BatchNorm1d(128); self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        h = F.relu(self.bn1(self.fc1(out["product_order"])))
        h = F.dropout(h,p=0.4,training=self.training)
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.dropout(h,p=0.4,training=self.training)
        return torch.sigmoid(self.fc3(h))
