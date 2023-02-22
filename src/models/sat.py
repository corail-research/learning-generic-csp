import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MeanAggregation

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.pool = MeanAggregation()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.pool(x, batch)
        x = self.lin(x)
        
        return x