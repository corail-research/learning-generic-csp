import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import global_mean_pool

class SatGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("variable", "connected_to", "value"): GATConv((-1, -1), hidden_channels, add_self_loops=False), 
                ("variable", "connected_to", "operator"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("variable", "connected_to", "constraint"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("operator", "connected_to", "constraint"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("constraint", "connected_to", "constraint"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("value", "rev_connected_to", "variable"): GATConv((-1, -1), hidden_channels, add_self_loops=False), 
                ("operator", "rev_connected_to", "variable"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("constraint", "rev_connected_to", "variable"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("constraint", "rev_connected_to", "operator"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr="mean")
            self.convs.append(conv)        
        self.lin = Linear(hidden_channels * 4, 2)
        self.out_layer = torch.nn.Softmax()

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        a = 1
        
        value_pool = global_mean_pool(x_dict["value"], None)
        variable_pool = global_mean_pool(x_dict["variable"], None)
        operator_pool = global_mean_pool(x_dict["operator"], None)
        constraint_pool = global_mean_pool(x_dict["constraint"], None)
        concatenated = torch.concat((value_pool, variable_pool, operator_pool, constraint_pool), dim=1)
        concatenated = F.relu(concatenated)

        x = self.lin(concatenated)

        return self.out_layer(x)