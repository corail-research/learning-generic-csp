import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn import HeteroConv, HGTConv
from torch_geometric.nn import global_mean_pool


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            num_in_features = data[node_type]['x'].size(1)
            self.lin_dict[node_type] = Linear(num_in_features, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels*2, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        
        variable_pool = global_mean_pool(x_dict["variable"], batch_dict["variable"])
        constraint_pool = global_mean_pool(x_dict["constraint"], batch_dict["constraint"])

        concatenated = torch.concat((variable_pool, constraint_pool), dim=1)
        x = self.lin(concatenated)

        return x


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
            }, aggr="sum")
            self.convs.append(conv)        
        self.lin = Linear(hidden_channels * 2, 2)
        self.out_layer = torch.nn.Softmax()

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(x.relu(), p=0.3, training=self.training) for key, x in x_dict.items()}
        a = 1
        
        variable_pool = global_mean_pool(x_dict["variable"], batch_dict["variable"])
        constraint_pool = global_mean_pool(x_dict["constraint"], batch_dict["constraint"])
        concatenated = torch.concat((variable_pool, constraint_pool), dim=1)
        # concatenated = F.relu(concatenated)

        x = self.lin(concatenated)

        # return self.out_layer(x)
        return x