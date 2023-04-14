import torch
from torch.nn import Linear
from torch.nn import GRU
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn import HeteroConv, HGTConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Dropout

class HGTSATSpecific(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data, dropout_prob=0):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            num_in_features = data[node_type]['x'].size(1)
            self.lin_dict[node_type] = Linear(num_in_features, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels * 2, out_channels)
        self.out_layer = torch.nn.Softmax(dim=1)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        
        if self.dropout_prob:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: self.dropout(value) for key, value in x_dict.items()}

        else:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
        
        constraint_pool = global_max_pool(x_dict["constraint"], batch_dict["constraint"])
        variable_pool = global_max_pool(x_dict["variable"], batch_dict["variable"])
        x = self.lin(torch.concat((variable_pool, constraint_pool), dim=1))
        x = self.out_layer(x)

        return x

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data, dropout_prob=0):
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

        self.lin = Linear(hidden_channels, out_channels)
        self.out_layer = torch.nn.Softmax(dim=1)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        
        if self.dropout_prob:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: self.dropout(value) for key, value in x_dict.items()}

        else:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
        
        diffs = batch_dict["constraint"].diff()
        diffs[0] = 1
        indices = (diffs == 1).nonzero(as_tuple=True)
        main_constraint = x_dict["constraint"][indices]
        x = self.lin(main_constraint)
        x = self.out_layer(x)

        return x
    
class GatedUpdate(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data, dropout_prob=0):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            num_in_features = data[node_type]['x'].size(1)
            self.lin_dict[node_type] = Linear(num_in_features, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.gru_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)
            self.gru_layers.append(GRU(hidden_channels, hidden_channels, batch_first=True))

        self.lin = Linear(hidden_channels * 2, hidden_channels * 2)
        self.lin_out = Linear(hidden_channels * 2, out_channels)
        self.out_layer = torch.nn.Softmax(dim=1)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        
        if self.dropout_prob:
            for conv, gru in zip(self.convs, self.gru_layers):
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: self.dropout(F.relu(value)) for key, value in x_dict.items()}
        else:
            for conv, gru in zip(self.convs, self.gru_layers):
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: gru(value.unsqueeze(1))[0].squeeze(1) for key, value in x_dict.items()}

        pooled_constraint = global_mean_pool(x_dict["constraint"], batch_dict["constraint"])
        pooled_operator = global_mean_pool(x_dict["operator"], batch_dict["operator"])
        concat = torch.cat((pooled_constraint, pooled_operator), dim=1)

        x = F.relu(self.lin(concat))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.lin(x))
        x = F.dropout(x, p=0.2)
        x = self.lin_out(x)

        return x

class HGTMeta(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data, dropout_prob=0):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            num_in_features = data[node_type]['x'].size(1)
            self.lin_dict[node_type] = Linear(num_in_features, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.out_layer = torch.nn.Softmax(dim=1)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

    def forward(self, x_dict, edge_index_dict, batch_dict, filename=None):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        
        if self.dropout_prob:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: self.dropout(value) for key, value in x_dict.items()}
        else:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)

        diffs = batch_dict["constraint"].diff()
        diffs[0] = 1
        indices = (diffs == 1).nonzero(as_tuple=True)
        main_constraint = x_dict["constraint"][indices]
        x = self.lin(main_constraint)
        x = self.out_layer(x)

        return x


class SatGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("variable", "connected_to", "variable"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("variable", "rev_connected_to", "variable"): GATConv((-1, -1), hidden_channels, add_self_loops=False), 
                ("variable", "connected_to", "constraint"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ("constraint", "rev_connected_to", "variable"): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr="sum")
            self.convs.append(conv)        
        self.lin = Linear(hidden_channels * 2, 2)
        self.out_layer = torch.nn.Softmax(dim=1)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # x_dict = {key: F.dropout(x.relu(), p=0.3, training=self.training) for key, x in x_dict.items()}
        
        variable_pool = global_mean_pool(x_dict["variable"], batch_dict["variable"])
        constraint_pool = global_mean_pool(x_dict["constraint"], batch_dict["constraint"])
        concatenated = torch.concat((variable_pool, constraint_pool), dim=1)
        x = self.lin(concatenated)
        x = self.out_layer(x)

        return x