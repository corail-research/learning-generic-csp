from collections import namedtuple

import torch
from torch.nn import Linear
from torch.nn import GRUCell
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GATConv, HANConv
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn import HeteroConv, HGTConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset

from torch.nn import Dropout
from typing import Dict, List
from mlp import MLP

def repeat_end(val, n, k):
    return [val for i in range(n)] + [k]


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
    def __init__(self, hidden_channels, out_channels, num_heads, num_conv_layers, data, dropout_prob=0, num_mlp_layers=3):
        super().__init__()

        self.mlp_input_dict = torch.nn.ModuleDict()
        self.mlp_output_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            num_in_features = data[node_type]['x'].size(1)
            self.mlp_input_dict[node_type] = MLP(in_channels=num_in_features, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=num_mlp_layers, dropout=dropout_prob, act="relu")
            self.mlp_output_dict[node_type] = MLP(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=2, num_layers=num_mlp_layers, dropout=dropout_prob, act="relu")

        self.convs = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads, group='mean')
            self.convs.append(conv)
        
        self.out_layer = torch.nn.Softmax(dim=1)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.mlp_input_dict[node_type](x)
        
        # if self.dropout_prob:
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        x = self.out_layer(self.mlp_output_dict["variable"](x_dict["variable"]))

        return global_mean_pool(x, batch_dict["variable"])



class LSTMConv(MessagePassing):
    def __init__(self, in_channels:Dict, out_channels:Dict, input_per_type:Dict[str, List[str]], device=None, metadata=None, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.flow = 'target_to_source'
        self.device = device if device is not None else torch.device('cpu')
        self.gru_cells = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            hidden_size = sum([in_channels[n_type] for n_type in input_per_type[node_type]])
            self.gru_cells[node_type] = GRUCell(in_channels[node_type], hidden_size, device=self.device)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     super().reset_parameters()
    #     glorot(self.gru_cells)
    
    def forward(self, x_dict, edge_index_dict):
        # Loop over each node type and each edge type
        h_list = []
        for node_type in x_dict.keys():
            for edge_type in edge_index_dict.keys():
                # Perform the message passing for the current node type and edge type
                x = x_dict[node_type]
                edge_index = edge_index_dict[edge_type]
                self.propagate(edge_index, x=x, edge_type=edge_type)
                # Append the resulting node features to the list of hidden states
                h_list.append(x_dict[node_type])
            # Concatenate the resulting node features for each node type into a single vector
            h_cat = torch.cat(h_list, dim=1)

            # Pass the concatenated vector as the hidden state of the LSTM cell
            h_gru, c_gru = self.gru_cells[node_type](h_cat)

        return x_dict
    
    def message(self, x_j, edge_index_j):
        return x_j

class LSTMUpdate(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data, dropout_prob=0):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.gru_layers = torch.nn.ModuleDict()
        self.mlp_in = torch.nn.ModuleDict()
        self.mlp_vote = MLP(hidden_channels, num_layers, 2, hidden_channels)
        for node_type in data.node_types:
            num_in_features = data[node_type]['x'].size(1)
            self.lin_dict[node_type] = Linear(num_in_features, hidden_channels)
            self.gru_layers[node_type] = GRU(hidden_channels, hidden_channels, batch_first=True)
            self.mlp_in[node_type] = MLP(hidden_channels, num_layers, hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels * 2, hidden_channels * 2)
        self.out_layer = torch.nn.Softmax(dim=1)
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x)
        
        if self.dropout_prob:
            for gru in self.gru_layers:
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