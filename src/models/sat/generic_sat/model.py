from collections import namedtuple

from mlp import MLP, LayerNormLSTMCell

import torch
from torch.nn import Linear
from torch.nn import GRU
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn import HeteroConv, HGTConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Dropout

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

class NeuroSATLSTM(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, data, num_updates=26, dropout_prob=0, device="cuda:0") -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm_state_tuple = namedtuple('LSTMState', ('h','c'))
        self.final_reducer = F.relu
        self.decode_transfer_fn = F.relu

        self.variable_init = torch.empty((1, hidden_channels),device=self.device)
        torch.nn.init.xavier_uniform_(self.variable_init)
        
        # self.value_init = torch.empty((1, hidden_channels),device=self.device)
        # torch.nn.init.normal_(self.variable_init)
        
        self.operator_init = torch.empty((1, hidden_channels),device=self.device)
        torch.nn.init.xavier_uniform_(self.operator_init)
        
        self.constraint_init = torch.empty((1, hidden_channels),device=self.device)
        torch.nn.init.xavier_uniform_(self.constraint_init)

        self.value_msg = MLP(hidden_channels, repeat_end(hidden_channels, num_layers, hidden_channels), device=self.device) # 1 for variable
        self.variable_msg = MLP(hidden_channels * 3, repeat_end(hidden_channels, num_layers, hidden_channels), device=self.device) # 3 for constraint, operator, value
        self.operator_msg = MLP(hidden_channels * 2, repeat_end(hidden_channels, num_layers, hidden_channels), device=self.device) # 2 for constraint, variable
        self.constraint_msg = MLP(hidden_channels * 2, repeat_end(hidden_channels, num_layers, hidden_channels), device=self.device) # 2 for operator, variable

        self.variable_update = LayerNormLSTMCell(hidden_channels, activation=self.decode_transfer_fn, state_tuple=self.lstm_state_tuple, device=self.device)
        self.constraint_update = LayerNormLSTMCell(hidden_channels, activation=self.decode_transfer_fn, state_tuple=self.lstm_state_tuple, device=self.device)
        self.operator_update = LayerNormLSTMCell(hidden_channels, activation=self.decode_transfer_fn, state_tuple=self.lstm_state_tuple, device=self.device)
        self.value_update = LayerNormLSTMCell(hidden_channels, activation=self.decode_transfer_fn, state_tuple=self.lstm_state_tuple, device=self.device)

        self.variable_vote = MLP(hidden_channels, repeat_end(hidden_channels, num_layers, 1),device=self.device)
        self.vote_bias = torch.nn.Parameter(torch.zeros(1, device=self.device))    

        self.param_list = list(self.value_msg.parameters()) \
            + list(self.variable_msg.parameters()) \
            + list(self.operator_msg.parameters()) \
            + list(self.constraint_msg.parameters()) \
            + list(self.value_update.parameters()) \
            + list(self.variable_update.parameters()) \
            + list(self.operator_update.parameters()) \
            + list(self.constraint_update.parameters()) \
            + list(self.variable_vote.parameters()) + [self.vote_bias]
            
    def forward(self, node_dict, edge_dict, batch_dict):
        
        value_output = torch.tile(self.value_init, [2, 1])
        variable_output = torch.tile(self.variable_init, [self.num_variables, 1])
        constraint_output = torch.tile(self.operator_init, [self.num_constraints, 1])
        operator_output = torch.tile(self.constraint_init, [self.num_operators, 1])

        value_state = self.lstm_state_tuple(h=value_output, c=torch.zeros([2, self.hidden_channels],device=self.device))
        variable_state = self.lstm_state_tuple(h=variable_output, c=torch.zeros([self.num_variables, self.hidden_channels],device=self.device))
        operator_state = self.lstm_state_tuple(h=operator_output, c=torch.zeros([self.num_operators, self.hidden_channels],device=self.device))
        constraint_state = self.lstm_state_tuple(h=constraint_output, c=torch.zeros([self.num_constraints, self.hidden_channels],device=self.device))

        for _ in range(self.num_updates):
            value_pre_msgs = self.value_msg.forward(variable_state.h)
            value_state = self.value_update(inputs=value_pre_msgs, state=value_state)
            
            variable_input = torch.cat([value_state.h, operator_state.h, constraint_state.h], axis=1)
            variable_pre_msgs = self.variable_msg.forward(variable_input)
            variable_state = self.variable_update(inputs=variable_pre_msgs, state=variable_state)
            
            operator_input = torch.cat([variable_state.h, constraint_state.h], axis=1)
            operator_pre_msgs = self.operator_msg.forward(operator_input)
            operator_state = self.operator_update(inputs=operator_pre_msgs, state=operator_state)

            constraint_input = torch.cat([variable_state.h, operator_state.h], axis=1)
            constraint_pre_msgs = self.constraint_msg.forward(constraint_input)
            constraint_state = self.constraint_update(inputs=constraint_pre_msgs, state=constraint_state)
        
        return variable_state.h

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