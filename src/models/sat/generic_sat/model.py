import torch
from torch.nn import Linear
from torch.nn import LSTMCell
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import HeteroConv, HGTConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.utils import is_sparse
from torch_geometric.typing import Adj, Size
from torch.nn import Dropout
from torch_scatter import scatter_mean
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



class LSTMConvV1(MessagePassing):
    """This class is used to perform LSTM Convolution in GNNs. It assumes that the input x_dict 
    has already been passed through a type-specific MLP layer
    """
    def __init__(self, in_channels:Dict, out_channels:Dict, device=None, metadata=None, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.device = device if device is not None else torch.device('cpu')
        self.entering_edges_per_node_type = self.get_entering_edge_types_per_node_type(metadata[1], metadata[0])
        self.input_type_per_node_type = self.get_input_per_node_type(metadata[1], metadata[0])
        self.lstm_cells = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            input_size = sum([in_channels[n_type] for n_type in self.input_type_per_node_type[node_type]])
            hidden_size = in_channels[node_type]
            self.lstm_cells[node_type] = LSTMCell(input_size, hidden_size, device=self.device)
        
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.lstm_cells.values():
            for name, param in cell.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def get_entering_edge_types_per_node_type(self, edge_types, node_types):
        input_per_node_type = {node_type:[]  for node_type in node_types}
        for edge_type in edge_types:
            _, _, dst_type = edge_type
            input_per_node_type[dst_type].append(edge_type)
        return input_per_node_type
    
    def get_input_per_node_type(self, edge_types, node_types):
        input_per_type = {node_type: set() for node_type in node_types}
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            input_per_type[dst_type].add(src_type)
        
        return input_per_type

    def forward(self, x_dict, edge_index_dict, previous_hidden_states:Dict, previous_cell_states:Dict):
        # Loop over each node type and each edge type
        sizes = {node_type: x_dict[node_type].size() for node_type in x_dict.keys()}
        output = {}
        for node_type in x_dict.keys():
            h_list = []
            for edge_type in self.entering_edges_per_node_type[node_type]:
                # Perform the message passing for the current node type and edge type
                source_node_type, _, _ = edge_type
                x = x_dict[source_node_type]
                edge_index = edge_index_dict[edge_type]
                size = (sizes[source_node_type][0], sizes[node_type][0])
                self.propagate(edge_index, size=size, x=x, edge_type=edge_type)
                # Append the resulting node features to the list of hidden states
                h_list.append(x_dict[node_type])
            # Concatenate the resulting node features for each node type into a single vector
            h_cat = torch.cat(h_list, dim=1)
            # Pass the concatenated vector as the hidden state of the LSTM cell
            hidden_state = previous_hidden_states[node_type]
            cell_state = previous_cell_states[node_type]
            if hidden_state is None:
                # h, c = self.lstm_cells[node_type](h_cat)
                h, c = self.lstm_cells[node_type](h_cat.float())
            else:
                # h, c = self.lstm_cells[node_type](h_cat, (hidden_state, cell_state))
                h, c = self.lstm_cells[node_type](h_cat.float(), (hidden_state, cell_state))
            output[node_type] = (h, c)

        return output
    
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :class:`torch.Tensor`,
                a :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is a :obj:`torch.Tensor`, its :obj:`dtype`
                should be :obj:`torch.long` and its shape needs to be defined
                as :obj:`[2, num_messages]` where messages from nodes in
                :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is a :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :meth:`propagate`.
            size ((int, int), optional): The size :obj:`(N, M)` of the
                assignment matrix in case :obj:`edge_index` is a
                :class:`torch.Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if is_sparse(edge_index) and self.fuse and not self.explain:
            coll_dict = self._collect(self._fused_user_args, edge_index, size,
                                    kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:  # Otherwise, run both functions in separation.
            if decomposed_layers > 1:
                user_args = self._user_args
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]

                coll_dict = self._collect(self._user_args, edge_index, size,
                                        kwargs)

                msg_kwargs = self.inspector.distribute('message', coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res

                if self.explain:
                    explain_msg_kwargs = self.inspector.distribute(
                        'explain_message', coll_dict)
                    out = self.explain_message(out, **explain_msg_kwargs)

                aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res

                out = self.aggregate(out, **aggr_kwargs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.distribute('update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out
    
    def message(self, x_j, edge_index_j):
        return x_j

class LSTMConvV2(LSTMConvV1):
    """This class is used to perform LSTM Convolution in GNNs. It assumes that the input x_dict 
    has not yet been passed through an MLP layer. Instead, it concatenates the input features 
    coming from the neighboring nodes before passing them through the MLP.
    """
    def __init__(self, in_channels:Dict, out_channels:Dict, device=None, metadata=None, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, device=device, metadata=metadata)
        self.device = device if device is not None else torch.device('cpu')
        self.entering_edges_per_node_type = self.get_entering_edge_types_per_node_type(metadata[1], metadata[0])
        self.input_type_per_node_type = self.get_input_per_node_type(metadata[1], metadata[0])
        self.lstm_cells = torch.nn.ModuleDict()
        self.mlp_blocks = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            mlp_input_size = sum([in_channels[n_type] for n_type in self.input_type_per_node_type[node_type]])
            hidden_size = in_channels[node_type]
            self.mlp_blocks[node_type] = MLP(mlp_input_size, 2, hidden_size, hidden_size, device=self.device)            
            self.lstm_cells[node_type] = LSTMCell(hidden_size, hidden_size, device=self.device)
        
        self.reset_parameters()

    def forward(self, x_dict, edge_index_dict, previous_hidden_states:Dict, previous_cell_states:Dict):
        # Loop over each node type and each edge type
        sizes = {node_type: x_dict[node_type].size() for node_type in x_dict.keys()}
        output = {}
        for node_type in x_dict.keys():
            h_list = []
            for edge_type in self.entering_edges_per_node_type[node_type]:
                # Perform the message passing for the current node type and edge type
                source_node_type, _, _ = edge_type
                x = x_dict[source_node_type]
                edge_index = edge_index_dict[edge_type]
                size = (sizes[source_node_type][0], sizes[node_type][0])
                self.propagate(edge_index, size=size, x=x, edge_type=edge_type)
                # Append the resulting node features to the list of hidden states
                h_list.append(x_dict[node_type])
            # Concatenate the resulting node features for each node type into a single vector
            h_cat = torch.cat(h_list, dim=1)
            # Pass the concatenated vector as the hidden state of the LSTM cell
            hidden_state = previous_hidden_states[node_type]
            cell_state = previous_cell_states[node_type]
            lstm_input = self.mlp_blocks[node_type](h_cat)
            if hidden_state is None:
                # h, c = self.lstm_cells[node_type](h_cat)
                h, c = self.lstm_cells[node_type](lstm_input.float())
            else:
                # h, c = self.lstm_cells[node_type](h_cat, (hidden_state, cell_state))
                h, c = self.lstm_cells[node_type](lstm_input.float(), (hidden_state, cell_state))
            output[node_type] = (h, c)

        return output

class AdaptedNeuroSAT(torch.nn.Module):
    def __init__(self, metadata, in_channels:Dict[str, int], out_channels:Dict[str, int], hidden_size:Dict[str, int]=128, num_passes:int=20, device="cpu"):
        """
        Args:
            metadata (_type_): metadata for the heterogeneous graph
            in_channels (Dict[str, int]): Dict mapping node type to the number of input features
            out_channels (Dict[str, int]): Dict mapping node type to the number of output features
            hidden_size (Dict[str, int], optional): Dict mapping node types to number of hidden channels. Defaults to 128.
        """
        super(AdaptedNeuroSAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_passes = num_passes
        self.device = device
        self.projection_layers = torch.nn.ModuleDict()
        self.mlp_layers = torch.nn.ModuleDict()
        self.vote = MLP(hidden_size["variable"], 2, 1, hidden_size["variable"], device=device)
        lstm_hidden_sizes = {node_type: hidden_size[node_type] for node_type in metadata[0]}
        self.lstm_conv_layers = LSTMConvV1(lstm_hidden_sizes, lstm_hidden_sizes, metadata=metadata, device=device)
        for node_type in metadata[0]:
            self.projection_layers[node_type] = torch.nn.Linear(in_channels[node_type], hidden_size[node_type])
            self.mlp_layers[node_type] = MLP(hidden_size[node_type], 3, hidden_size[node_type], hidden_size[node_type], device=device)
        
    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict = {node_type: self.projection_layers[node_type](x) for node_type, x in x_dict.items()}
        for i in range(self.num_passes):
            x_dict = {node_type: self.mlp_layers[node_type](x) for node_type, x in x_dict.items()}
            if i == 0:
                previous_hidden_state = {node_type: None for node_type, x in x_dict.items()}
                previous_cell_state = {node_type: None for node_type, x in x_dict.items()}
            else:
                previous_hidden_state = {node_type: out[node_type][0] for node_type in x_dict.keys()}
                previous_cell_state = {node_type: out[node_type][1] for node_type in x_dict.keys()}
            out = self.lstm_conv_layers(x_dict, edge_index_dict, previous_hidden_state, previous_cell_state)
            x_dict = {node_type: out[node_type][1] for node_type in x_dict.keys()}
        raw_votes = self.vote(x_dict["variable"])
        votes = scatter_mean(raw_votes, batch_dict["variable"], dim=0)

        return votes

class AdaptedNeuroSATV2(AdaptedNeuroSAT):
    def __init__(self, metadata, in_channels:Dict[str, int], out_channels:Dict[str, int], hidden_size:Dict[str, int]=128, num_passes:int=20, device="cpu"):
        """
        Args:
            metadata (_type_): metadata for the heterogeneous graph
            in_channels (Dict[str, int]): Dict mapping node type to the number of input features
            out_channels (Dict[str, int]): Dict mapping node type to the number of output features
            hidden_size (Dict[str, int], optional): Dict mapping node types to number of hidden channels. Defaults to 128.
        """
        super(AdaptedNeuroSAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_passes = num_passes
        self.device = device
        self.projection_layers = torch.nn.ModuleDict()
        self.vote = MLP(hidden_size["variable"], 2, 1, hidden_size["variable"], device=device)
        lstm_hidden_sizes = {node_type: hidden_size[node_type] for node_type in metadata[0]}
        self.lstm_conv_layers = LSTMConvV2(lstm_hidden_sizes, lstm_hidden_sizes, metadata=metadata, device=device)
        for node_type in metadata[0]:
            self.projection_layers[node_type] = torch.nn.Linear(in_channels[node_type], hidden_size[node_type])
        
    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict = {node_type: self.projection_layers[node_type](x) for node_type, x in x_dict.items()}
        for i in range(self.num_passes):
            if i == 0:
                previous_hidden_state = {node_type: None for node_type, x in x_dict.items()}
                previous_cell_state = {node_type: None for node_type, x in x_dict.items()}
            else:
                previous_hidden_state = {node_type: out[node_type][0] for node_type in x_dict.keys()}
                previous_cell_state = {node_type: out[node_type][1] for node_type in x_dict.keys()}
            out = self.lstm_conv_layers(x_dict, edge_index_dict, previous_hidden_state, previous_cell_state)
            x_dict = {node_type: out[node_type][1] for node_type in x_dict.keys()}
        raw_votes = self.vote(x_dict["variable"])
        votes = scatter_mean(raw_votes, batch_dict["variable"], dim=0)

        return votes


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