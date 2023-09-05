from collections import namedtuple
import torch
from torch.nn import LSTMCell
from torch_scatter import scatter_mean
from typing import Dict
from ..common.pytorch_models import MLP, MLPCustom, LayerNormLSTMCell
from ..common.lstm_conv import AdaptedNeuroSAT, LSTMConvV1


class GNNTSP(AdaptedNeuroSAT):
    def __init__(self, metadata, in_channels:Dict[str, int], out_channels:Dict[str, int], hidden_size:Dict[str, int]=128, num_passes:int=20, device="cpu"):
        """
        Args:
            metadata (_type_): metadata for the heterogeneous graph
            model_type (str): choice of 'sat_spec' or 'generic' influences the number of channels of the LSTM cells
            in_channels (Dict[str, int]): Dict mapping node type to the number of input features
            out_channels (Dict[str, int]): Dict mapping node type to the number of output features
            hidden_size (Dict[str, int], optional): Dict mapping node types to number of hidden channels. Defaults to 128.
            num_passes (int): number of LSTM passes
            device (str): device where computations happen
        """
        super(AdaptedNeuroSAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_passes = num_passes
        self.device = device
        self.projection_layers = torch.nn.ModuleDict()
        self.vote = MLP(hidden_size["arc"], 2, 1, hidden_size["arc"], device=device)
        lstm_hidden_sizes = {node_type: hidden_size[node_type] for node_type in metadata[0]}
        self.lstm_conv_layers = DTSPLSTMConv(lstm_hidden_sizes, lstm_hidden_sizes, metadata=metadata, device=device)
        self.lstm_state_tuple = namedtuple('LSTMState',('h','c'))
        for node_type in metadata[0]:
            if node_type == "city":
                self.projection_layers[node_type] = torch.nn.Linear(in_channels[node_type], hidden_size[node_type])
            else:
                self.projection_layers[node_type] = MLPCustom([8, 16, 32], in_channels[node_type], hidden_size["arc"], device=device)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict = {node_type: self.projection_layers[node_type](x) for node_type, x in x_dict.items()}
        for i in range(self.num_passes):
            if i == 0:
                previous_state_tuple = {node_type: self.lstm_state_tuple(value, torch.zeros_like(value)) for node_type, value in x_dict.items()}
            else:
                previous_state_tuple = {node_type: self.lstm_state_tuple(out[node_type].h, out[node_type].c) for node_type in x_dict.keys()}
            out = self.lstm_conv_layers(x_dict, edge_index_dict, previous_state_tuple, batch_dict)
            x_dict = {node_type: out[node_type][1] for node_type in x_dict.keys()}
        raw_votes = self.vote(x_dict["arc"])
        votes = scatter_mean(raw_votes, batch_dict["arc"], dim=0)

        return votes

class DTSPLSTMConv(LSTMConvV1):
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
        self.lstm_sizes = {}
        self.lstm_state_tuple = namedtuple('LSTMState',('h','c'))

        for node_type in metadata[0]:            
            hidden_size = in_channels[node_type]
            for edge_type in self.entering_edges_per_node_type[node_type]:
                src_node_type = edge_type[0]
                mlp_input_size = in_channels[src_node_type]
                self.mlp_blocks[str(edge_type)] = MLP(mlp_input_size, 2, hidden_size, hidden_size, device=self.device)
            lstm_input_size = sum([in_channels[src_node_type] for src_node_type in self.input_type_per_node_type[node_type]])            
            self.lstm_sizes[node_type] = lstm_input_size
            self.lstm_cells[node_type] = LayerNormLSTMCell(lstm_input_size, hidden_size, activation=torch.relu, state_tuple=self.lstm_state_tuple, device=self.device)
        
        self.reset_parameters()

    def forward(self, x_dict, edge_index_dict, previous_state_tuple:namedtuple, batch_dict: Dict):
        # Loop over each node type and each edge type
        sizes = {node_type: x_dict[node_type].size() for node_type in x_dict.keys()}
        output = {}
        for node_type in x_dict.keys():
            inputs = []
            hidden_state = previous_state_tuple[node_type].h
            for edge_type in self.entering_edges_per_node_type[node_type]:
                # Perform the message passing for the current node type and edge type
                source_node_type, _, _ = edge_type
                x = x_dict[source_node_type]
                x = self.mlp_blocks[str(edge_type)](x) # before the layer, x is the result of the projection 
                edge_index = edge_index_dict[edge_type]
                size = (sizes[source_node_type][0], sizes[node_type][0])
                agg = self.propagate(edge_index, size=size, x=x, edge_type=edge_type) # Here, we perform the "add" aggregation after the base features are passed through an MLP
                # Append the resulting node features to the list of hidden states
                inputs.append(agg)
            # Concatenate the resulting node features for each node type into a single vector
            h_cat = torch.cat(inputs, dim=1)            
            state_tuple = self.lstm_cells[node_type](inputs=h_cat, state=previous_state_tuple[node_type])
            output[node_type] = state_tuple

        return output