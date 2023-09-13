from collections import namedtuple
import torch
from torch.nn import LSTMCell
from torch_scatter import scatter_mean
from typing import Dict
from ..common.pytorch_models import MLP, LayerNormLSTMCell
from ..common.lstm_conv import AdaptedNeuroSAT, LSTMConvV1


# class NeuroSAT(AdaptedNeuroSAT):
#     def __init__(self, metadata, model_type:str, in_channels:Dict[str, int], out_channels:Dict[str, int], hidden_size:Dict[str, int]=128, num_passes:int=20, device="cpu", flip_inputs:bool=False):
#         """
#         Args:
#             metadata (_type_): metadata for the heterogeneous graph
#             model_type (str): choice of 'sat_spec' or 'generic' influences the number of channels of the LSTM cells
#             in_channels (Dict[str, int]): Dict mapping node type to the number of input features
#             out_channels (Dict[str, int]): Dict mapping node type to the number of output features
#             hidden_size (Dict[str, int], optional): Dict mapping node types to number of hidden channels. Defaults to 128.
#             num_passes (int): number of LSTM passes
#             device (str): device where computations happen
#             flip_inputs (bool): whether or not to flip the inputs for variable hidden states
#         """
#         super(AdaptedNeuroSAT, self).__init__()
#         self.model_type = model_type
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.hidden_size = hidden_size
#         self.num_passes = num_passes
#         self.device = device
#         self.projection_layers = torch.nn.ModuleDict()
#         self.vote = MLP(hidden_size["variable"], 2, 1, hidden_size["variable"], device=device)
#         lstm_hidden_sizes = {node_type: hidden_size[node_type] for node_type in metadata[0]}
#         self.lstm_conv_layers = NeuroSatLSTMConv(lstm_hidden_sizes, lstm_hidden_sizes, model_type=self.model_type, metadata=metadata, device=device, flip_inputs=flip_inputs)
#         self.lstm_state_tuple = namedtuple('LSTMState',('h','c'))
#         for node_type in metadata[0]:
#             self.projection_layers[node_type] = torch.nn.Linear(in_channels[node_type], hidden_size[node_type])
        
#     def forward(self, x_dict, edge_index_dict, batch_dict):
#         x_dict = {node_type: self.projection_layers[node_type](x) for node_type, x in x_dict.items()}
#         for i in range(self.num_passes):
#             if i == 0:
#                 previous_state_tuple = {node_type: self.lstm_state_tuple(value, torch.zeros_like(value)) for node_type, value in x_dict.items()}
#             else:
#                 previous_state_tuple = {node_type: self.lstm_state_tuple(out[node_type].h, out[node_type].c) for node_type in x_dict.keys()}
#             out = self.lstm_conv_layers(x_dict, edge_index_dict, previous_state_tuple, batch_dict)
#             x_dict = {node_type: out[node_type][1] for node_type in x_dict.keys()}
#         raw_votes = self.vote(x_dict["variable"])
#         votes = scatter_mean(raw_votes, batch_dict["variable"], dim=0)

#         return votes

# class NeuroSatLSTMConv(LSTMConvV1):
#     """This class is used to perform LSTM Convolution in GNNs. It assumes that the input x_dict 
#     has not yet been passed through an MLP layer. Instead, it concatenates the input features 
#     coming from the neighboring nodes before passing them through the MLP.
#     """
#     def __init__(self, in_channels:Dict, out_channels:Dict, device=None, metadata=None, model_type=None, **kwargs):
#         super().__init__(in_channels=in_channels, out_channels=out_channels, device=device, metadata=metadata)
#         self.device = device if device is not None else torch.device('cpu')
#         self.entering_edges_per_node_type = self.get_entering_edge_types_per_node_type(metadata[1], metadata[0])
#         self.input_type_per_node_type = self.get_input_per_node_type(metadata[1], metadata[0])
#         self.lstm_cells = torch.nn.ModuleDict()
#         self.mlp_blocks = torch.nn.ModuleDict()
#         self.lstm_sizes = {}
#         self.lstm_state_tuple = namedtuple('LSTMState',('h','c'))
#         self.flip_inputs = kwargs.get("flip_inputs", False)
#         self.model_type = model_type

#         for node_type in metadata[0]:            
#             hidden_size = in_channels[node_type]
#             for edge_type in self.entering_edges_per_node_type[node_type]:
#                 src_node_type = edge_type[0]
#                 mlp_input_size = in_channels[src_node_type]
#                 self.mlp_blocks[str(edge_type)] = MLP(mlp_input_size, 2, hidden_size, hidden_size, device=self.device)
#             lstm_input_size = sum([in_channels[src_node_type] for src_node_type in self.input_type_per_node_type[node_type]])            
#             self.lstm_sizes[node_type] = lstm_input_size
#             self.lstm_cells[node_type] = LayerNormLSTMCell(lstm_input_size, hidden_size, state_tuple=self.lstm_state_tuple, device=self.device)
        
#         self.reset_parameters()

#     def forward(self, x_dict, edge_index_dict, previous_state_tuple:namedtuple, batch_dict: Dict):
#         # Loop over each node type and each edge type
#         sizes = {node_type: x_dict[node_type].size() for node_type in x_dict.keys()}
#         output = {}
#         for node_type in x_dict.keys():
#             inputs = []
#             hidden_state = previous_state_tuple[node_type].h
#             for edge_type in self.entering_edges_per_node_type[node_type]:
#                 # Perform the message passing for the current node type and edge type
#                 source_node_type, _, _ = edge_type
#                 x = x_dict[source_node_type]
#                 if edge_type == ("variable", "is_negation_of", "variable"):
#                     if self.flip_inputs:
#                         flipped = self.flip(hidden_state, batch_dict)
#                         inputs.append(flipped)
#                     else:
#                         inputs.append(hidden_state)
#                 else:
#                     x = self.mlp_blocks[str(edge_type)](x) # before the layer, x is the result of the projection 
#                     edge_index = edge_index_dict[edge_type]
#                     size = (sizes[source_node_type][0], sizes[node_type][0])
#                     agg = self.propagate(edge_index, size=size, x=x, edge_type=edge_type) # Here, we perform the "add" aggregation after the base features are passed through an MLP
#                     # Append the resulting node features to the list of hidden states
#                     inputs.append(agg)
#             # Concatenate the resulting node features for each node type into a single vector
#             h_cat = torch.cat(inputs, dim=1)            
#             """
#             This part is edge-type-dependent: 
#             for lit -> clause, we perform the update in the following way:
#                 - The LSTM input is directly the aggregated output of the mlp (agg = self.propagate ...)
#                 - The hidden and cell states are the following:
#                     - The initial LSTM cell state is a zeros tensor of the same size as the hidden state
#                     - The initial LSTM hidden state is the result of the initial projection
#                 - For the following steps, they are the following:
#                     - The LSTM cell state is the cell state from the previous step
#                     - The LSTM hidden state is the hidden state from the previous step
#             for clause -> lit, we perform the update in the following way:
#                 - The LSTM input is the concatenation of the aggregated output of the mlp and the flipped hidden state from the previous step. 
#                   For the first step, the hidden state is the result of the initial projection.
#                 - The hidden and cell states are the following:
#                     - The initial LSTM cell state is a zeros tensor of the same size as the hidden state
#                     - The initial LSTM hidden state is the result of the initial projection
#                 - For the following steps, they are the following:
#                     - The LSTM cell state is the cell state from the previous step
#                     - The LSTM hidden state is the hidden state from the previous step
#             """  
#             state_tuple = self.lstm_cells[node_type](inputs=h_cat, state=previous_state_tuple[node_type])
#             output[node_type] = state_tuple

#         return output
    
#     def flip(self, hidden_state, batch_dict):
#         """Performs the flip operation, as describred in the NeuroSAT paper. This operation replaces -"flips"- the hidden state 
#         of the variable nodes with the hidden state of the negated variable
#         """
#         num_vars_per_instance = torch.bincount(batch_dict["variable"])
#         offset = 0
#         new_index = []
#         for i, num_variables in enumerate(num_vars_per_instance):
#             flipped_indices = torch.arange(offset + num_variables - 1, offset - 1, -1, device=self.device)
#             new_index.append(flipped_indices)
#             offset += num_variables
#         new_index = torch.cat(new_index)
#         flipped = torch.index_select(hidden_state, 0, index=new_index.long())

#         return flipped


class NeuroSAT(AdaptedNeuroSAT):
    def __init__(self,
                metadata,
                in_channels:Dict[str, int],
                out_channels:Dict[str, int],
                hidden_size:Dict[str, int]=128,
                num_passes:int=20,
                device="cpu",
                flip_inputs:bool=False,
                layernorm_lstm_cell:bool=True,
                **kwargs
            ):
        """
        Args:
            metadata (_type_): metadata for the heterogeneous graph
            in_channels (Dict[str, int]): Dict mapping node type to the number of input features
            out_channels (Dict[str, int]): Dict mapping node type to the number of output features
            hidden_size (Dict[str, int], optional): Dict mapping node types to number of hidden channels. Defaults to 128.
            num_passes (int): number of LSTM passes
            device (str): device where computations happen
            flip_inputs (bool): whether or not to flip the inputs for variable hidden states
            layernorm_lstm_cell (bool): whether or not to use the layernorm lstm cell
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
        self.lstm_conv_layers = NeuroSatLSTMConv(lstm_hidden_sizes, lstm_hidden_sizes, metadata=metadata, device=device, flip_inputs=flip_inputs, layernorm_lstm_cell=layernorm_lstm_cell, **kwargs)
        for node_type in metadata[0]:
            self.projection_layers[node_type] = torch.nn.Linear(in_channels[node_type], hidden_size[node_type])
        
    def forward(self, x_dict, edge_index_dict, batch_dict):
        with torch.no_grad():
            x_dict = {node_type: self.projection_layers[node_type](x) for node_type, x in x_dict.items()}
        for i in range(self.num_passes):
            if i == 0:
                previous_hidden_state = {node_type: value for node_type, value in x_dict.items()}
                previous_cell_state = {node_type: None for node_type, x in x_dict.items()}
            else:
                previous_hidden_state = {node_type: out[node_type][0] for node_type in x_dict.keys()}
                previous_cell_state = {node_type: out[node_type][1] for node_type in x_dict.keys()}
            out = self.lstm_conv_layers(x_dict, edge_index_dict, previous_hidden_state, previous_cell_state, batch_dict)
            x_dict = {node_type: out[node_type][1] for node_type in x_dict.keys()}
        raw_votes = self.vote(x_dict["variable"])
        votes = scatter_mean(raw_votes, batch_dict["variable"], dim=0)

        return votes

class NeuroSatLSTMConv(LSTMConvV1):
    """This class is used to perform LSTM Convolution in GNNs. It assumes that the input x_dict 
    has not yet been passed through an MLP layer. Instead, it concatenates the input features 
    coming from the neighboring nodes before passing them through the MLP.
    """
    def __init__(self, in_channels:Dict, out_channels:Dict, device=None, metadata=None, layernorm_lstm_cell=True, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, device=device, metadata=metadata, **kwargs)
        self.device = device if device is not None else torch.device('cpu')
        self.entering_edges_per_node_type = self.get_entering_edge_types_per_node_type(metadata[1], metadata[0])
        self.input_type_per_node_type = self.get_input_per_node_type(metadata[1], metadata[0])
        self.lstm_cells = torch.nn.ModuleDict()
        self.mlp_blocks = torch.nn.ModuleDict()
        self.lstm_sizes = {}
        self.flip_inputs = kwargs.get("flip_inputs", False)

        for node_type in metadata[0]:            
            hidden_size = in_channels[node_type]
            for edge_type in self.entering_edges_per_node_type[node_type]:
                src_node_type = edge_type[0]
                mlp_input_size = in_channels[src_node_type]
                self.mlp_blocks[str(edge_type)] = MLP(mlp_input_size, 2, hidden_size, hidden_size, device=self.device)
            lstm_input_size = sum([in_channels[src_node_type] for src_node_type in self.input_type_per_node_type[node_type]])            
            self.lstm_sizes[node_type] = lstm_input_size
            
            if layernorm_lstm_cell:
                self.lstm_cells[node_type] = LayerNormLSTMCell(lstm_input_size, hidden_size, torch.relu, device=self.device)
            else:
                self.lstm_cells[node_type] = LSTMCell(lstm_input_size, hidden_size, device=self.device)
            
        
        self.reset_parameters()

    def forward(self, x_dict, edge_index_dict, previous_hidden_states:Dict, previous_cell_states:Dict, batch_dict: Dict):
        # Loop over each node type and each edge type
        sizes = {node_type: x_dict[node_type].size() for node_type in x_dict.keys()}
        output = {}
        for node_type in x_dict.keys():
            inputs = []
            hidden_state = previous_hidden_states[node_type]
            for edge_type in self.entering_edges_per_node_type[node_type]:
                # Perform the message passing for the current node type and edge type
                source_node_type, _, _ = edge_type
                x = x_dict[source_node_type]
                if edge_type == ("variable", "is_negation_of", "variable"):
                    if self.flip_inputs:
                        flipped = self.flip(hidden_state, batch_dict)
                        inputs.append(flipped)
                    else:
                        inputs.append(hidden_state)
                else:
                    x = self.mlp_blocks[str(edge_type)](x) # before the layer, x is the result of the projection 
                    edge_index = edge_index_dict[edge_type]
                    size = (sizes[source_node_type][0], sizes[node_type][0])
                    agg = self.propagate(edge_index, size=size, x=x, edge_type=edge_type) # Here, we perform the "add" aggregation after the base features are passed through an MLP
                    # Append the resulting node features to the list of hidden states
                    inputs.append(agg)
            # Concatenate the resulting node features for each node type into a single vector
            h_cat = torch.cat(inputs, dim=1)            
            if previous_cell_states[node_type] is None:
                cell_state = torch.zeros_like(hidden_state)
            else:
                cell_state = previous_cell_states[node_type]
            """
            This part is edge-type-dependent: 
            for lit -> clause, we perform the update in the following way:
                - The LSTM input is directly the aggregated output of the mlp (agg = self.propagate ...)
                - The hidden and cell states are the following:
                    - The initial LSTM cell state is a zeros tensor of the same size as the hidden state
                    - The initial LSTM hidden state is the result of the initial projection
                - For the following steps, they are the following:
                    - The LSTM cell state is the cell state from the previous step
                    - The LSTM hidden state is the hidden state from the previous step
            for clause -> lit, we perform the update in the following way:
                - The LSTM input is the concatenation of the aggregated output of the mlp and the flipped hidden state from the previous step. 
                  For the first step, the hidden state is the result of the initial projection.
                - The hidden and cell states are the following:
                    - The initial LSTM cell state is a zeros tensor of the same size as the hidden state
                    - The initial LSTM hidden state is the result of the initial projection
                - For the following steps, they are the following:
                    - The LSTM cell state is the cell state from the previous step
                    - The LSTM hidden state is the hidden state from the previous step
            """ 
            h, c = self.lstm_cells[node_type](h_cat, (hidden_state, cell_state))
            output[node_type] = (h, c)

        return output
    
    def flip(self, hidden_state, batch_dict):
        """Performs the flip operation, as describred in the NeuroSAT paper. This operation replaces -"flips"- the hidden state 
        of the variable nodes with the hidden state of the negated variable
        """
        num_vars_per_instance = torch.bincount(batch_dict["variable"])
        offset = 0
        new_index = []
        for i, num_variables in enumerate(num_vars_per_instance):
            flipped_indices = torch.arange(offset + num_variables - 1, offset - 1, -1, device=self.device)
            new_index.append(flipped_indices)
            offset += num_variables
        new_index = torch.cat(new_index)
        flipped = torch.index_select(hidden_state, 0, index=new_index.long())

        return flipped