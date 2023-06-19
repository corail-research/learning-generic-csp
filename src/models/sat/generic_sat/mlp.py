import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_end(val, n, k):
    return [val for i in range(n)] + [k]

# class MLP(nn.Module):
#     def __init__(self, d_in, d_outs, device):
#         super().__init__()
#         self.transfer_fn = F.relu
#         self.output_size = d_outs[-1]
#         dims = [d_in] + d_outs
#         self.layers = nn.ModuleList([
#             nn.Linear(dims[i], dims[i+1],device=device) for i in range(len(dims)-1)
#         ])

#         for layer in self.layers:
#             nn.init.xavier_normal_(layer.weight) #Normal or uniform?
#             nn.init.zeros_(layer.bias)

#     def forward(self, z):
#         x = z
#         for layer in self.layers[:-1]:
#             x = self.transfer_fn(layer(x))

#         return self.layers[-1](x)

class MLP(nn.Module):
    def __init__(self, input_dimension, num_layers, output_size, layer_size, device="cuda:0"):
        super().__init__()
        self.input_dimension = input_dimension
        self.num_layers = num_layers
        self.output_size = output_size
        self.layer_size = layer_size
        self.device = device
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dimension, self.layer_size, device=self.device)
        ])
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.layer_size, self.layer_size, device=self.device))
        self.layers.append(nn.Linear(self.layer_size, self.output_size, device=self.device))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return self.layers[-1](x)

class LayerNormLSTMCell(nn.Module):
    def __init__(self, hidden_channels, activation=torch.tanh, state_tuple=None, device='cuda:0'):
        super().__init__()
        self.activation = activation
        self.hidden_size = hidden_channels
        self.lstm_cell = None
        self.ln_ih = None
        self.ln_hh = None
        self.ln_ho = None
        self.state_tuple = state_tuple
        self.device = device
        self.dropout = nn.Dropout(0.1)

    def infer_size(self,x):
        self.input_size = x.size(-1)
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size,device=self.device)
        self.ln_ih = nn.LayerNorm(self.hidden_size * 4,device=self.device)
        self.ln_hh = nn.LayerNorm(self.hidden_size * 4,device=self.device)
        self.ln_ho = nn.LayerNorm(self.hidden_size,device=self.device)

    def set_input_size(self,size):
        self.input_size = size
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size,device=self.device)
        self.ln_ih = nn.LayerNorm(self.hidden_size * 4,device=self.device)
        self.ln_hh = nn.LayerNorm(self.hidden_size * 4,device=self.device)
        self.ln_ho = nn.LayerNorm(self.hidden_size,device=self.device)

    def forward(self, inputs, state):

        if self.lstm_cell is None:
            self.infer_size(inputs)
        
        hx, cx = state
        i2h = inputs @ self.lstm_cell.weight_ih.t()
        h2h = hx @ self.lstm_cell.weight_hh.t() + self.lstm_cell.bias_hh
        gates = self.ln_ih(i2h) + self.ln_hh(h2h)
        
        i, f, o, g = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = self.activation(g)

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(self.activation(cy))
        hy = self.dropout(hy)

        return self.state_tuple(h=hy, c=cy)