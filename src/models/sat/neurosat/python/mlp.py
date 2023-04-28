import numpy as np
import torch
import torch.nn as nn
from util import decode_transfer_fn, repeat_end


class MLP(nn.Module):
    def __init__(self, d_in, d_outs,device):
        super().__init__()
        self.transfer_fn = decode_transfer_fn("relu")
        self.output_size = d_outs[-1]

        dims = [d_in] + d_outs

        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1],device=device) for i in range(len(dims)-1)
        ])

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight) #Normal or uniform?
            nn.init.zeros_(layer.bias)

    def forward(self, z):

        x = z
        for layer in self.layers[:-1]:
            x = self.transfer_fn(layer(x))

        return self.layers[-1](x)
    

class LayerNormLSTMCell(nn.Module):
    def __init__(self, opts, activation=torch.tanh, state_tuple=None,device='cpu'):
        super().__init__()
        self.activation = activation
        self.hidden_size = opts.d
        self.lstm_cell = None
        self.ln_ih = None
        self.ln_hh = None
        self.ln_ho = None
        self.state_tuple = state_tuple
        self.device = device
        self.dropout = nn.Dropout(0.1) #FIXME: recurrent dropout


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