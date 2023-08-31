import torch
import torch.nn as nn
import torch.nn.functional as F
import collections


class MLP(nn.Module):
    def __init__(self, input_dimension: int, num_layers: int, output_size: int, layer_size: int, device: str = "cuda:0") -> None:
        """
        A multi-layer perceptron (MLP) is a feedforward neural network that consists of one or more layers of fully
        connected neurons. This class is a PyTorch implementation of an MLP with a configurable number of hidden layers
        and hidden units, where all hidden layers have the same number of hidden units.

        Args:
            input_dimension (int): The size of the input layer.
            num_layers (int): The number of hidden layers in the MLP.
            output_size (int): The size of the output layer.
            layer_size (int): The number of hidden units in each hidden layer.
            device (str): The device to use for the model parameters (default is "cuda:0").
        """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return self.layers[-1](x)


class MLPCustom(nn.Module):
    def __init__(self, hidden_sizes: list[int], input_size: int, output_size: int, device: str = "cuda:0") -> None:
        """
        A custom implementation of a multi-layer perceptron (MLP) that allows you to specify the number of hidden units
        in each hidden layer as a list. This class is similar to the `MLP` class, but provides more flexibility in
        specifying the architecture of the network.

        Args:
            hidden_sizes (list of int): A list of integers specifying the number of hidden units in each hidden layer.
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            device (str): The device to use for the model parameters (default is "cuda:0").
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        
        self.layers = nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_sizes[0], device=self.device)
        ])
        
        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1], device=self.device))
        
        self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size, device=self.device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return self.layers[-1](x)


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, activation: callable = torch.tanh, state_tuple:collections.namedtuple=None, device='cpu'):
        """
        A long short-term memory (LSTM) cell with layer normalization applied to the input-to-hidden and hidden-to-hidden
        weight matrices.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state.
            activation (callable): The activation function to use (default is torch.tanh).
            state_tuple (collections.namedtuple): A named tuple to use for the output state (default is None).
            device (str): The device to use for the model parameters (default is 'cpu').
        """
        super().__init__()
        self.activation = activation
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, device=self.device)
        self.ln_ih = nn.LayerNorm(self.hidden_size * 4, device=self.device)
        self.ln_hh = nn.LayerNorm(self.hidden_size * 4,device=self.device)
        self.ln_ho = nn.LayerNorm(self.hidden_size,device=self.device)
        self.state_tuple = state_tuple # namedtuple('LSTMState',('h','c'))
        self.dropout = nn.Dropout(0.1) #FIXME: recurrent dropout

    def forward(self, inputs, state):
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