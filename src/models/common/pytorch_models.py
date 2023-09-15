import torch
import torch.nn as nn
import torch.nn.functional as F
import collections


class MLP(nn.Module):
    def __init__(self, input_dimension: int, num_layers: int, output_size: int, layer_size: int, device: str = "cuda:0", weight_init:callable=nn.init.xavier_uniform_, bias_init:callable=nn.init.zeros_) -> None:
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
            weight_init (callable): The weight initialization function to use for the model parameters (default is `nn.init.xavier_uniform_`).
            bias_init (callable): The bias initialization function to use for the model parameters (default is `nn.init.zeros_`).
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.num_layers = num_layers
        self.output_size = output_size
        self.layer_size = layer_size
        self.device = device
        self.layers = nn.ModuleList([])
    
        input_layer = nn.Linear(self.input_dimension, self.layer_size, device=self.device)
        weight_init(input_layer.weight)
        bias_init(input_layer.bias)
        self.layers.append(input_layer)
        for i in range(self.num_layers - 1):
            layer = nn.Linear(self.layer_size, self.layer_size, device=self.device)
            weight_init(layer.weight)
            bias_init(layer.bias)
            self.layers.append(layer)

        output_layer = nn.Linear(self.layer_size, self.output_size, device=self.device)
        weight_init(output_layer.weight)
        bias_init(output_layer.bias)
        self.layers.append(output_layer)
    
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
    def __init__(self, hidden_sizes: list[int], input_size: int, output_size: int, device: str = "cuda:0", weight_init:callable=nn.init.xavier_uniform_, bias_init:callable=nn.init.zeros_) -> None:
        """
        A custom implementation of a multi-layer perceptron (MLP) that allows you to specify the number of hidden units
        in each hidden layer as a list. This class is similar to the `MLP` class, but provides more flexibility in
        specifying the architecture of the network.

        Args:
            hidden_sizes (list of int): A list of integers specifying the number of hidden units in each hidden layer.
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            device (str): The device to use for the model parameters (default is "cuda:0").
            weight_init (callable): The weight initialization function to use for the model parameters (default is `nn.init.xavier_uniform_`).
            bias_init (callable): The bias initialization function to use for the model parameters (default is `nn.init.zeros_`).
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.layers = nn.ModuleList([])
    
        input_layer = nn.Linear(self.input_size, self.hidden_sizes[0], device=self.device)
        weight_init(input_layer.weight)
        bias_init(input_layer.bias)
        self.layers.append(input_layer)
        
        for i in range(len(self.hidden_sizes) - 1):
            layer = nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1], device=self.device)
            weight_init(layer.weight)
            bias_init(layer.bias)
            self.layers.append(layer)
        
        output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size, device=self.device)
        weight_init(output_layer.weight)
        bias_init(output_layer.bias)
        self.layers.append(output_layer)
    
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



class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, activation=torch.tanh, device="cpu", bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.activation = activation
        self.ln_ih = nn.LayerNorm(4 * hidden_size).to(device)
        self.ln_hh = nn.LayerNorm(4 * hidden_size).to(device)
        self.ln_ho = nn.LayerNorm(hidden_size).to(device)

    def forward(self, input, hidden=None):
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.activation(self.ln_ho(cy))
        return hy, cy