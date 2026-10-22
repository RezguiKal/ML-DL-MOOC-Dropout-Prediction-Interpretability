import torch

from torch import nn
import torch.nn.functional as F

class MLayeredDNN(nn.Module):

    """
        Builds a M-layered dense neural network

        Params:
            - input_shape (tuple): the input shape (time_len, features)
            - m (int): the number of layers
            - alpha (int): the shrinkage parameter that determines the number
                of neurons from one layer to the other
    """
    def __init__(self, input_shape, m=5, alpha=2):
        super(MLayeredDNN, self).__init__()

        h,w = input_shape

        self.m = m
        self.alpha = alpha

        self.layers = list()

        self.layers = [nn.Linear(w//(2**i), w//(2**(i+1))) for i in range(m-1)]
        self.layers.append(nn.Linear(w//(2**(m-1)), 1))

        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        return torch.sigmoid(self.layers[-1](x))
