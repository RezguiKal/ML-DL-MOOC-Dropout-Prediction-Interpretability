import torch
import torch.nn as nn

import numpy as np

import torch.nn.functional as F

class ConRec(nn.Module):
    
    def __init__(self, input_size, h_size=50,\
        kernel_conv_size=2, kernel_pool_size=2,
        num_channels = [20,50]):

        super(ConRec, self).__init__()

        c, self.h, self.w = input_size

        self.h_size = h_size
        self.num_channels = num_channels
        self.kernel_conv_size = kernel_conv_size
        self.kernel_pool_size = kernel_pool_size

        self.fc_input_size = np.abs((self.h - 2*self.kernel_conv_size \
            - 2*self.kernel_pool_size + 4) * \
                (self.w - 2*self.kernel_conv_size\
                    - 2*self.kernel_pool_size + 4))
        
        # first convolution layer
        self.conv_first = nn.Conv2d(in_channels=1,\
            out_channels=self.num_channels[0],\
                kernel_size=self.kernel_conv_size)
        # second convolution layer
        self.conv_second = nn.Conv2d(in_channels=20,\
            out_channels=self.num_channels[1],\
                kernel_size=self.kernel_conv_size)
        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=self.kernel_pool_size,\
            stride=1)
        # flattening layer
        self.flatten_one = nn.Flatten(start_dim=2)
        self.flatten_last = nn.Flatten()
        # fully connected layer
        self.fc_one = nn.Linear(in_features=self.fc_input_size,\
            out_features=20)
        self.fc_last = nn.Linear(in_features=self.h_size,\
            out_features=1)
        # recurrent layer
        self.rnn = nn.RNN(input_size=20, hidden_size=self.h_size,
                        nonlinearity='relu', batch_first=True)
            
    def forward(self, x):
        x = self.max_pool(F.relu(self.conv_first(x)))
        x = self.max_pool(F.relu(self.conv_second(x)))
        x = self.flatten_one(x)
        x = F.relu(self.fc_one(x))
        x, h = self.rnn(x)
        x = F.relu(h)
        return torch.sigmoid(self.fc_last(x))