import torch

from torch import nn
import torch.nn.functional as F

class SimpleLSTM(nn.Module):

    """
        Builds a SimpleLSTM object with the following layers:
         * lstm (relu) -> flatten -> fc (sigmoid)

        Params:
            - input_shape (tuple): the input shape (time_len, features)
            - h_size (int) : the size of the hidden vector of the LSTM
    """
    def __init__(self, input_shape, h_size=100):
        super(SimpleLSTM, self).__init__()

        h, w = input_shape

        self.lstm = nn.LSTM(input_size=w, hidden_size=h_size, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(h*h_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        x = F.relu(output)
        x = self.flatten(x)
        return torch.sigmoid(self.fc1(x))