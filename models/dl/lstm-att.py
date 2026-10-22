
import torch
from torch import nn
import torch.nn.functional as F


class LSTM_ATT(nn.Module):
    """
    LSTM-ATT for MOOC dropout prediction.

    Architecture:
        LSTM -> Attention (Tanh -> Linear -> Softmax)
        -> Weighted Context -> Fully Connected -> Sigmoid

    Parameters:
        input_shape (tuple): (time_len, features)
        h_size (int): LSTM hidden size, default = 100
    """

    def __init__(self, input_shape, h_size=100):
        super(LSTM_ATT, self).__init__()

        time_len, n_features = input_shape

        # LSTM: input_size = number of features
        # hidden_size = 100
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=h_size,
            batch_first=True
        )

        # Attention: input size = 100, output size = 1
        self.attention = nn.Linear(h_size, 1)

        # Fully connected: input size = 100, output size = 1
        self.fc = nn.Linear(h_size, 1)

    def forward(self, x):
        # x shape: (batch_size, time_len, n_features)
        lstm_output, _ = self.lstm(x)

        # Apply Tanh before calculating attention scores
        attention_scores = torch.tanh(
            self.attention(lstm_output)
        )

        # Normalize attention scores over time
        attention_weights = F.softmax(
            attention_scores, dim=1
        )

        # Weighted context vector: (batch_size, 100)
        context = torch.sum(
            attention_weights * lstm_output,
            dim=1
        )

        # Fully connected layer + Sigmoid
        logits = self.fc(context)

        return torch.sigmoid(logits)
