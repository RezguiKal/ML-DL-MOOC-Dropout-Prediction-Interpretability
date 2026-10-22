
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM model for MOOC dropout prediction.

    Architecture:
        Input -> Conv1D(6, kernel=5, padding=2) -> ReLU
              -> MaxPool1D(2, stride=2)
              -> Conv1D(16, kernel=5, padding=2) -> ReLU
              -> MaxPool1D(2, stride=2)
              -> LSTM(hidden_size=100)
              -> Concatenate CNN and LSTM representations
              -> FC1 -> ReLU
              -> FC2 -> ReLU
              -> FC3 -> Sigmoid

    Parameters:
        input_shape (tuple): (time_len, n_features)
        h_size (int): LSTM hidden size, default = 100
    """

    def __init__(self, input_shape, h_size=100):
        super(CNN_LSTM, self).__init__()

        time_len, n_features = input_shape

        self.h_size = h_size
        self.time_len = time_len

        # First Conv1D block
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=6,
            kernel_size=5,
            padding=2
        )
        self.pool1 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        # Second Conv1D block
        self.conv2 = nn.Conv1d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            padding=2
        )
        self.pool2 = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        # LSTM operates on the original input sequence
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=h_size,
            batch_first=True
        )

        # Compute CNN output length after two pooling layers
        cnn_length = time_len
        cnn_length = (cnn_length - 2) // 2 + 1
        cnn_length = (cnn_length - 2) // 2 + 1

        cnn_output_size = 16 * cnn_length

        # Concatenate flattened CNN output and final LSTM hidden state
        fusion_size = cnn_output_size + h_size

        # Fully connected layers
        self.fc1 = nn.Linear(fusion_size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # x: (batch_size, time_len, n_features)

        # CNN expects (batch_size, n_features, time_len)
        cnn_x = x.transpose(1, 2)

        cnn_x = F.relu(self.conv1(cnn_x))
        cnn_x = self.pool1(cnn_x)

        cnn_x = F.relu(self.conv2(cnn_x))
        cnn_x = self.pool2(cnn_x)

        # Flatten CNN features
        cnn_features = torch.flatten(cnn_x, start_dim=1)

        # LSTM branch processes the original sequence
        _, (h_n, _) = self.lstm(x)

        # Final layer's hidden state: (batch_size, h_size)
        lstm_features = h_n[-1]

        # Concatenate CNN and LSTM representations
        combined = torch.cat(
            (cnn_features, lstm_features),
            dim=1
        )

        # Classification head
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x
