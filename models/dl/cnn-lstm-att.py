
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with base = 1000."""

    def __init__(self, d_model=256, base=1000):
        super().__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        device, dtype = x.device, x.dtype

        positions = torch.arange(
            seq_len, device=device, dtype=dtype
        ).unsqueeze(1)

        even_dims = torch.arange(
            0, self.d_model, 2, device=device, dtype=dtype
        )

        div_term = torch.exp(
            -math.log(self.base) * even_dims / self.d_model
        )

        angles = positions * div_term

        pe = torch.zeros(
            seq_len, self.d_model, device=device, dtype=dtype
        )
        pe[:, 0::2] = torch.sin(angles)

        odd_count = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(angles[:, :odd_count])

        return x + pe.unsqueeze(0)


class CNN_LSTM_ATT(nn.Module):
    """
    Parallel CNN-LSTM with positional encoding and multi-head attention.

    Input:  (batch_size, time_len, n_features)
    Output: (batch_size, 1), dropout probability
    """

    def __init__(
        self,
        input_shape,
        lstm_hidden_size=256,
        num_heads=16,
        attention_dropout=0.2,
        fc_dropout=0.5,
        positional_base=1000,
    ):
        super().__init__()

        time_len, n_features = input_shape

        # CNN branch
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=16,
            kernel_size=5,
            padding=2,
        )

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            padding=2,
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
        )

        # CNN output dimension after one pooling layer
        cnn_length = time_len // 2
        cnn_output_size = 16 * cnn_length

        # LSTM branch processes the original input
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=lstm_hidden_size,
            base=positional_base,
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(lstm_hidden_size)

        # Fusion: CNN features + attention-enhanced LSTM features
        fusion_size = cnn_output_size + lstm_hidden_size

        # Fully connected classification head
        self.fc1 = nn.Linear(fusion_size, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

        self.dropout = nn.Dropout(fc_dropout)

    def forward(self, x):
        # Input: (batch, time_len, n_features)

        # ----- CNN branch -----
        cnn_x = x.transpose(1, 2)

        cnn_x = F.relu(self.conv1(cnn_x))
        cnn_x = F.relu(self.conv2(cnn_x))
        cnn_x = self.max_pool(cnn_x)

        cnn_features = torch.flatten(cnn_x, start_dim=1)

        # ----- LSTM-attention branch -----
        lstm_output, _ = self.lstm(x)

        encoded = self.positional_encoding(lstm_output)

        attended, attention_weights = self.attention(
            encoded,
            encoded,
            encoded,
            need_weights=True,
            average_attn_weights=False,
        )

        # Residual connection and normalization
        attended = self.attention_norm(encoded + attended)

        # Temporal mean pooling
        att_features = attended.mean(dim=1)

        # ----- Feature fusion -----
        fused = torch.cat(
            [cnn_features, att_features],
            dim=1,
        )

        # ----- Classification -----
        fused = self.dropout(F.relu(self.fc1(fused)))
        fused = self.dropout(F.relu(self.fc2(fused)))

        logits = self.fc3(fused)

        return torch.sigmoid(logits)
