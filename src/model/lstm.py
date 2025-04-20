import torch
import torch.nn as nn


class LSTM(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        x = self.dropout(lstm_out[:, -1, :])  # last timestep
        logits = self.fc(x)  # (batch, output_dim)

        return logits
