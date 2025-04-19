import torch
import torch.nn as nn

# class LSTM(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers=1):
#         super(LSTM, self).__init__()
#         self.lstm = torch.nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#         )
#         self.fc = torch.nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
#         logits = self.fc(lstm_out[:, -1, :])  # Use the last time step's output

#         return logits


class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
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
