from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_sizes: List[int],
        embedding_dim: int,
        output_dim: int,
    ):
        super(ConvNet, self).__init__()

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    stride=1,
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, embedding_dim)
        """

        # Apply each convolution → ReLU → max-over-time pooling
        conved = [F.relu(conv(x)) for conv in self.convs]  # [(B, F, L_out), ...]
        pooled = [
            F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2) for c in conved
        ]  # [(B, F), ...]

        cat = torch.cat(pooled, dim=1)  # (batch_size, F * len(kernel_sizes))

        dropped = self.dropout(cat)
        logits = self.fc(dropped)

        return logits
