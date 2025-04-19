import torch
from omegaconf import DictConfig


class ConvNet(torch.nn.Module):
    def __init__(
        self, hyparams: DictConfig, in_channels: int, seq_len: int, output_dim: int
    ):
        super(ConvNet, self).__init__()

        self.kernel_sizes = hyparams.kernel_sizes

        agg_seq_len = 0
        for kernel_size in self.kernel_sizes:
            module_name = f"conv-maxpool-{kernel_size}"

            conv_seq_len = self.compute_seq_len(
                seq_len, kernel_size, hyparams.cnn_stride
            )
            pooling_seq_len = self.compute_seq_len(
                conv_seq_len, kernel_size, hyparams.pooling_stride
            )
            module = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hyparams.out_channels,
                    kernel_size=kernel_size,
                    stride=hyparams.cnn_stride,
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(
                    kernel_size=kernel_size, stride=hyparams.pooling_stride
                ),
            )

            agg_seq_len += pooling_seq_len
            setattr(self, module_name, module)

        self.flatten = torch.nn.Flatten()
        self.dropout1 = torch.nn.Dropout(p=hyparams.dropouts.p1)
        self.fc1 = torch.nn.Linear(
            hyparams.out_channels * agg_seq_len, hyparams.fc_features
        )
        self.dropout2 = torch.nn.Dropout(p=hyparams.dropouts.p2)
        self.fc2 = torch.nn.Linear(hyparams.fc_features, output_dim)

    @staticmethod
    def compute_seq_len(input_height: int, kernel_size: int, stride: int) -> int:
        return int((input_height - kernel_size) / stride) + 1

    def forward(self, batch):

        conv_maxpool_outputs = [
            getattr(self, f"conv-maxpool-{kernel_size}")(batch)
            for kernel_size in self.kernel_sizes
        ]
        x = torch.cat(conv_maxpool_outputs, axis=2)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
