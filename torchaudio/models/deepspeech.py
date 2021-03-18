import torch
import torch.nn as nn

__all__ = ["DeepSpeech"]


class FullyConnected(nn.Module):
    """
    Args:
        in_features: Number of input features
        hidden_size: Internal hidden unit size.
    """

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 dropout: float,
                 relu_max_clip: int = 20) -> None:
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_features, hidden_size, bias=True)
        self.nonlinearity = nn.Sequential(*[
            nn.ReLU(),
            nn.Hardtanh(0, relu_max_clip)
        ])
        if dropout:
            self.nonlinearity = nn.Sequential(*[
                self.nonlinearity,
                nn.Dropout(dropout)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.nonlinearity(x)
        return x


class DeepSpeech(nn.Module):
    """
    DeepSpeech model architecture from
    `"Deep Speech: Scaling up end-to-end speech recognition"`
    <https://arxiv.org/abs/1412.5567> paper.

    Args:
        in_features: Number of input features
        hidden_size: Internal hidden unit size.
        num_classes: Number of output classes
    """

    def __init__(self,
                 in_features: int,
                 hidden_size: int = 2048,
                 num_classes: int = 40,
                 dropout: float = 0.0) -> None:
        super(DeepSpeech, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = FullyConnected(in_features, hidden_size, dropout)
        self.fc2 = FullyConnected(hidden_size, hidden_size, dropout)
        self.fc3 = FullyConnected(hidden_size, hidden_size, dropout)
        self.bi_rnn = nn.RNN(
            hidden_size, hidden_size, num_layers=1, nonlinearity='relu', bidirectional=True)
        self.nonlinearity = nn.ReLU()
        self.fc4 = FullyConnected(hidden_size, hidden_size, dropout)
        self.out = nn.Sequential(*[
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=2)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N x C x T x F
        x = self.fc1(x)
        # N x C x T x H
        x = self.fc2(x)
        # N x C x T x H
        x = self.fc3(x)
        # N x C x T x H
        x = x.squeeze(1)
        # N x T x H
        x = x.transpose(0, 1)
        # T x N x H
        x, _ = self.bi_rnn(x)
        # The fifth (non-recurrent) layer takes both the forward and backward units as inputs
        x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]
        # T x N x H
        x = self.fc4(x)
        # T x N x H
        x = self.out(x)
        # T x N x num_classes
        return x
