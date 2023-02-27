import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def transform_wb_pesq_range(x: float) -> float:
    """The metric defined by ITU-T P.862 is often called 'PESQ score', which is defined
    for narrow-band signals and has a value range of [-0.5, 4.5] exactly. Here, we use the metric
    defined by ITU-T P.862.2, commonly known as 'wide-band PESQ' and will be referred to as "PESQ score".

    Args:
        x (float): Narrow-band PESQ score.

    Returns:
        (float): Wide-band PESQ score.
    """
    return 0.999 + (4.999 - 0.999) / (1 + math.exp(-1.3669 * x + 3.8224))


PESQRange: Tuple[float, float] = (
    1.0,  # P.862.2 uses a different input filter than P.862, and the lower bound of
    # the raw score is not -0.5 anymore. It's hard to figure out the true lower bound.
    # We are using 1.0 as a reasonable approximation.
    transform_wb_pesq_range(4.5),
)


class RangeSigmoid(nn.Module):
    def __init__(self, val_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        super(RangeSigmoid, self).__init__()
        assert isinstance(val_range, tuple) and len(val_range) == 2
        self.val_range: Tuple[float, float] = val_range
        self.sigmoid: nn.modules.Module = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sigmoid(x) * (self.val_range[1] - self.val_range[0]) + self.val_range[0]
        return out


class Encoder(nn.Module):
    """Encoder module that transform 1D waveform to 2D representations.

    Args:
        feat_dim (int, optional): The feature dimension after Encoder module. (Default: 512)
        win_len (int, optional): kernel size in the Conv1D layer. (Default: 32)
    """

    def __init__(self, feat_dim: int = 512, win_len: int = 32) -> None:
        super(Encoder, self).__init__()

        self.conv1d = nn.Conv1d(1, feat_dim, win_len, stride=win_len // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply waveforms to convolutional layer and ReLU layer.

        Args:
            x (torch.Tensor): Input waveforms. Tensor with dimensions `(batch, time)`.

        Returns:
            (torch,Tensor): Feature Tensor with dimensions `(batch, channel, frame)`.
        """
        out = x.unsqueeze(dim=1)
        out = F.relu(self.conv1d(out))
        return out


class SingleRNN(nn.Module):
    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn: nn.modules.Module = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: batch, seq, dim
        out, _ = self.rnn(x)
        out = self.proj(out)
        return out


class DPRNN(nn.Module):
    """*Dual-path recurrent neural networks (DPRNN)* :cite:`luo2020dual`.

    Args:
        feat_dim (int, optional): The feature dimension after Encoder module. (Default: 64)
        hidden_dim (int, optional): Hidden dimension in the RNN layer of DPRNN. (Default: 128)
        num_blocks (int, optional): Number of DPRNN layers. (Default: 6)
        rnn_type (str, optional): Type of RNN in DPRNN. Valid options are ["RNN", "LSTM", "GRU"]. (Default: "LSTM")
        d_model (int, optional): The number of expected features in the input. (Default: 256)
        chunk_size (int, optional): Chunk size of input for DPRNN. (Default: 100)
        chunk_stride (int, optional): Stride of chunk input for DPRNN. (Default: 50)
    """

    def __init__(
        self,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        rnn_type: str = "LSTM",
        d_model: int = 256,
        chunk_size: int = 100,
        chunk_stride: int = 50,
    ) -> None:
        super(DPRNN, self).__init__()

        self.num_blocks = num_blocks

        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for _ in range(num_blocks):
            self.row_rnn.append(SingleRNN(rnn_type, feat_dim, hidden_dim))
            self.col_rnn.append(SingleRNN(rnn_type, feat_dim, hidden_dim))
            self.row_norm.append(nn.GroupNorm(1, feat_dim, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, feat_dim, eps=1e-8))
        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim, d_model, 1),
            nn.PReLU(),
        )
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride

    def pad_chunk(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # input shape: (B, N, T)
        seq_len = x.shape[-1]

        rest = self.chunk_size - (self.chunk_stride + seq_len % self.chunk_size) % self.chunk_size
        out = F.pad(x, [self.chunk_stride, rest + self.chunk_stride])

        return out, rest

    def chunking(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        out, rest = self.pad_chunk(x)
        batch_size, feat_dim, seq_len = out.shape

        segments1 = out[:, :, : -self.chunk_stride].contiguous().view(batch_size, feat_dim, -1, self.chunk_size)
        segments2 = out[:, :, self.chunk_stride :].contiguous().view(batch_size, feat_dim, -1, self.chunk_size)
        out = torch.cat([segments1, segments2], dim=3)
        out = out.view(batch_size, feat_dim, -1, self.chunk_size).transpose(2, 3).contiguous()

        return out, rest

    def merging(self, x: torch.Tensor, rest: int) -> torch.Tensor:
        batch_size, dim, _, _ = x.shape
        out = x.transpose(2, 3).contiguous().view(batch_size, dim, -1, self.chunk_size * 2)
        out1 = out[:, :, :, : self.chunk_size].contiguous().view(batch_size, dim, -1)[:, :, self.chunk_stride :]
        out2 = out[:, :, :, self.chunk_size :].contiguous().view(batch_size, dim, -1)[:, :, : -self.chunk_stride]
        out = out1 + out2
        if rest > 0:
            out = out[:, :, :-rest]
        out = out.contiguous()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, rest = self.chunking(x)
        batch_size, _, dim1, dim2 = x.shape
        out = x
        for row_rnn, row_norm, col_rnn, col_norm in zip(self.row_rnn, self.row_norm, self.col_rnn, self.col_norm):
            row_in = out.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1).contiguous()
            row_out = row_rnn(row_in)
            row_out = row_out.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            row_out = row_norm(row_out)
            out = out + row_out

            col_in = out.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1).contiguous()
            col_out = col_rnn(col_in)
            col_out = col_out.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            col_out = col_norm(col_out)
            out = out + col_out
        out = self.conv(out)
        out = self.merging(out, rest)
        out = out.transpose(1, 2).contiguous()
        return out


class AutoPool(nn.Module):
    def __init__(self, pool_dim: int = 1) -> None:
        super(AutoPool, self).__init__()
        self.pool_dim: int = pool_dim
        self.softmax: nn.modules.Module = nn.Softmax(dim=pool_dim)
        self.register_parameter("alpha", nn.Parameter(torch.ones(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.softmax(torch.mul(x, self.alpha))
        out = torch.sum(torch.mul(x, weight), dim=self.pool_dim)
        return out


class SquimObjective(nn.Module):
    """Speech Quality and Intelligibility Measures (SQUIM) model that predicts **objective** metric scores
    for speech enhancement (e.g., STOI, PESQ, and SI-SDR).

    Args:
        encoder (torch.nn.Module): Encoder module to transform 1D waveform to 2D feature representation.
        dprnn (torch.nn.Module): DPRNN module to model sequential feature.
        branches (torch.nn.ModuleList): Transformer branches in which each branch estimate one objective metirc score.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dprnn: nn.Module,
        branches: nn.ModuleList,
    ):
        super(SquimObjective, self).__init__()
        self.encoder = encoder
        self.dprnn = dprnn
        self.branches = branches

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input waveforms. Tensor with dimensions `(batch, time)`.

        Returns:
            List(torch.Tensor): List of score Tenosrs. Each Tensor is with dimension `(batch,)`.
        """
        if x.ndim != 2:
            raise ValueError(f"The input must be a 2D Tensor. Found dimension {x.ndim}.")
        x = x / (torch.mean(x**2, dim=1, keepdim=True) ** 0.5 * 20)
        out = self.encoder(x)
        out = self.dprnn(out)
        scores = []
        for branch in self.branches:
            scores.append(branch(out).squeeze(dim=1))
        return scores


def _create_branch(d_model: int, nhead: int, metric: str) -> nn.modules.Module:
    """Create branch module after DPRNN model for predicting metric score.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): Number of heads in the multi-head attention model.
        metric (str): The metric name to predict.

    Returns:
        (nn.Module): Returned module to predict corresponding metric score.
    """
    layer1 = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.0, batch_first=True)
    layer2 = AutoPool()
    if metric == "stoi":
        layer3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, 1),
            RangeSigmoid(),
        )
    elif metric == "pesq":
        layer3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, 1),
            RangeSigmoid(val_range=PESQRange),
        )
    else:
        layer3: nn.modules.Module = nn.Sequential(nn.Linear(d_model, d_model), nn.PReLU(), nn.Linear(d_model, 1))
    return nn.Sequential(layer1, layer2, layer3)


def squim_objective_model(
    feat_dim: int,
    win_len: int,
    d_model: int,
    nhead: int,
    hidden_dim: int,
    num_blocks: int,
    rnn_type: str,
    chunk_size: int,
    chunk_stride: Optional[int] = None,
) -> SquimObjective:
    """Build a custome :class:`torchaudio.prototype.models.SquimObjective` model.

    Args:
        feat_dim (int, optional): The feature dimension after Encoder module.
        win_len (int): Kernel size in the Encoder module.
        d_model (int): The number of expected features in the input.
        nhead (int): Number of heads in the multi-head attention model.
        hidden_dim (int): Hidden dimension in the RNN layer of DPRNN.
        num_blocks (int): Number of DPRNN layers.
        rnn_type (str): Type of RNN in DPRNN. Valid options are ["RNN", "LSTM", "GRU"].
        chunk_size (int): Chunk size of input for DPRNN.
        chunk_stride (int or None, optional): Stride of chunk input for DPRNN.
    """
    if chunk_stride is None:
        chunk_stride = chunk_size // 2
    encoder = Encoder(feat_dim, win_len)
    dprnn = DPRNN(feat_dim, hidden_dim, num_blocks, rnn_type, d_model, chunk_size, chunk_stride)
    branches = nn.ModuleList(
        [
            _create_branch(d_model, nhead, "stoi"),
            _create_branch(d_model, nhead, "pesq"),
            _create_branch(d_model, nhead, "sisdr"),
        ]
    )
    return SquimObjective(encoder, dprnn, branches)


def squim_objective_base() -> SquimObjective:
    """Build :class:`torchaudio.prototype.models.SquimObjective` model with default arguments."""
    return squim_objective_model(
        feat_dim=256,
        win_len=64,
        d_model=256,
        nhead=4,
        hidden_dim=256,
        num_blocks=2,
        rnn_type="LSTM",
        chunk_size=71,
    )
