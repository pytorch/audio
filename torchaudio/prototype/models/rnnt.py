from typing import List, Optional, Tuple

import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _CustomLSTM, _Joiner, _TimeReduction


class _Transcriber(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_reduction = _TimeReduction(4)
        self.input_linear = torch.nn.Linear(320, 256)
        self.conformer = Conformer(
            num_layers=16,
            input_dim=256,
            ffn_dim=1024,
            num_heads=4,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
            use_group_norm=True,
            convolution_first=True,
        )
        self.output_linear = torch.nn.Linear(256, 1024)
        self.layer_norm = torch.nn.LayerNorm(1024)

    def forward(self, input, lengths):
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, lengths = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, lengths


class _Predictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Args:
        num_symbols (int): size of target token lexicon.
        output_dim (int): feature dimension of each output sequence element.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_layer_norm (bool, optional): if ``True``, enables layer normalization
            for LSTM layers. (Default: ``False``)
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in
            LSTM layer normalization layers. (Default: 1e-5)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)

    """

    def __init__(
        self,
        num_symbols: int,
        output_dim: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_hidden_dim: int,
        lstm_layer_norm: bool = False,
        lstm_layer_norm_epsilon: float = 1e-5,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                _CustomLSTM(
                    symbol_embedding_dim if idx == 0 else lstm_hidden_dim,
                    lstm_hidden_dim,
                    layer_norm=lstm_layer_norm,
                    layer_norm_epsilon=lstm_layer_norm_epsilon,
                )
                for idx in range(num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=lstm_dropout)
        self.linear = torch.nn.Linear(lstm_hidden_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.lstm_dropout = lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else state[layer_idx])
            lstm_out = self.dropout(lstm_out)
            state_out.append(lstm_state_out)

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), lengths, state_out


def conformer_rnnt_base():
    encoder = _Transcriber()
    decoder = _Predictor(
        num_symbols=1024,
        output_dim=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
    )
    joiner = _Joiner(1024, 1024)
    return RNNT(encoder, decoder, joiner)
