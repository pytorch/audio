from typing import List, Optional, Tuple

import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber


class _ConformerEncoder(torch.nn.Module, _Transcriber):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        time_reduction_stride: int,
        conformer_input_dim: int,
        conformer_ffn_dim: int,
        conformer_num_layers: int,
        conformer_num_heads: int,
        conformer_depthwise_conv_kernel_size: int,
        conformer_dropout: float,
    ) -> None:
        super().__init__()
        self.time_reduction = _TimeReduction(time_reduction_stride)
        self.input_linear = torch.nn.Linear(input_dim * time_reduction_stride, conformer_input_dim)
        self.conformer = Conformer(
            num_layers=conformer_num_layers,
            input_dim=conformer_input_dim,
            ffn_dim=conformer_ffn_dim,
            num_heads=conformer_num_heads,
            depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
            dropout=conformer_dropout,
            use_group_norm=True,
            convolution_first=True,
        )
        self.output_linear = torch.nn.Linear(conformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, lengths = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, lengths

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise RuntimeError("Conformer does not support streaming inference.")


def conformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_hidden_dim: int,
    lstm_layer_norm: int,
    lstm_layer_norm_epsilon: int,
    lstm_dropout: int,
    joiner_activation: str,
) -> RNNT:
    r"""Builds Conformer-based recurrent neural network transducer (RNN-T) model.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        conformer_input_dim (int): dimension of Conformer input.
        conformer_ffn_dim (int): hidden layer dimension of each Conformer layer's feedforward network.
        conformer_num_layers (int): number of Conformer layers to instantiate.
        conformer_num_heads (int): number of attention heads in each Conformer layer.
        conformer_depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        conformer_dropout (float): Conformer dropout probability.
        num_symbols (int): cardinality of set of target tokens.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.
        joiner_activation (str): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

        Returns:
            RNNT:
                Conformer RNN-T model.
    """
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )
    predictor = _Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols, activation=joiner_activation)
    return RNNT(encoder, predictor, joiner)


def conformer_rnnt_base() -> RNNT:
    r"""Builds basic version of Conformer RNN-T model.

    Returns:
        RNNT:
            Conformer RNN-T model.
    """
    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        time_reduction_stride=4,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
    )
