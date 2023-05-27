import torch
import torch.nn as nn
from torchaudio.prototype.models.rnnt import _ConformerEncoder

import math
from typing import Dict, List, Optional, Tuple


class CTCModel(torch.nn.Module):
    r"""
    """

    def __init__(self, encoder, encoder_output_layer) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_output_layer = encoder_output_layer

    def ctc_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            The output tensor from the transformer encoder.
            Its shape is (B, T', D')

        Returns:
          Return a tensor that can be used for CTC decoding.
          Its shape is (B, T, V), where V is the number of classes
        """
        x = self.encoder_output_layer(x)
        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        x = nn.functional.log_softmax(x, dim=-1)  # (N, T, C)
        return x
        
    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: feature dimension of each source sequence element.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            predictor_state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing prediction network internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    joint network output, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing prediction network internal state generated in current invocation
                    of ``forward``.
        """
        source_encodings, source_lengths = self.encoder(
            input=sources,
            lengths=source_lengths,
        )

        ctc_log_prob = self.ctc_output(source_encodings)

        return (
            ctc_log_prob,
            source_lengths,
        )


def conformer_ctc_model(
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
):
    r"""Builds Conformer-based CTC model.

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
    encoder_output_layer = nn.Sequential(
        nn.Dropout(p=conformer_dropout), nn.Linear(encoding_dim, num_symbols)
    )

    return CTCModel(encoder, encoder_output_layer)


def conformer_ctc_model_base():
    return conformer_ctc_model(
        input_dim=80,
        encoding_dim=512,
        time_reduction_stride=4,
        conformer_input_dim=512,
        conformer_ffn_dim=2048,
        conformer_num_layers=12,
        conformer_num_heads=8,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
    )