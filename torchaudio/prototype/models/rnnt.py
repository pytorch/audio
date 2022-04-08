from typing import List, Optional, Tuple

import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber


class _ConformerTranscriber(torch.nn.Module, _Transcriber):
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

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise RuntimeError("Conformer does not support streaming inference.")


def conformer_rnnt_base():
    r"""Builds basic version of Conformer RNN-T model.

        Returns:
            RNNT:
                Conformer RNN-T model.
    """
    encoder = _ConformerTranscriber()
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
    joiner = _Joiner(1024, 1024, activation="tanh")
    return RNNT(encoder, decoder, joiner)
