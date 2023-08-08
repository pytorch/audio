from typing import List, Optional, Union, Tuple
import k2
import torch
from torch import Tensor, nn

from bpe_graph_compiler import BpeCtcTrainingGraphCompiler


class MaximumLikelihoodLoss(nn.Module):
    """
    Computes maximum likelihood loss.

    TODO: more detailed description
    """

    def __init__(
        self,
        graph_compiler: BpeCtcTrainingGraphCompiler,
        padding_value=1.0,
        subsampling_factor: int = 4,
        ctc_beam_size: float = 10.0,
        reduction = "sum",
        use_double_scores = True,
    ):
        super().__init__()
        self.graph_compiler = graph_compiler
        self.padding_value = padding_value

        self.subsampling_factor = subsampling_factor
        self.ctc_beam_size = ctc_beam_size
        self.reduction = reduction
        self.use_double_scores = use_double_scores

    def encode_supervisions(
        self, targets, input_lengths
    ) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
        """
        Encodes Lhotse's ``batch["supervisions"]`` dict into
        a pair of torch Tensor, and a list of transcription strings or token indexes

        The supervision tensor has shape ``(batch_size, 3)``.
        Its second dimension contains information about sequence index [0],
        start frames [1] and num frames [2].

        The batch items might become re-ordered during this operation -- the
        returned tensor and list of strings are guaranteed to be consistent with
        each other.
        """
        batch_size = targets.size(0)
        supervision_segments = torch.stack(
            (
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]

        res = targets[indices].tolist()
        res = [[i + 1 for i in l if i != self.padding_value] for l in res]  # hard-coded for torchaudio

        return supervision_segments, res

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        supervision_segments, texts = self.encode_supervisions(targets, input_lengths)
        token_ids = texts
        
        decoding_graph = self.graph_compiler.compile(token_ids)

        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        log_probs = torch.roll(log_probs, 1, -1)  # Now blank symbol has the index of 0

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,
            supervision_segments,
            allow_truncate=self.subsampling_factor - 1,
        )

        loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.ctc_beam_size,
            reduction=self.reduction,
            use_double_scores=self.use_double_scores,
        )
        return loss
    