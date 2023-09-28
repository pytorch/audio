import logging
import math
from collections import namedtuple
from typing import List, Tuple

import sentencepiece as spm
import torch
from bpe_graph_compiler import BpeCtcTrainingGraphCompiler

# from torchaudio.models import Conformer
from ctc_model import conformer_ctc_model
from loss import MaximumLikelihoodLoss
from pytorch_lightning import LightningModule
from torchaudio.models import Hypothesis


logger = logging.getLogger()

_expected_spm_vocab_size = 1023

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    r"""Learning rate scheduler that performs linear warmup and exponential annealing.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to use.
        warmup_steps (int): number of scheduler steps for which to warm up learning rate.
        force_anneal_step (int): scheduler step at which annealing of learning rate begins.
        anneal_factor (float): factor to scale base learning rate by at each annealing step.
        last_epoch (int, optional): The index of last epoch. (Default: -1)
        verbose (bool, optional): If ``True``, prints a message to stdout for
            each update. (Default: ``False``)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        force_anneal_step: int,
        anneal_factor: float,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.force_anneal_step = force_anneal_step
        self.anneal_factor = anneal_factor
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.force_anneal_step:
            return [(min(1.0, self._step_count / self.warmup_steps)) * base_lr for base_lr in self.base_lrs]
        else:
            scaling_factor = self.anneal_factor ** (self._step_count - self.force_anneal_step)
            return [scaling_factor * base_lr for base_lr in self.base_lrs]


def post_process_hypos(
    hypos: List[Hypothesis], sp_model: spm.SentencePieceProcessor
) -> List[Tuple[str, float, List[int], List[int]]]:
    tokens_idx = 0
    score_idx = 3
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    filtered_hypo_tokens = [
        [token_index for token_index in h[tokens_idx][1:] if token_index not in post_process_remove_list] for h in hypos
    ]
    hypos_str = [sp_model.decode(s) for s in filtered_hypo_tokens]
    hypos_ids = [h[tokens_idx][1:] for h in hypos]
    hypos_score = [[math.exp(h[score_idx])] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))

    return nbest_batch


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i.item()] for i in indices])
        return joined.replace("|", " ").strip().split()


def conformer_ctc_customized():
    return conformer_ctc_model(
        input_dim=80,
        encoding_dim=512,
        time_reduction_stride=1,
        conformer_input_dim=512,
        conformer_ffn_dim=2048,
        conformer_num_layers=12,
        conformer_num_heads=8,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        subsampling_type="conv",
    )


class ConformerCTCModule(LightningModule):
    def __init__(self, sp_model, inference_args=None):
        super().__init__()

        self.sp_model = sp_model
        spm_vocab_size = self.sp_model.get_piece_size()
        assert spm_vocab_size == _expected_spm_vocab_size, (
            "The model returned by conformer_rnnt_base expects a SentencePiece model of "
            f"vocabulary size {_expected_spm_vocab_size}, but the given SentencePiece model has a vocabulary size "
            f"of {spm_vocab_size}. Please provide a correctly configured SentencePiece model."
        )
        self.blank_idx = spm_vocab_size

        # ``conformer_rnnt_base`` hardcodes a specific Conformer RNN-T configuration.
        # For greater customizability, please refer to ``conformer_rnnt_model``.
        # self.model = conformer_ctc_model_base()
        self.model = conformer_ctc_customized()

        # Option 1:
        # self.loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction="sum")

        # Option 2:
        # graph_compiler = BpeCtcTrainingGraphCompiler(
        #     bpe_model_path="./spm_unigram_1023.model",
        #     device=self.device,  # torch.device("cuda", self.global_rank),
        #     topo_type="ctc",
        # )
        # self.loss = MaximumLikelihoodLoss(graph_compiler, subsampling_factor=4)
        self.loss = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8e-4, betas=(0.9, 0.98), eps=1e-9)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 40, 120, 0.96)

        if inference_args:
            tokens = {i: sp_model.id_to_piece(i) for i in range(sp_model.vocab_size())}
            greedy_decoder = GreedyCTCDecoder(
                labels=tokens,
                blank=self.blank_idx,
            )
            self.decoder = greedy_decoder

    def initialize_loss_func(self, topo_type="ctc", subsampling_factor=4):
        graph_compiler = BpeCtcTrainingGraphCompiler(
            bpe_model_path="./spm_unigram_1023.model",
            device=self.device,  # torch.device("cuda", self.global_rank),
            topo_type=topo_type,
        )
        self.loss = MaximumLikelihoodLoss(graph_compiler, subsampling_factor=subsampling_factor)

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        output, src_lengths = self.model(
            batch.features,
            batch.feature_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [{"scheduler": self.warmup_lr_scheduler, "interval": "epoch"}],
        )

    def forward(self, batch: Batch):
        with torch.inference_mode():
            output, src_lengths = self.model(
                batch.features.to(self.device),
                batch.feature_lengths.to(self.device),
            )
        emission = output.cpu()
        beam_search_result = self.decoder(emission)
        # beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()  # assuming batch_size=1
        beam_search_transcript = " ".join(beam_search_result).strip()
        return beam_search_transcript

    def training_step(self, batch: Batch, batch_idx):
        """Custom training step.

        By default, DDP does the following on each train step:
        - For each GPU, compute loss and gradient on shard of training data.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / N, where N is the world
          size (total number of GPUs).
        - Update parameters on each GPU.

        Here, we do the following:
        - For k-th GPU, compute loss and scale it by (N / B_total), where B_total is
          the sum of batch sizes across all GPUs. Compute gradient from scaled loss.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / B_total.
        - Update parameters on each GPU.

        Doing so allows us to account for the variability in batch sizes that
        variable-length sequential data yield.
        """
        try:
            loss = self._step(batch, batch_idx, "train")
        except BaseException:
            loss = 0
            for _model_param_name, model_param_value in self.model.named_parameters():  # encoder_output_layer.
                loss += model_param_value.abs().sum()
            loss = loss * 1e-5
            logger.info(f"[{self.global_rank}] batch {batch_idx} is bad")
        batch_size = batch.features.size(0)
        batch_sizes = self.all_gather(batch_size)
        self.log("Gathered batch size", batch_sizes.sum(), on_step=True, on_epoch=True)
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
