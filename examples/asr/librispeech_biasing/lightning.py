import logging
import math
from collections import namedtuple
from typing import List, Tuple

import sentencepiece as spm
import torch
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.prototype.models import Hypothesis, RNNTBeamSearchBiasing
from torchaudio.prototype.models import conformer_rnnt_biasing_base


logger = logging.getLogger()

_expected_spm_vocab_size = 1023

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths", "tries"])


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


class ConformerRNNTModule(LightningModule):
    def __init__(self, sp_model):
        super().__init__()

        self.sp_model = sp_model
        self.sp_model = spm.SentencePieceProcessor(model_file=self.sp_model)
        spm_vocab_size = self.sp_model.get_piece_size()
        self.char_list = [self.sp_model.id_to_piece(idx) for idx in range(spm_vocab_size)]
        assert spm_vocab_size == _expected_spm_vocab_size, (
            "The model returned by conformer_rnnt_base expects a SentencePiece model of "
            f"vocabulary size {_expected_spm_vocab_size}, but the given SentencePiece model has a vocabulary size "
            f"of {spm_vocab_size}. Please provide a correctly configured SentencePiece model."
        )
        self.blank_idx = spm_vocab_size
        self.char_list.append('<blank>')

        # ``conformer_rnnt_base`` hardcodes a specific Conformer RNN-T configuration.
        # For greater customizability, please refer to ``conformer_rnnt_model``.
        self.model = conformer_rnnt_biasing_base(charlist=self.char_list)
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8e-4, betas=(0.9, 0.98), eps=1e-9)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 40, 120, 0.96)

        self.automatic_optimization = False

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _, tcpgen_dist, p_gen = self.model(
            batch.features,
            batch.feature_lengths,
            prepended_targets,
            prepended_target_lengths,
            batch.tries,
            self.current_epoch
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        if tcpgen_dist is not None and p_gen is not None:
            loss += tcpgen_dist.mean() * 0
            loss += p_gen.mean() * 0
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True, batch_size=batch.targets.size(0))

        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [{"scheduler": self.warmup_lr_scheduler, "interval": "epoch"}],
        )

    def forward(self, batch: Batch):
        decoder = RNNTBeamSearchBiasing(self.model, self.blank_idx, trie=batch.tries)
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device), 10)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

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
        variable-length sequential data commonly yields.
        """

        opt = self.optimizers()
        opt.zero_grad()
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch.features.size(0)
        batch_sizes = self.all_gather(batch_size)
        self.log("Gathered batch size", batch_sizes.sum(), on_step=True, on_epoch=True, batch_size=batch.targets.size(0))
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        opt.step()

        # step every epoch
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            sch.step()

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")
