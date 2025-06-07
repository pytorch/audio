import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.models.wav2vec2.components as components
from dataset import (
    _get_lengths_librilightlimited,
    _get_lengths_librispeech,
    BucketizeBatchSampler,
    CollateFnHubert,
    CollateFnLibriLightLimited,
    DistributedBatchSampler,
    HuBERTDataSet,
)
from lightning.pytorch import LightningModule
from loss import hubert_loss
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


Batch = Tuple[Tensor, Tensor, Tensor]
Batch_FineTune = Tuple[Tensor, Tensor, Tensor, Tensor]


class LinearDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        max_updates: int,
        last_epoch: int = -1,
    ):
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [self._step_count / self.warmup_updates * base_lr for base_lr in self.base_lrs]
        elif self._step_count >= self.max_updates:
            return [0.0 for _ in self.base_lrs]
        else:
            pct_remaining = (self.max_updates - self._step_count) / (self.max_updates - self.warmup_updates)
            return [base_lr * pct_remaining for base_lr in self.base_lrs]


class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]


def _compute_accuracy(logits: torch.Tensor):
    with torch.no_grad():
        max = logits.argmax(-1) == 0
        min = logits.argmin(-1) == 0
        both = max & min
        corr = max.long().sum().item() - both.long().sum().item()
        count = max.numel()
    return corr, count


def _reset_stats():
    return {
        "train": {
            "correct": 0.0,
            "count": 0.0,
        },
        "val": {
            "correct": 0.0,
            "count": 0.0,
        },
    }


class HuBERTPreTrainModule(LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        feature_grad_mult: float,
        num_classes: int,
        dataset: str,
        dataset_path: str,
        feature_type: str,
        seconds_per_batch: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        clip_norm: Optional[float],
        warmup_updates: int,
        max_updates: int,
    ):
        super().__init__()

        if model_name == "hubert_pretrain_base":
            self.model = torchaudio.models.hubert_pretrain_base(
                feature_grad_mult=feature_grad_mult, num_classes=num_classes
            )
        elif model_name == "hubert_pretrain_large":
            self.model = torchaudio.models.hubert_pretrain_large()
        elif model_name == "hubert_pretrain_xlarge":
            self.model = torchaudio.models.hubert_pretrain_xlarge()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler()

        self.loss = hubert_loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.clip_norm = clip_norm
        self.lr_scheduler = LinearDecayLRScheduler(self.optimizer, warmup_updates, max_updates)
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.seconds_per_batch = seconds_per_batch
        self.mask_stats = _reset_stats()
        self.unmask_stats = _reset_stats()
        self.nan_loss_count = 0.0

    def _step(self, batch: Batch, batch_idx, step_type):
        if batch is None:
            return None, None
        waveforms, labels, audio_lengths = batch
        if step_type == "val":
            with torch.no_grad():
                logit_m, logit_u, feature_penalty = self.model(
                    waveforms,
                    labels,
                    audio_lengths,
                )
        else:
            logit_m, logit_u, feature_penalty = self.model(
                waveforms,
                labels,
                audio_lengths,
            )
        loss = self.loss(logit_m, logit_u, feature_penalty)
        if not torch.isinf(loss) and not torch.isnan(loss):
            self.log(f"{step_type}_loss", loss.item() / logit_m.size(0), on_step=True, on_epoch=True)
        else:
            self.nan_loss_count += 1
            self.log("nan_loss_count", self.nan_loss_count, on_step=True, on_epoch=True)

        # log accuracies of masked and unmasked frames
        correct_m, count_m = _compute_accuracy(logit_m)
        correct_u, count_u = _compute_accuracy(logit_u)
        self.mask_stats[step_type]["correct"] += correct_m
        self.mask_stats[step_type]["count"] += count_m
        self.unmask_stats[step_type]["correct"] += correct_u
        self.unmask_stats[step_type]["count"] += count_u
        self.log(
            f"{step_type}_masked_accuracy",
            self.mask_stats[step_type]["correct"] / self.mask_stats[step_type]["count"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=step_type == "train",
        )
        self.log(
            f"{step_type}_unmasked_accuracy",
            self.unmask_stats[step_type]["correct"] / self.unmask_stats[step_type]["count"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=step_type == "train",
        )
        return loss, logit_m.size(0)

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [
                {
                    "scheduler": self.lr_scheduler,
                    "interval": "step",
                },
            ],
        )

    def training_step(self, batch: Batch, batch_idx):
        """Custom training step with loss normalization and automatic mixed precision training.

        By default, DDP does the following on each train step:
        - For each GPU, compute loss and gradient on shard of training data.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / N, where N is the world
          size (total number of GPUs).
        - Update parameters on each GPU.

        Here, we do the following:
        - For k-th GPU, compute loss and scale it by (N / num_frames), where num_frames is
          the sum of masked frames across all GPUs. Compute gradient from scaled loss.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / num_frames.
        - Update parameters on each GPU.

        Doing so allows us to account for the variability in number of masked frames in
        variable-length sequential data.
        """
        opt = self.optimizers()
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, num_frame = self._step(batch, batch_idx, "train")
        if torch.isinf(loss) or torch.isnan(loss):
            opt.zero_grad()
            return None

        # normalize the loss based on the sum of num_frame across all GPUs
        num_frames = self.all_gather(num_frame)
        self.log("Gathered number of frames", num_frames.float().sum(), on_step=True, on_epoch=True)
        loss *= num_frames.size(0) / num_frames.sum()  # world size / num_frames

        # backward the loss and clip the gradients
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.unscale_(opt)
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        # optimization
        self.scaler.step(opt)
        sch = self.lr_schedulers()
        sch.step()
        self.scaler.update()
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "val")[0]

    def on_validation_end(self):
        self.mask_stats = _reset_stats()
        self.unmask_stats = _reset_stats()

    def train_dataloader(self):
        dataset = HuBERTDataSet(self.dataset_path, self.dataset, "train")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=True,
            seed=self.trainer.current_epoch,
        )
        sampler = DistributedBatchSampler(sampler, shuffle=True)
        sampler.set_epoch(self.current_epoch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(feature_type=self.feature_type, pad=False, rand_crop=True),
            num_workers=10,
        )
        return dataloader

    def val_dataloader(self):
        dataset = HuBERTDataSet(self.dataset_path, self.dataset, "valid")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(feature_type=self.feature_type, pad=False, rand_crop=True),
            num_workers=10,
        )
        return dataloader


class HuBERTFineTuneModule(LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        encoder_projection_dropout: float,
        encoder_attention_dropout: float,
        encoder_ff_interm_dropout: float,
        encoder_dropout: float,
        encoder_layer_drop: float,
        mask_prob: float,
        mask_channel_prob: float,
        mask_channel_length: float,
        num_classes: int,
        aux_num_out: int,
        checkpoint: str,
        dataset_path: str,
        seconds_per_batch: float,
        subset: str,
        learning_rate: float,
        betas: Tuple[float, float],
        adam_eps: float,
        weight_decay: float,
        freeze_encoder_updates: int,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
    ):
        super().__init__()

        if model_name == "hubert_pretrain_base":
            self.model = torchaudio.models.hubert_pretrain_base(
                encoder_projection_dropout=encoder_projection_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                encoder_ff_interm_dropout=encoder_ff_interm_dropout,
                encoder_dropout=encoder_dropout,
                encoder_layer_drop=encoder_layer_drop,
                mask_prob=mask_prob,
                mask_channel_prob=mask_channel_prob,
                mask_channel_length=mask_channel_length,
                num_classes=num_classes,
            )
            self.aux = torch.nn.Linear(768, aux_num_out)
        elif model_name == "hubert_pretrain_large":
            self.model = torchaudio.models.hubert_pretrain_large(
                encoder_projection_dropout=encoder_projection_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                encoder_ff_interm_dropout=encoder_ff_interm_dropout,
                encoder_dropout=encoder_dropout,
                encoder_layer_drop=encoder_layer_drop,
                mask_prob=mask_prob,
                mask_channel_prob=mask_channel_prob,
                mask_channel_length=mask_channel_length,
                num_classes=num_classes,
            )
            self.aux = torch.nn.Linear(1024, aux_num_out)
        elif model_name == "hubert_pretrain_xlarge":
            self.model = torchaudio.models.hubert_pretrain_xlarge(
                encoder_projection_dropout=encoder_projection_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                encoder_ff_interm_dropout=encoder_ff_interm_dropout,
                encoder_dropout=encoder_dropout,
                encoder_layer_drop=encoder_layer_drop,
                mask_prob=mask_prob,
                mask_channel_prob=mask_channel_prob,
                mask_channel_length=mask_channel_length,
                num_classes=num_classes,
            )
            self.aux = torch.nn.Linear(1280, aux_num_out)
        else:
            raise ValueError(f"Unsupported model name: {model_name}.")
        self._load_checkpoint(checkpoint)
        for p in self.model.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False
        self.loss_fn = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        self.optimizer = torch.optim.AdamW(
            list(self.aux.parameters()) + list(self.model.parameters()),
            lr=learning_rate,
            betas=betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.freeze_encoder_updates = freeze_encoder_updates
        self.lr_scheduler = TriStageLRScheduler(self.optimizer, warmup_updates, hold_updates, decay_updates)
        self.dataset_path = dataset_path
        self.seconds_per_batch = seconds_per_batch
        self.subset = subset
        self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler()

    def _load_checkpoint(self, checkpoint):
        # load pretrain model from checkpoint
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        state_dict = state_dict["state_dict"]
        s = {}
        for k in state_dict:
            if "model." in k:
                s[k.replace("model.", "")] = state_dict[k]
        self.model.load_state_dict(s)

    def _step(self, batch: Batch_FineTune, batch_idx, step_type):
        if batch is None:
            return None
        waveforms, labels, audio_lengths, label_lengths = batch
        if self.global_step <= self.freeze_encoder_updates:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, audio_lengths)
                padding_mask = components._get_padding_mask(x, out_len)
                x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
                x, _ = self.model.mask_generator(x, padding_mask)
                x = self.model.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, audio_lengths)
                padding_mask = components._get_padding_mask(x, out_len)
            x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
            x, _ = self.model.mask_generator(x, padding_mask)
            x = self.model.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        logits = self.aux(x)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)
        loss = self.loss_fn(
            log_probs,
            labels,
            out_len,
            label_lengths,
        )
        self.log(f"{step_type}_loss", loss.item() / waveforms.size(0), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return (
            [
                self.optimizer,
            ],
            [
                {"scheduler": self.lr_scheduler, "interval": "step"},
            ],
        )

    def training_step(self, batch: Batch_FineTune, batch_idx):
        """Custom training step with loss normalization and automatic mixed precision training.

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
        with torch.cuda.amp.autocast(enabled=True):
            loss = self._step(batch, batch_idx, "train")

        # normalize the loss based on the sum of batch_sie across all GPUs
        batch_size = batch[0].size(0)
        batch_sizes = self.all_gather(batch_size)
        self.log("Gathered batch size", batch_sizes.sum(), on_step=True, on_epoch=True)
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size

        # backward the loss and clip the gradients
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

        # optimization
        self.scaler.step(opt)
        sch = self.lr_schedulers()
        sch.step()
        self.scaler.update()

    def validation_step(self, batch: Batch_FineTune, batch_idx):
        return self._step(batch, batch_idx, "val")

    def train_dataloader(self):
        dataset = torchaudio.datasets.LibriLightLimited(self.dataset_path, self.subset)
        lengths = _get_lengths_librilightlimited(dataset._fileids_paths, dataset._path, dataset._ext_audio)
        sampler = BucketizeBatchSampler(
            lengths,
            num_buckets=100,
            max_token_count=self.seconds_per_batch * 16000,
            shuffle=True,
            seed=self.global_step,
        )
        sampler = DistributedBatchSampler(sampler, shuffle=True)
        sampler.set_epoch(self.global_step)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnLibriLightLimited(),
            num_workers=10,
        )
        return dataloader

    def val_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "dev-other")
        lengths = _get_lengths_librispeech(dataset._walker, dataset._path, dataset._ext_audio)
        sampler = BucketizeBatchSampler(
            lengths, num_buckets=100, max_token_count=self.seconds_per_batch * 16000, shuffle=False
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnLibriLightLimited(),
            num_workers=10,
        )
        return dataloader
