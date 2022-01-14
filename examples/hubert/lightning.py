from typing import Tuple

import torch
import torchaudio
from dataset import BucketizeSampler, DistributedBatchSampler, HuBERTDataSet, CollateFnHubert
from loss import hubert_loss
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


Batch = Tuple[Tensor, Tensor, Tensor]


class PolynomialDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """PolynomialDecay learning rate scheduler with warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        max_updates: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.warmup_factor = 1.0 / self.warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self.optimizer._step_count <= self.warmup_updates:
            self.warmup_factor = self.optimizer._step_count / self.warmup_updates
            return [self.warmup_factor * group["lr"] for group in self.optimizer.param_groups]
        elif self.optimizer._step_count >= self.max_updates:
            return [0.0 for _ in self.optimizer.param_groups]
        else:
            lrs = []
            for group in self.optimizer.param_groups:
                pct_remaining = (self.max_updates - self.optimizer._step_count) / (
                    self.max_updates - self.warmup_updates
                )
                lr = group["lr"] * pct_remaining
                lrs.append(lr)
            return lrs


class HuBERTModule(LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        dataset: str,
        root_path: str,
        feature_type: str,
        seconds_per_batch: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        warmup_updates: int,
        max_updates: int,
    ):
        super().__init__()

        if model_name == "huebrt_pretrain_base":
            self.model = torchaudio.models.hubert_pretrain_base(num_classes=num_classes)
        elif model_name == "huebrt_pretrain_large":
            self.model = torchaudio.models.hubert_pretrain_large()
        elif model_name == "huebrt_pretrain_xlarge":
            self.model = torchaudio.models.hubert_pretrain_xlarge()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.loss = hubert_loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.lr_scheduler = PolynomialDecayLRScheduler(self.optimizer, warmup_updates, max_updates)
        self.dataset = dataset
        self.root_path = root_path
        self.feature_type = feature_type
        self.seconds_per_batch = seconds_per_batch

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None
        waveforms, labels, audio_lengths = batch
        logit_m, logit_u, feature_pen = self.model(
            waveforms,
            labels,
            audio_lengths,
        )
        loss = self.loss(logit_m, logit_u, feature_pen)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [
                {
                    "scheduler": self.lr_scheduler,
                    "monitor": "Losses/val_loss",
                    "interval": "epoch",
                },
            ],
        )

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def train_dataloader(self):
        dataset = HuBERTDataSet(self.root_path, self.dataset, "train")
        sampler = BucketizeSampler(dataset, num_buckets=1000, max_token_count=self.seconds_per_batch * 16000)
        sampler = DistributedBatchSampler(sampler)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(feature_type=self.feature_type, pad=False, rand_crop=True),
            num_workers=10,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = HuBERTDataSet(self.root_path, self.dataset, "valid")
        sampler = BucketizeSampler(dataset, num_buckets=1000, max_token_count=self.seconds_per_batch * 16000)
        sampler = DistributedBatchSampler(sampler)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(feature_type=self.feature_type, pad=False, rand_crop=True),
            num_workers=10,
            pin_memory=True,
        )
        return dataloader
