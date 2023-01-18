from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class SSLPretrainModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Union[Callable, nn.Module],
        optimizer: Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def configure_optimizers(self):
        pass

    def log_metric(self, batch, output, loss, step_type):
        pass

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            out = self.model(*batch["input"])
            (loss, num_frame) = self.loss_fn(*out, *batch["label"])
        self.log_metric(batch, out, loss, "train")

        # normalize the loss based on the sum of num_frame across all GPUs
        num_frames = self.all_gather(num_frame)
        self.log(
            "Gathered number of frames",
            num_frames.float().sum(),
            on_step=True,
            on_epoch=True,
        )
        loss *= num_frames.size(0) / num_frames.sum()  # world size / num_frames

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(*batch["input"])
        loss, _ = self.loss_fn(*out, *batch["label"])
        self.log_metric(batch, out, loss, "val")
        return loss
