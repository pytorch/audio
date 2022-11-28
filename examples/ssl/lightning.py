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
        clip_norm: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clip_norm = clip_norm
        self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler()

    def configure_optimizers(self):
        pass

    def log_metric(self, batch, output, loss, step_type):
        pass

    def get_sample_size(self, output):
        pass

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            out = self.model(*batch["input"])
            loss = self.loss_fn(*out, *batch["label"])
        self.log_metric(batch, out, loss, "train")

        # normalize the loss based on the sum of num_frame across all GPUs
        sample_size = self.get_sample_size(out)
        sample_sizes = self.all_gather(sample_size)
        self.log(
            "Gathered number of frames",
            sample_sizes.float().sum(),
            on_step=True,
            on_epoch=True,
        )
        loss *= sample_sizes.size(0) / sample_sizes.sum()  # world size / sample_sizes

        # backward the loss and clip the gradients
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.unscale_(opt)
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        # optimization
        self.scaler.step(opt)
        # check if lr scheduler is called for every epoch or every iteration
        if self.trainer.lr_scheduler_configs[0].interval == "step" or (
            self.trainer.lr_scheduler_configs[0].interval == "epoch"
            and self.trainer.is_last_batch
        ):
            sch = self.lr_schedulers()
            sch.step()
        self.scaler.update()
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(*batch["input"])
        loss = self.loss_fn(*out, *batch["label"])
        self.log_metric(batch, out, loss, "val")
        return loss
