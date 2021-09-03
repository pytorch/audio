#!/usr/bin/env python3

# pyre-strict

import pathlib
from argparse import ArgumentParser
from typing import (
    Any,
    Dict,
    Mapping,
    List,
    Optional,
    Tuple,
    TypedDict,
)

import torch
import torchaudio
import torchaudio.models
from pytorch_lightning import LightningModule, LightningDataModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from utils import metrics
from utils.dataset import utils as dataset_utils


class Batch(TypedDict):
    mix: torch.Tensor  # (batch, time)
    src: torch.Tensor  # (batch, source, time)
    mask: torch.Tensor  # (batch, source, time)


class SI_SDRi_Metric(nn.Module):
    def __init(self):
        super().__init__()

    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mix: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the improvement of scale-invariant SDR. (SI-SNRi).

        Args:
            estimate (torch.Tensor): Estimated source signals.
                Shape: [batch, speakers, time frame]
            reference (torch.Tensor): Reference (original) source signals.
                Shape: [batch, speakers, time frame]
            mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
                Shape: [batch, speakers == 1, time frame]
            mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
                Shape: [batch, 1, time frame]


        Returns:
            torch.Tensor: Improved SI-SDR. Shape: [batch, ]

        References:
            - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
            Luo, Yi and Mesgarani, Nima
            https://arxiv.org/abs/1809.07454
        """
        with torch.no_grad():
            estimate = estimate - estimate.mean(axis=2, keepdim=True)
            reference = reference - reference.mean(axis=2, keepdim=True)
            mix = mix - mix.mean(axis=2, keepdim=True)

            si_sdri = metrics.sdri(estimate, reference, mix, mask=mask)

        return si_sdri.mean().item()


class SDRi_Metric(nn.Module):
    def __init(self):
        super().__init__()

    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mix: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the improvement of SDR. (SDRi).

            Args:
                estimate (torch.Tensor): Estimated source signals.
                    Shape: [batch, speakers, time frame]
                reference (torch.Tensor): Reference (original) source signals.
                    Shape: [batch, speakers, time frame]
                mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
                    Shape: [batch, speakers == 1, time frame]
                mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
                    Shape: [batch, 1, time frame]


            Returns:
                torch.Tensor: Improved SDR. Shape: [batch, ]

            References:
                - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
                Luo, Yi and Mesgarani, Nima
                https://arxiv.org/abs/1809.07454
            """
        with torch.no_grad():
            sdri = metrics.sdri(estimate, reference, mix, mask=mask)
        return sdri.mean().item()


class SI_SNR(nn.Module):
    def __init(self):
        super().__init__()

    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        estimate = estimate - estimate.mean(axis=2, keepdim=True)
        reference = reference - reference.mean(axis=2, keepdim=True)

        si_sdri = metrics.sdr_pit(estimate, reference, mask=mask)
        return -si_sdri.mean()


class ConvTasNetModule(LightningModule):
    """
    This Lightning Module is used to perform single-channel source separation.

    Args:
        model: The model to use for the classification task.
        loss: The loss to use.
        optim: The optimizer to use.
        metrics: The metrics to track, which will be used for both train and validation.
        lr_scheduler: The LR Scheduler, optional.
    """

    def __init__(
        self,
        model: Any,
        loss: Any,
        optim: Any,
        metrics: List[Any],
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__()

        self.model: nn.Module = model
        self.loss: nn.Module = loss
        self.optim: torch.optim.Optimizer = optim
        self.lr_scheduler: Optional[_LRScheduler] = None
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler

        self.metrics: Mapping[str, nn.Module] = metrics

        self.train_metrics: nn.ModuleDict = nn.ModuleDict()
        self.val_metrics: nn.ModuleDict = nn.ModuleDict()
        self.test_metrics: nn.ModuleDict = nn.ModuleDict()

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_metrics.update(self.metrics)
            self.val_metrics.update(self.metrics)
        else:
            self.test_metrics.update(self.metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Operates on a single batch of data from the validation set.
        """
        return self._step(batch, batch_idx, "val")

    def test_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Operates on a single batch of data from the test set.
        """
        return self._step(batch, batch_idx, "test")

    def _step(self, batch: Batch, batch_idx: int, phase_type: str) -> Dict[str, Any]:
        mix, src, mask = batch
        pred = self.model(mix)
        loss = self.loss(pred, src, mask)
        self.log(f"Losses/{phase_type}_loss", loss.item(), on_step=True, on_epoch=True)

        metrics_result = self._compute_metrics(pred, src, mix, mask, phase_type)
        self.log_dict(metrics_result, on_epoch=True)

        return loss

    def configure_optimizers(
        self,
    ) -> Dict[str, Any]:
        lr_scheduler = self.lr_scheduler
        if not lr_scheduler:
            return self.optim
        return {
            'optimizer': self.optim,
            'lr_scheduler': lr_scheduler,
            'monitor': 'Losses/val_loss'
        }

    def _compute_metrics(
            self,
            pred: torch.Tensor,
            label: torch.Tensor,
            inputs: torch.Tensor,
            mask: torch.Tensor,
            phase_type: str,
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = getattr(self, f"{phase_type}_metrics")
        metrics_result = {}
        for name, metric in metrics_dict.items():
            metrics_result[f"Metrics/{phase_type}/{name}"] = metric(pred, label, inputs, mask)
        return metrics_result


class LibriMixDataModule(LightningDataModule):
    def __init__(
            self,
            dataset: str,
            dataset_dir: str,
            num_speakers: int = 2,
            sample_rate: int = 8000,
            batch_size: int = 12,
            num_workers: int = 8,
            task: str = 'sep_clean',
            librimix_tr_split: str = 'train-360',
    ) -> None:
        super().__init__()
        train_loader, valid_loader, eval_loader = _get_dataloader(
            dataset,
            dataset_dir,
            num_speakers,
            sample_rate,
            batch_size,
            num_workers,
            task,
            librimix_tr_split,
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.eval_loader


def _get_model(
    num_sources,
    enc_kernel_size=16,
    enc_num_feats=512,
    msk_kernel_size=3,
    msk_num_feats=128,
    msk_num_hidden_feats=512,
    msk_num_layers=8,
    msk_num_stacks=3,
):
    model = torchaudio.models.ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
    )
    return model


def _get_dataloader(
        dataset_type: str,
        dataset_dir: pathlib.Path,
        num_speakers: int = 2,
        sample_rate: int = 8000,
        batch_size: int = 6,
        num_workers: int = 4,
        librimix_task: Optional[str] = None,
        librimix_tr_split: Optional[str] = None,
) -> Tuple[DataLoader]:
    train_dataset, valid_dataset, eval_dataset = dataset_utils.get_dataset(
        dataset_type, dataset_dir, num_speakers, sample_rate, librimix_task, librimix_tr_split
    )
    train_collate_fn = dataset_utils.get_collate_fn(
        dataset_type, mode='train', sample_rate=sample_rate, duration=3
    )

    test_collate_fn = dataset_utils.get_collate_fn(dataset_type, mode='test', sample_rate=sample_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=test_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=test_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, eval_loader


def cli_main():
    seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--dataset", default="librimix", type=str, choices=["wsj0-mix", "librimix"])
    parser.add_argument("--data_dir", default=pathlib.Path("./Libri2Mix/wav8k/min"), type=pathlib.Path)
    parser.add_argument(
        "--librimix_tr_split",
        default="train-360",
        choices=["train-360", "train-100"],
        help="The training partition of librimix dataset. (default: ``train-360``)",
    )
    parser.add_argument(
        "--librimix_task",
        default="sep_clean",
        type=str,
        choices=["sep_clean", "sep_noisy", "enh_single", "enh_both"],
        help="The task to perform (separation or enhancement, noisy or clean). (default: ``sep_clean``)",
    )
    parser.add_argument(
        "--num_speakers", default=2, type=int, help="The number of speakers in the mixture. (default: 2)"
    )
    parser.add_argument(
        "--sample_rate",
        default=8000,
        type=int,
        help="Sample rate of audio files in the given dataset. (default: 8000)",
    )
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="The directory to save checkpoints and logs."
    )
    parser.add_argument(
        "--epochs",
        metavar="NUM_EPOCHS",
        default=200,
        type=int,
        help="The number of epochs to train. (default: 200)",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Initial learning rate. (default: 1e-3)",
    )
    parser.add_argument(
        "--num_gpu",
        default=4,
        type=int,
        help="The number of GPUs for training. (default: 4)",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="The number of workers for dataloader. (default: 4)",
    )

    args = parser.parse_args()

    model = _get_model(num_sources=args.num_speakers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    data_module = LibriMixDataModule(
        args.dataset,
        args.data_dir,
        args.num_speakers,
        args.sample_rate,
        args.batch_size,
        args.num_workers,
        args.librimix_task,
        args.librimix_tr_split,
    )
    loss = SI_SNR()
    metric_dict = {
        "sdri": SDRi_Metric(),
        "sisdri": SI_SDRi_Metric(),
    }
    model = ConvTasNetModule(
        model=model,
        loss=loss,
        optim=optimizer,
        metrics=metric_dict,
        lr_scheduler=lr_scheduler,
    )
    checkpoint_dir = args.exp_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            checkpoint_dir, monitor="Losses/val_loss", mode="min", verbose=True
        ),
        EarlyStopping(monitor="Losses/val_loss", mode="min", patience=30, verbose=True),
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        gpus=args.num_gpu,
        accelerator="ddp",
        plugins=DDPPlugin(find_unused_parameters=False),
        gradient_clip_val=5.0,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    cli_main()
