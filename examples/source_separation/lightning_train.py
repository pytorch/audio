#!/usr/bin/env python3

# pyre-strict
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypedDict, Union

import torch
import torchaudio
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from utils import metrics
from utils.dataset import utils as dataset_utils


class Batch(TypedDict):
    mix: torch.Tensor  # (batch, time)
    src: torch.Tensor  # (batch, source, time)
    mask: torch.Tensor  # (batch, source, time)


def sisdri_metric(
    estimate: torch.Tensor, reference: torch.Tensor, mix: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute the improvement of scale-invariant SDR. (SI-SDRi).

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Tensor of dimension (batch, speakers, time)
        reference (torch.Tensor): Reference (original) source signals.
            Tensor of dimension (batch, speakers, time)
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Tensor of dimension (batch, speakers == 1, time)
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Tensor of dimension (batch, 1, time)

    Returns:
        torch.Tensor: Improved SI-SDR. Tensor of dimension (batch, )

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


def sdri_metric(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mix: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the improvement of SDR. (SDRi).

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Tensor of dimension (batch, speakers, time)
        reference (torch.Tensor): Reference (original) source signals.
            Tensor of dimension (batch, speakers, time)
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Tensor of dimension (batch, speakers == 1, time)
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Tensor of dimension (batch, 1, time)

    Returns:
        torch.Tensor: Improved SDR. Tensor of dimension (batch, )

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
        Luo, Yi and Mesgarani, Nima
        https://arxiv.org/abs/1809.07454
    """
    with torch.no_grad():
        sdri = metrics.sdri(estimate, reference, mix, mask=mask)
    return sdri.mean().item()


def si_sdr_loss(estimate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the Si-SDR loss.

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Tensor of dimension (batch, speakers, time)
        reference (torch.Tensor): Reference (original) source signals.
            Tensor of dimension (batch, speakers, time)
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Tensor of dimension (batch, 1, time)

    Returns:
        torch.Tensor: Si-SDR loss. Tensor of dimension (batch, )
    """
    estimate = estimate - estimate.mean(axis=2, keepdim=True)
    reference = reference - reference.mean(axis=2, keepdim=True)

    si_sdri = metrics.sdr_pit(estimate, reference, mask=mask)
    return -si_sdri.mean()


class ConvTasNetModule(LightningModule):
    """
    The Lightning Module for speech separation.

    Args:
        model (Any): The model to use for the classification task.
        train_loader (DataLoader): the training dataloader.
        val_loader (DataLoader or None): the validation dataloader.
        loss (Any): The loss function to use.
        optim (Any): The optimizer to use.
        metrics (List of methods): The metrics to track, which will be used for both train and validation.
        lr_scheduler (Any or None): The LR Scheduler.
    """

    def __init__(
        self,
        model: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
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

        self.metrics: Mapping[str, Callable] = metrics

        self.train_metrics: Dict = {}
        self.val_metrics: Dict = {}
        self.test_metrics: Dict = {}

        self.save_hyperparameters()
        self.train_loader = train_loader
        self.val_loader = val_loader

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

    def training_step(self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Operates on a single batch of data from the validation set.
        """
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Operates on a single batch of data from the test set.
        """
        return self._step(batch, batch_idx, "test")

    def _step(self, batch: Batch, batch_idx: int, phase_type: str) -> Dict[str, Any]:
        """
        Common step for training, validation, and testing.
        """
        mix, src, mask = batch
        pred = self.model(mix)
        loss = self.loss(pred, src, mask)
        self.log(f"Losses/{phase_type}_loss", loss.item(), on_step=True, on_epoch=True)

        metrics_result = self._compute_metrics(pred, src, mix, mask, phase_type)
        self.log_dict(metrics_result, on_epoch=True)

        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[Any]:
        lr_scheduler = self.lr_scheduler
        if not lr_scheduler:
            return self.optim
        epoch_schedulers = {"scheduler": lr_scheduler, "monitor": "Losses/val_loss", "interval": "epoch"}
        return [self.optim], [epoch_schedulers]

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

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader


def _get_model(
    num_sources,
    enc_kernel_size=16,
    enc_num_feats=512,
    msk_kernel_size=3,
    msk_num_feats=128,
    msk_num_hidden_feats=512,
    msk_num_layers=8,
    msk_num_stacks=3,
    msk_activate="relu",
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
        msk_activate=msk_activate,
    )
    return model


def _get_dataloader(
    dataset_type: str,
    root_dir: Union[str, Path],
    num_speakers: int = 2,
    sample_rate: int = 8000,
    batch_size: int = 6,
    num_workers: int = 4,
    librimix_task: Optional[str] = None,
    librimix_tr_split: Optional[str] = None,
) -> Tuple[DataLoader]:
    """Get dataloaders for training, validation, and testing.

    Args:
        dataset_type (str): the dataset to use.
        root_dir (str or Path): the root directory of the dataset.
        num_speakers (int, optional): the number of speakers in the mixture. (Default: 2)
        sample_rate (int, optional): the sample rate of the audio. (Default: 8000)
        batch_size (int, optional): the batch size of the dataset. (Default: 6)
        num_workers (int, optional): the number of workers for each dataloader. (Default: 4)
        librimix_task (str or None, optional): the task in LibriMix dataset.
        librimix_tr_split (str or None, optional): the training split in LibriMix dataset.

    Returns:
        tuple: (train_loader, valid_loader, eval_loader)
    """
    train_dataset, valid_dataset, eval_dataset = dataset_utils.get_dataset(
        dataset_type, root_dir, num_speakers, sample_rate, librimix_task, librimix_tr_split
    )
    train_collate_fn = dataset_utils.get_collate_fn(dataset_type, mode="train", sample_rate=sample_rate, duration=3)

    test_collate_fn = dataset_utils.get_collate_fn(dataset_type, mode="test", sample_rate=sample_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=test_collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=test_collate_fn,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, eval_loader


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", default=6, type=int)
    parser.add_argument("--dataset", default="librimix", type=str, choices=["wsj0mix", "librimix"])
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="The path to the directory where the directory ``Libri2Mix`` or ``Libri3Mix`` is stored.",
    )
    parser.add_argument(
        "--librimix-tr-split",
        default="train-360",
        choices=["train-360", "train-100"],
        help="The training partition of librimix dataset. (default: ``train-360``)",
    )
    parser.add_argument(
        "--librimix-task",
        default="sep_clean",
        type=str,
        choices=["sep_clean", "sep_noisy", "enh_single", "enh_both"],
        help="The task to perform (separation or enhancement, noisy or clean). (default: ``sep_clean``)",
    )
    parser.add_argument(
        "--num-speakers", default=2, type=int, help="The number of speakers in the mixture. (default: 2)"
    )
    parser.add_argument(
        "--sample-rate",
        default=8000,
        type=int,
        help="Sample rate of audio files in the given dataset. (default: 8000)",
    )
    parser.add_argument(
        "--exp-dir", default=Path("./exp"), type=Path, help="The directory to save checkpoints and logs."
    )
    parser.add_argument(
        "--epochs",
        metavar="NUM_EPOCHS",
        default=200,
        type=int,
        help="The number of epochs to train. (default: 200)",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Initial learning rate. (default: 1e-3)",
    )
    parser.add_argument(
        "--num-gpu",
        default=1,
        type=int,
        help="The number of GPUs for training. (default: 1)",
    )
    parser.add_argument(
        "--num-node",
        default=1,
        type=int,
        help="The number of nodes in the cluster for training. (default: 1)",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="The number of workers for dataloader. (default: 4)",
    )

    args = parser.parse_args()

    model = _get_model(num_sources=args.num_speakers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    train_loader, valid_loader, eval_loader = _get_dataloader(
        args.dataset,
        args.root_dir,
        args.num_speakers,
        args.sample_rate,
        args.batch_size,
        args.num_workers,
        args.librimix_task,
        args.librimix_tr_split,
    )
    loss = si_sdr_loss
    metric_dict = {
        "sdri": sdri_metric,
        "sisdri": sisdri_metric,
    }
    model = ConvTasNetModule(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        loss=loss,
        optim=optimizer,
        metrics=metric_dict,
        lr_scheduler=lr_scheduler,
    )
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="Losses/val_loss", mode="min", save_top_k=5, save_weights_only=True, verbose=True
    )
    callbacks = [
        checkpoint,
        EarlyStopping(monitor="Losses/val_loss", mode="min", patience=30, verbose=True),
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_node,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        devices=args.num_gpu,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        callbacks=callbacks,
    )
    trainer.fit(model)
    model.load_from_checkpoint(checkpoint.best_model_path)
    state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    torch.save(state_dict, args.exp_dir / "best_model.pth")
    trainer.test(model, eval_loader)


if __name__ == "__main__":
    cli_main()
