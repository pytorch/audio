import logging
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter
from typing import Dict, Tuple

import torch
import torchaudio.models
from data_modules import HuBERTDataModule
from lightning import SSLPretrainModule
from losses import hubert_loss
from lr_schedulers import LinearDecayLRScheduler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything


class _Formatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    # To use ArgumentDefaultsHelpFormatter as the formatter_class and
    # RawDescriptionHelpFormatter to add custom formatting to description or epilog.
    # Check: https://stackoverflow.com/a/18462760
    pass


def _compute_accuracy(logits: torch.Tensor):
    with torch.no_grad():
        max = logits.argmax(-1) == 0
        min = logits.argmin(-1) == 0
        both = max & min
        corr = max.long().sum().item() - both.long().sum().item()
        count = max.numel()
    return corr / count


class HuBERTModule(SSLPretrainModule):
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

    def log_metric(self, batch: Dict, output: Tuple, loss: torch.Tensor, step_type: str):
        logit_m, logit_u, _ = output
        self.log(
            f"{step_type}_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
        )
        acc_m = _compute_accuracy(logit_m)
        acc_u = _compute_accuracy(logit_u)
        self.log(
            f"{step_type}_acc_m",
            acc_m,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=step_type == "train",
        )
        self.log(
            f"{step_type}_acc_u",
            acc_u,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=step_type == "train",
        )

    def get_sample_size(self, output):
        logit_m, _, _ = output
        return logit_m.shape[0]


def run_train(args):
    seed_everything(1337)
    checkpoint_dir = args.exp_dir / f"checkpoints_{args.dataset}_{args.model_name}"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    callbacks = [
        checkpoint,
        train_checkpoint,
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_steps=args.max_updates,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        replace_sampler_ddp=False,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )
    if args.model_name not in ["hubert_pretrain_base", "hubert_pretrain_large", "hubert_pretrain_xlarge"]:
        raise ValueError(
            "Expect model_name to be one of 'hubert_pretrain_base', 'hubert_pretrain_large', 'hubert_pretrain_xlarge'."
            f"Found {args.model_name}."
        )
    model = getattr(torchaudio.models, args.model_name)()
    loss_fn = hubert_loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LinearDecayLRScheduler(optimizer, args.warmup_updates, args.max_updates)
    lightning_module = HuBERTModule(
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        args.clip_norm,
    )
    data_module = HuBERTDataModule(
        dataset_path=args.dataset_path,
        dataset="librispeech",
        feature_type="mfcc",
        seconds_per_batch=200,
        train_shuffle=True,
        num_workers=10,
    )
    trainer.fit(lightning_module, datamodule=data_module)


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        required=True,
        help="Path to the feature and label directories.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to the feature and label directories. (Default: None)",
    )
    parser.add_argument(
        "--feature-type",
        choices=["mfcc", "hubert"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--feature-grad-mult",
        default=0.1,
        type=float,
        help="The scaling factor to multiply the feature extractor gradient. (Default: 0.1)",
    )
    parser.add_argument(
        "--num-classes",
        choices=[100, 500],
        type=int,
        required=True,
        help="The ``num_class`` when building the hubert_pretrain_base model.",
    )
    parser.add_argument(
        "--model-name",
        default="hubert_pretrain_base",
        choices=[
            "hubert_pretrain_base",
            "hubert_pretrain_large",
            "hubert_pretrain_xlarge",
        ],
        type=str,
        help="The HuBERT model to train. (Default: 'hubert_pretrain_base')",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--dataset",
        default="librispeech",
        choices=["librispeech", "librilight"],
        type=str,
        help="The dataset for training. (Default: 'librispeech')",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.0005,
        type=float,
        help="The peak learning rate. (Default: 0.0005)",
    )
    parser.add_argument(
        "--betas",
        default=(0.9, 0.98),
        type=Tuple,
        help="The coefficients for computing running averages of gradient and its square (Default: (0.9, 0.98))",
    )
    parser.add_argument(
        "--eps",
        default=1e-6,
        type=float,
        help="Epsilon value in Adam optimizer. (Default: 1e-6)",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.01,
        type=float,
        help="Weight decay (L2 penalty) (default: 0.01)",
    )
    parser.add_argument(
        "--clip-norm",
        default=10.0,
        type=float,
        help="The gradient norm value to clip. (Default: 10.0)",
    )
    parser.add_argument(
        "--num-nodes",
        default=4,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--warmup-updates",
        default=32000,
        type=int,
        help="Number of steps for warm up the learning rate. (Default: 32000)",
    )
    parser.add_argument(
        "--max-updates",
        default=250000,
        type=int,
        help="Total number of training steps. (Default: 250000)",
    )
    parser.add_argument(
        "--seconds-per-batch",
        default=87.5,
        type=float,
        help="Number of seconds of audio in a mini-batch. (Default: 87.5)",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    run_train(args)


if __name__ == "__main__":
    cli_main()
