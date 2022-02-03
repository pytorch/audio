#!/usr/bin/env python3
"""Train the HuBERTPretrain model by using 960 hours of LibriSpeech training sets.
Example:
python train.py --root-path ./exp/data/mfcc/ --feature-type mfcc --num-classes 100
"""

import logging
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter
from typing import Tuple

from lightning import HuBERTPreTrainModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger(__name__)


class _Formatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    # https://stackoverflow.com/a/18462760
    pass


def run_train(args):
    checkpoint_dir = args.exp_dir / f"checkpoints_{args.dataset}_{args.model_name}"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
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
        gpus=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        replace_sampler_ddp=False,
        gradient_clip_val=args.clip_norm,
        callbacks=callbacks,
    )

    model = HuBERTPreTrainModule(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dataset=args.dataset,
        root_path=args.root_path,
        feature_type=args.feature_type,
        seconds_per_batch=args.seconds_per_batch,
        learning_rate=args.learning_rate,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
    )
    trainer.fit(model)


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--root-path",
        type=pathlib.Path,
        required=True,
        help="Path to the feature and label directories.",
    )
    parser.add_argument(
        "--feature-type",
        choices=["mfcc", "hubert"],
        type=str,
        required=True,
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
        choices=["hubert_pretrain_base", "hubert_pretrain_large", "hubert_pretrain_xlarge"],
        type=str,
        help="The HuBERT model to train.",
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
        default=0.003,
        type=float,
    )
    parser.add_argument(
        "--betas",
        default=(0.9, 0.98),
        type=Tuple,
        help=" coefficients for computing running averages of gradient and its square (default: (0.9, 0.98))",
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
        default=1.0,
        type=float,
        help="The gradient norm value to clip. (Default: 1.0)",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
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
