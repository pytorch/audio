import pathlib
from argparse import ArgumentParser
from typing import Tuple

from lightning import HuBERTModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


def run_train(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
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
        plugins=DDPPlugin(find_unused_parameters=False),  # make sure there is no unused params
        replace_sampler_ddp=False,
        gradient_clip_val=args.clip_norm,
        callbacks=callbacks,
    )

    model = HuBERTModule(
        dataset=args.dataset,
        root_path=args.root_path,
        feature_type=args.feature_type,
        learning_rate=args.learning_rate,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
    )
    trainer.fit(model)


def cli_main():
    parser = ArgumentParser()
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
        "--root-path",
        type=pathlib.Path,
        help="Path to the feature and label directories.",
    )
    parser.add_argument(
        "--feature-type",
        default="mfcc",
        choices=["mfcc", "hubert"],
        type=str,
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
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
