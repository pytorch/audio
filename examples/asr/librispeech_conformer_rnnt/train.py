from argparse import ArgumentParser
import pathlib

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lightning import RNNTModule


def run_train(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint,
        train_checkpoint,
        lr_monitor,
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu",
        # strategy="ddp",
        strategy=DDPPlugin(find_unused_parameters=False),
        # gradient_clip_val=10.0,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )

    model = RNNTModule(
        librispeech_path=str(args.librispeech_path),
        sp_model_path=str(args.sp_model_path),
        global_stats_path=str(args.global_stats_path),
    )
    trainer.fit(model)
    # trainer.fit(model, ckpt_path="/fsx/users/jeffhwang/experiments_conformer_bias_dropout_shuffle_batch_manual_opt_lr_2e4/checkpoints/epoch=27-step=73583.ckpt")


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--global_stats_path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech_path", type=pathlib.Path, help="Path to LibriSpeech datasets.",
    )
    parser.add_argument(
        "--sp_model_path", type=pathlib.Path, help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--num_nodes",
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
        "--epochs",
        default=120,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
