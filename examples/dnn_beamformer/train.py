import pathlib
from argparse import ArgumentParser

import lightning.pytorch as pl
from datamodule import L3DAS22DataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from model import DNNBeamformer, DNNBeamformerLightningModule


def run_train(args):
    pl.seed_everything(1)
    logger = TensorBoardLogger(args.exp_dir)
    callbacks = [
        ModelCheckpoint(
            args.checkpoint_path,
            monitor="val/loss",
            save_top_k=5,
            mode="min",
            save_last=True,
        ),
    ]

    trainer = pl.trainer.trainer.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=args.gpus,
        accumulate_grad_batches=1,
        logger=logger,
        gradient_clip_val=5,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    model = DNNBeamformer()
    model_module = DNNBeamformerLightningModule(model)
    data_module = L3DAS22DataModule(dataset_path=args.dataset_path, batch_size=args.batch_size)

    trainer.fit(model_module, datamodule=data_module)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp/"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp/')",
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to L3DAS22 datasets.",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs to train for. (Default: 100)",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
