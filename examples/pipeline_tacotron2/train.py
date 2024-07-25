# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
"""
Modified from
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/train.py
"""

import argparse
import logging
import os
import random
from datetime import datetime
from functools import partial
from time import time

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchaudio.models import Tacotron2
from tqdm import tqdm

plt.switch_backend("agg")

from datasets import SpectralNormalization, split_process_dataset, text_mel_collate_fn
from loss import Tacotron2Loss
from text.text_preprocessing import available_phonemizers, available_symbol_set, get_symbol_list, text_to_sequence
from utils import save_checkpoint


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(os.path.basename(__file__))


def parse_args(parser):
    """Parse commandline arguments."""

    parser.add_argument(
        "--dataset", default="ljspeech", choices=["ljspeech"], type=str, help="select dataset to train with"
    )
    parser.add_argument("--logging-dir", type=str, default=None, help="directory to save the log files")
    parser.add_argument("--dataset-path", type=str, default="./", help="path to dataset")
    parser.add_argument("--val-ratio", default=0.1, type=float, help="the ratio of waveforms for validation")

    parser.add_argument("--anneal-steps", nargs="*", help="epochs after which decrease learning rate")
    parser.add_argument(
        "--anneal-factor", type=float, choices=[0.1, 0.3], default=0.1, help="factor for annealing learning rate"
    )

    parser.add_argument("--master-addr", default=None, type=str, help="the address to use for distributed training")
    parser.add_argument("--master-port", default=None, type=str, help="the port to use for distributed training")

    preprocessor = parser.add_argument_group("text preprocessor setup")
    preprocessor.add_argument(
        "--text-preprocessor",
        default="english_characters",
        type=str,
        choices=available_symbol_set,
        help="select text preprocessor to use.",
    )
    preprocessor.add_argument(
        "--phonemizer",
        type=str,
        choices=available_phonemizers,
        help='select phonemizer to use, only used when text-preprocessor is "english_phonemes"',
    )
    preprocessor.add_argument(
        "--phonemizer-checkpoint",
        type=str,
        help="the path or name of the checkpoint for the phonemizer, "
        'only used when text-preprocessor is "english_phonemes"',
    )
    preprocessor.add_argument(
        "--cmudict-root", default="./", type=str, help="the root directory for storing cmudictionary files"
    )

    # training
    training = parser.add_argument_group("training setup")
    training.add_argument("--epochs", type=int, required=True, help="number of total epochs to run")
    training.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="checkpoint path. If a file exists, " "the program will load it and resume training.",
    )
    training.add_argument("--workers", default=8, type=int, help="number of data loading workers")
    training.add_argument(
        "--validate-and-checkpoint-freq",
        default=10,
        type=int,
        metavar="N",
        help="validation and saving checkpoint frequency in epochs",
    )
    training.add_argument("--logging-freq", default=10, type=int, metavar="N", help="logging frequency in epochs")

    optimization = parser.add_argument_group("optimization setup")
    optimization.add_argument("--learning-rate", default=1e-3, type=float, help="initial learing rate")
    optimization.add_argument("--weight-decay", default=1e-6, type=float, help="weight decay")
    optimization.add_argument("--batch-size", default=32, type=int, help="batch size per GPU")
    optimization.add_argument(
        "--grad-clip", default=5.0, type=float, help="clipping gradient with maximum gradient norm value"
    )

    # model parameters
    model = parser.add_argument_group("model parameters")
    model.add_argument("--mask-padding", action="store_true", default=False, help="use mask padding")
    model.add_argument("--symbols-embedding-dim", default=512, type=int, help="input embedding dimension")

    # encoder
    model.add_argument("--encoder-embedding-dim", default=512, type=int, help="encoder embedding dimension")
    model.add_argument("--encoder-n-convolution", default=3, type=int, help="number of encoder convolutions")
    model.add_argument("--encoder-kernel-size", default=5, type=int, help="encoder kernel size")
    # decoder
    model.add_argument(
        "--n-frames-per-step",
        default=1,
        type=int,
        help="number of frames processed per step (currently only 1 is supported)",
    )
    model.add_argument("--decoder-rnn-dim", default=1024, type=int, help="number of units in decoder LSTM")
    model.add_argument("--decoder-dropout", default=0.1, type=float, help="dropout probability for decoder LSTM")
    model.add_argument("--decoder-max-step", default=2000, type=int, help="maximum number of output mel spectrograms")
    model.add_argument(
        "--decoder-no-early-stopping",
        action="store_true",
        default=False,
        help="stop decoding only when all samples are finished",
    )

    # attention model
    model.add_argument(
        "--attention-hidden-dim", default=128, type=int, help="dimension of attention hidden representation"
    )
    model.add_argument("--attention-rnn-dim", default=1024, type=int, help="number of units in attention LSTM")
    model.add_argument(
        "--attention-location-n-filter", default=32, type=int, help="number of filters for location-sensitive attention"
    )
    model.add_argument(
        "--attention-location-kernel-size", default=31, type=int, help="kernel size for location-sensitive attention"
    )
    model.add_argument("--attention-dropout", default=0.1, type=float, help="dropout probability for attention LSTM")

    model.add_argument("--prenet-dim", default=256, type=int, help="number of ReLU units in prenet layers")

    # mel-post processing network parameters
    model.add_argument("--postnet-n-convolution", default=5, type=float, help="number of postnet convolutions")
    model.add_argument("--postnet-kernel-size", default=5, type=float, help="postnet kernel size")
    model.add_argument("--postnet-embedding-dim", default=512, type=float, help="postnet embedding dimension")

    model.add_argument("--gate-threshold", default=0.5, type=float, help="probability threshold for stop token")

    # audio parameters
    audio = parser.add_argument_group("audio parameters")
    audio.add_argument("--sample-rate", default=22050, type=int, help="Sampling rate")
    audio.add_argument("--n-fft", default=1024, type=int, help="Filter length for STFT")
    audio.add_argument("--hop-length", default=256, type=int, help="Hop (stride) length")
    audio.add_argument("--win-length", default=1024, type=int, help="Window length")
    audio.add_argument("--n-mels", default=80, type=int, help="")
    audio.add_argument("--mel-fmin", default=0.0, type=float, help="Minimum mel frequency")
    audio.add_argument("--mel-fmax", default=8000.0, type=float, help="Maximum mel frequency")

    return parser


def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps, anneal_factor):
    """Adjust learning rate base on the initial setting."""
    p = 0
    if anneal_steps is not None:
        for _, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor**p)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def batch_to_gpu(batch):
    text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths, gate_padded = batch
    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    mel_specgram_padded = to_gpu(mel_specgram_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    mel_specgram_lengths = to_gpu(mel_specgram_lengths).long()
    x = (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    y = (mel_specgram_padded, gate_padded)
    return x, y


def training_step(model, train_batch, batch_idx):
    (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths), y = batch_to_gpu(train_batch)
    y_pred = model(text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    y[0].requires_grad = False
    y[1].requires_grad = False
    losses = Tacotron2Loss()(y_pred[:3], y)
    return losses[0] + losses[1] + losses[2], losses


def validation_step(model, val_batch, batch_idx):
    (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths), y = batch_to_gpu(val_batch)
    y_pred = model(text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    losses = Tacotron2Loss()(y_pred[:3], y)
    return losses[0] + losses[1] + losses[2], losses


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / world_size
    else:
        rt = rt // world_size
    return rt


def log_additional_info(writer, model, loader, epoch):
    model.eval()
    data = next(iter(loader))
    with torch.no_grad():
        (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths), _ = batch_to_gpu(data)
        y_pred = model(text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
        mel_out, mel_out_postnet, gate_out, alignment = y_pred

    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(mel_out[0].cpu().numpy())
    writer.add_figure("trn/mel_out", fig, epoch)
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(mel_out_postnet[0].cpu().numpy())
    writer.add_figure("trn/mel_out_postnet", fig, epoch)
    writer.add_image("trn/gate_out", torch.tile(gate_out[:1], (10, 1)), epoch, dataformats="HW")
    writer.add_image("trn/alignment", alignment[0], epoch, dataformats="HW")


def get_datasets(args):
    text_preprocessor = partial(
        text_to_sequence,
        symbol_list=args.text_preprocessor,
        phonemizer=args.phonemizer,
        checkpoint=args.phonemizer_checkpoint,
        cmudict_root=args.cmudict_root,
    )

    transforms = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            f_min=args.mel_fmin,
            f_max=args.mel_fmax,
            n_mels=args.n_mels,
            mel_scale="slaney",
            normalized=False,
            power=1,
            norm="slaney",
        ),
        SpectralNormalization(),
    )
    trainset, valset = split_process_dataset(
        args.dataset, args.dataset_path, args.val_ratio, transforms, text_preprocessor
    )
    return trainset, valset


def train(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0 and args.logging_dir:
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        filehandler = logging.FileHandler(os.path.join(args.logging_dir, "train.log"))
        filehandler.setLevel(logging.INFO)
        logger.addHandler(filehandler)

        writer = SummaryWriter(log_dir=args.logging_dir)
    else:
        writer = None

    torch.manual_seed(0)

    torch.cuda.set_device(rank)

    symbols = get_symbol_list(args.text_preprocessor)

    model = Tacotron2(
        mask_padding=args.mask_padding,
        n_mels=args.n_mels,
        n_symbol=len(symbols),
        n_frames_per_step=args.n_frames_per_step,
        symbol_embedding_dim=args.symbols_embedding_dim,
        encoder_embedding_dim=args.encoder_embedding_dim,
        encoder_n_convolution=args.encoder_n_convolution,
        encoder_kernel_size=args.encoder_kernel_size,
        decoder_rnn_dim=args.decoder_rnn_dim,
        decoder_max_step=args.decoder_max_step,
        decoder_dropout=args.decoder_dropout,
        decoder_early_stopping=(not args.decoder_no_early_stopping),
        attention_rnn_dim=args.attention_rnn_dim,
        attention_hidden_dim=args.attention_hidden_dim,
        attention_location_n_filter=args.attention_location_n_filter,
        attention_location_kernel_size=args.attention_location_kernel_size,
        attention_dropout=args.attention_dropout,
        prenet_dim=args.prenet_dim,
        postnet_n_convolution=args.postnet_n_convolution,
        postnet_kernel_size=args.postnet_kernel_size,
        postnet_embedding_dim=args.postnet_embedding_dim,
        gate_threshold=args.gate_threshold,
    ).cuda(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    best_loss = float("inf")
    start_epoch = 0

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        logger.info(f"Checkpoint: loading '{args.checkpoint_path}'")
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        checkpoint = torch.load(args.checkpoint_path, map_location=map_location)

        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        logger.info(f"Checkpoint: loaded '{args.checkpoint_path}' at epoch {checkpoint['epoch']}")

    trainset, valset = get_datasets(args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        valset,
        shuffle=False,
        num_replicas=world_size,
        rank=rank,
    )

    loader_params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "prefetch_factor": 1024,
        "persistent_workers": True,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": partial(text_mel_collate_fn, n_frames_per_step=args.n_frames_per_step),
    }

    train_loader = DataLoader(trainset, sampler=train_sampler, **loader_params)
    val_loader = DataLoader(valset, sampler=val_sampler, **loader_params)
    dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        start = time()

        model.train()
        trn_loss, counts = 0, 0

        if rank == 0:
            iterator = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
        else:
            iterator = enumerate(train_loader)

        for i, batch in iterator:
            adjust_learning_rate(epoch, optimizer, args.learning_rate, args.anneal_steps, args.anneal_factor)

            model.zero_grad()

            loss, losses = training_step(model, batch, i)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            if rank == 0 and writer:
                global_iters = epoch * len(train_loader)
                writer.add_scalar("trn/mel_loss", losses[0], global_iters)
                writer.add_scalar("trn/mel_postnet_loss", losses[1], global_iters)
                writer.add_scalar("trn/gate_loss", losses[2], global_iters)

            trn_loss += loss * len(batch[0])
            counts += len(batch[0])

        trn_loss = trn_loss / counts

        trn_loss = reduce_tensor(trn_loss, world_size)
        if rank == 0:
            logger.info(f"[Epoch: {epoch}] time: {time()-start}; trn_loss: {trn_loss}")
            if writer:
                writer.add_scalar("trn_loss", trn_loss, epoch)

        if ((epoch + 1) % args.validate_and_checkpoint_freq == 0) or (epoch == args.epochs - 1):

            val_start_time = time()
            model.eval()

            val_loss, counts = 0, 0
            iterator = tqdm(enumerate(val_loader), desc=f"[Rank: {rank}; Epoch: {epoch}; Eval]", total=len(val_loader))

            with torch.no_grad():
                for val_batch_idx, val_batch in iterator:
                    val_loss = val_loss + validation_step(model, val_batch, val_batch_idx)[0] * len(val_batch[0])
                    counts = counts + len(val_batch[0])
                val_loss = val_loss / counts

            val_loss = reduce_tensor(val_loss, world_size)
            if rank == 0 and writer:
                writer.add_scalar("val_loss", val_loss, epoch)
                log_additional_info(writer, model, val_loader, epoch)

            if rank == 0:
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)
                logger.info(f"[Rank: {rank}, Epoch: {epoch}; Eval] time: {time()-val_start_time}; val_loss: {val_loss}")
                logger.info(f"[Epoch: {epoch}] Saving checkpoint to {args.checkpoint_path}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    args.checkpoint_path,
                )

    dist.destroy_process_group()


def main(args):
    logger.info("Start time: {}".format(str(datetime.now())))

    torch.manual_seed(0)
    random.seed(0)

    if args.master_addr is not None:
        os.environ["MASTER_ADDR"] = args.master_addr
    elif "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    if args.master_port is not None:
        os.environ["MASTER_PORT"] = args.master_port
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "17778"

    device_counts = torch.cuda.device_count()

    logger.info(f"# available GPUs: {device_counts}")

    # download dataset is not already downloaded
    if args.dataset == "ljspeech":
        if not os.path.exists(os.path.join(args.dataset_path, "LJSpeech-1.1")):
            from torchaudio.datasets import LJSPEECH

            LJSPEECH(root=args.dataset_path, download=True)

    if device_counts == 1:
        train(0, 1, args)
    else:
        mp.spawn(
            train,
            args=(
                device_counts,
                args,
            ),
            nprocs=device_counts,
            join=True,
        )

    logger.info(f"End time: {datetime.now()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Tacotron 2 Training")
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    main(args)
