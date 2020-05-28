import argparse
import collections
import itertools
import os
import shutil
import signal
import statistics
import string
from collections import defaultdict
from datetime import datetime
from typing import Optional

import torch
import torchaudio
from torch import nn, topk
from torch.optim import SGD, Adadelta, Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH
from torchaudio.datasets.utils import bg_iterator, diskcache_iterator
from torchaudio.models.wav2letter import Wav2Letter
from torchaudio.transforms import MFCC, Resample
from tqdm import tqdm

SIGNAL_RECEIVED = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint.pth.tar",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint",
    )

    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="manual epoch number"
    )
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency in epochs",
    )
    parser.add_argument(
        "--progress-bar", action="store_true", help="use progress bar while training"
    )

    parser.add_argument(
        "--arch",
        metavar="ARCH",
        default="wav2letter",
        choices=["wav2letter"],
        help="model architecture",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
    )

    parser.add_argument(
        "--n-bins",
        default=13,
        type=int,
        metavar="N",
        help="number of bins in transforms",
    )
    parser.add_argument(
        "--learning-rate",
        default=1.0,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--gamma",
        default=0.96,
        type=float,
        metavar="GAMMA",
        help="learning rate exponential decay constant",
    )
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        "--weight-decay", default=1e-5, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument("--eps", metavar="EPS", type=float, default=1e-8)
    parser.add_argument("--rho", metavar="RHO", type=float, default=0.95)
    parser.add_argument("--clip-norm", metavar="NORM", type=float, default=0.0)

    parser.add_argument(
        "--dataset",
        default="librispeech",
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument(
        "--distributed", action="store_true", help="enable DistributedDataParallel"
    )
    parser.add_argument("--jit", action="store_true", help="if used, model is jitted")

    args = parser.parse_args()

    return args


def signal_handler(a, b):
    global SIGNAL_RECEIVED
    print("Signal received", a, datetime.now().strftime("%y%m%d.%H%M%S"), flush=True)
    SIGNAL_RECEIVED = True


def save_checkpoint(state, is_best, filename):
    """
    Save the model to a temporary file first,
    then copy it to filename, in case the signal interrupts
    the torch.save() process.
    """
    tempfile = filename + ".temp"

    # Remove tempfile, in case the signal arrives in the
    # middle of copying from tempfile to filename
    if os.path.isfile(tempfile):
        os.remove(tempfile)

    torch.save(state, tempfile)
    if os.path.isfile(tempfile):
        os.rename(tempfile, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")
    print("Checkpoint: saved")


class LanguageModel:
    def __init__(self, labels, char_blank, char_space):

        self.char_space = char_space
        self.char_blank = char_blank

        labels = [l for l in labels]
        self.length = len(labels)
        enumerated = list(enumerate(labels))
        flipped = [(sub[1], sub[0]) for sub in enumerated]

        d1 = collections.OrderedDict(enumerated)
        d2 = collections.OrderedDict(flipped)
        self.mapping = {**d1, **d2}

    def encode(self, iterable):
        if isinstance(iterable, list):
            return [self.encode(i) for i in iterable]
        else:
            return [self.mapping[i] + self.mapping[self.char_blank] for i in iterable]

    def decode(self, tensor):
        if isinstance(tensor[0], list):
            return [self.decode(t) for t in tensor]
        else:
            # not idempotent, since clean string
            x = (self.mapping[i] for i in tensor)
            x = "".join(i for i, _ in itertools.groupby(x))
            x = x.replace(self.char_blank, "")
            # x = x.strip()
            return x


def model_length_function(tensor):
    return int(tensor.shape[0]) // 2 + 1


class IterableMemoryCache:
    def __init__(self, iterable):
        self.iterable = iterable
        self._iter = iter(iterable)
        self._done = False
        self._values = []

    def __iter__(self):
        if self._done:
            return iter(self._values)
        return itertools.chain(self._values, self._gen_iter())

    def _gen_iter(self):
        for new_value in self._iter:
            self._values.append(new_value)
            yield new_value
        self._done = True

    def __len__(self):
        return len(self._iterable)


class MapMemoryCache(torch.utils.data.Dataset):
    """
    Wrap a dataset so that, whenever a new item is returned, it is saved to memory.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n] is not None:
            return self._cache[n]

        item = self.dataset[n]
        self._cache[n] = item

        return item

    def __len__(self):
        return len(self.dataset)


class ProcessedLIBRISPEECH(LIBRISPEECH):
    def __init__(self, transforms, encode, *args, **kwargs):
        self.transforms = transforms
        self.encode = encode
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        return self.process_datapoint(item)

    def __next__(self):
        item = super().__next__()
        return self.process_datapoint(item)

    def process_datapoint(self, item):
        transformed = item[0]  # .to(device, non_blocking=non_blocking)
        target = item[2].lower()

        transformed = self.transforms(transformed)
        transformed = transformed[0, ...].transpose(0, -1)

        target = " " + target + " "
        target = self.encode(target)
        target = torch.tensor(target, dtype=torch.long, device=transformed.device)

        transformed = transformed  # .to("cpu")
        target = target  # .to("cpu")
        return transformed, target


def datasets_librispeech(
    transforms,
    language_model,
    root="/datasets01/",
    folder_in_archive="librispeech/062419/",
):
    def create(tag):

        if isinstance(tag, str):
            tag = [tag]

        data = torch.utils.data.ConcatDataset(
            [
                ProcessedLIBRISPEECH(
                    transforms,
                    language_model.encode,
                    root,
                    t,
                    folder_in_archive=folder_in_archive,
                    download=False,
                )
                for t in tag
            ]
        )

        # data = diskcache_iterator(data)
        data = MapMemoryCache(data)
        return data

    return create("train-clean-100"), create("dev-clean"), None
    # return create(["train-clean-100", "train-clean-360", "train-other-500"]), create(["dev-clean", "dev-other"]), None


def greedy_decode(outputs):
    """Greedy Decoder. Returns highest probability of class labels for each timestep

    Args:
        outputs (torch.Tensor): shape (input length, batch size, number of classes (including blank))

    Returns:
        torch.Tensor: class labels per time step.
    """
    _, indices = topk(outputs, k=1, dim=-1)
    return indices[..., 0]


def levenshtein_distance(r: str, h: str, device: Optional[str] = None):

    # initialisation
    d = torch.zeros((2, len(h) + 1), dtype=torch.long)  # , device=device)
    dold = 0
    dnew = 1

    # computation
    for i in range(1, len(r) + 1):
        d[dnew, 0] = 0
        for j in range(1, len(h) + 1):

            if r[i - 1] == h[j - 1]:
                d[dnew, j] = d[dnew - 1, j - 1]
            else:
                substitution = d[dnew - 1, j - 1] + 1
                insertion = d[dnew, j - 1] + 1
                deletion = d[dnew - 1, j] + 1
                d[dnew, j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return d[dnew, -1].item()


def collate_fn(batch):

    tensors = [b[0] for b in batch if b]

    tensors_lengths = torch.tensor(
        [model_length_function(t) for t in tensors],
        dtype=torch.long,
        device=tensors[0].device,
    )

    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors = tensors.transpose(1, -1)

    targets = [b[1] for b in batch if b]
    target_lengths = torch.tensor(
        [target.shape[0] for target in targets], dtype=torch.long, device=tensors.device
    )
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return tensors, targets, tensors_lengths, target_lengths


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model,
    criterion,
    optimizer,
    scheduler,
    data_loader,
    device,
    pbar=None,
    non_blocking=False,
):

    model.train()

    sums = defaultdict(lambda: 0.0)
    for inputs, targets, tensors_lengths, target_lengths in bg_iterator(
        data_loader, maxsize=2
    ):

        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        # keep batch first for data parallel
        outputs = model(inputs).transpose(-1, -2).transpose(0, 1)

        # CTC
        # outputs: input length, batch size, number of classes (including blank)
        # targets: batch size, max target length
        # input_lengths: batch size
        # target_lengths: batch size

        loss = criterion(outputs, targets, tensors_lengths, target_lengths)
        sums["loss"] += loss.item()

        optimizer.zero_grad()
        loss.backward()

        if args.clip_norm > 0:
            sums["gradient"] += torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_norm
            )

        optimizer.step()

        if SIGNAL_RECEIVED:
            return

        if pbar is not None:
            pbar.update(1 / len(data_loader))

    # Average
    for k in sums.keys():
        sums[k] /= len(data_loader)

    print(f"Training loss: {sums['loss']:4.5f}", flush=True)
    if "gradient" in sums:
        print(f"Average gradient norm: {sums['gradient']:4.5f}", flush=True)

    scheduler.step()


def evaluate(
    model, criterion, data_loader, decoder, language_model, device, non_blocking=False
):

    with torch.no_grad():

        model.eval()

        sums = defaultdict(lambda: 0.0)

        for inputs, targets, tensors_lengths, target_lengths in bg_iterator(
            data_loader, maxsize=2
        ):

            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            # keep batch first for data parallel
            outputs = model(inputs).transpose(-1, -2).transpose(0, 1)

            # CTC
            # outputs: input length, batch size, number of classes (including blank)
            # targets: batch size, max target length
            # input_lengths: batch size
            # target_lengths: batch size

            sums["loss"] += criterion(
                outputs, targets, tensors_lengths, target_lengths
            ).item()

            output = outputs.transpose(0, 1).to("cpu")
            output = decoder(output)

            output = language_model.decode(output.tolist())
            target = language_model.decode(targets.tolist())

            print_length = 20
            for i in range(2):
                output_print = output[i].ljust(print_length)[:print_length]
                target_print = target[i].ljust(print_length)[:print_length]
                print(f"Target: {target_print}   Output: {output_print}", flush=True)

            cers = [levenshtein_distance(a, b) for a, b in zip(target, output)]
            # cers_normalized = [d / len(a) for a, d in zip(target, cers)]
            cers = sum(cers)
            sums["cer"] += cers

            output = [o.split(language_model.char_space) for o in output]
            target = [o.split(language_model.char_space) for o in target]

            wers = [levenshtein_distance(a, b) for a, b in zip(target, output)]
            # wers_normalized = [d / len(a) for a, d in zip(target, wers)]
            wers = sum(wers)
            sums["wer"] += wers

            if SIGNAL_RECEIVED:
                break

        # Average
        for k in sums.keys():
            sums[k] /= len(data_loader)

        print(f"Validation loss: {sums['loss']:.5f}", flush=True)
        print(f"CER: {sums['cer']}  WER: {sums['wer']}", flush=True)

        return sums["loss"]


def main(args):

    print("Start time: {}".format(str(datetime.now())), flush=True)

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Install signal handler
    signal.signal(signal.SIGUSR1, lambda a, b: signal_handler(a, b))

    # Change backend
    torchaudio.set_audio_backend("soundfile")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # num_devices = torch.cuda.device_count()

    loader_training_params = {
        "num_workers": args.workers,
        "pin_memory": True,
        "shuffle": True,
        "drop_last": True,
    }
    loader_validation_params = loader_training_params.copy()
    loader_validation_params["shuffle"] = False

    non_blocking = True

    # audio

    melkwargs = {
        "n_fft": 512,
        "n_mels": args.n_bins,  # 13, 20, 128
        "hop_length": 80,  # (160, 80)
    }

    sample_rate_original = 16000

    transforms = nn.Sequential(
        # torchaudio.transforms.Resample(sample_rate_original, sample_rate_original//2),
        # torchaudio.transforms.MFCC(sample_rate=sample_rate_original, n_mfcc=args.n_bins, melkwargs=melkwargs),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate_original, **melkwargs
        ),
        # torchaudio.transforms.FrequencyMasking(freq_mask_param=args.n_bins),
        # torchaudio.transforms.TimeMasking(time_mask_param=35)
    )

    # Text preprocessing

    char_blank = "*"
    char_space = " "
    char_apostrophe = "'"
    labels = char_blank + char_space + char_apostrophe + string.ascii_lowercase
    language_model = LanguageModel(labels, char_blank, char_space)

    training, validation, _ = datasets_librispeech(transforms, language_model)

    model = Wav2Letter(
        num_classes=language_model.length, input_type="mfcc", num_features=args.n_bins
    )

    if args.jit:
        model = torch.jit.script(model)

    if not args.distributed:
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    model = model.to(device, non_blocking=non_blocking)

    n = count_parameters(model)
    print(f"Number of parameters: {n}", flush=True)

    # Optimizer

    optimizer_params = {
        "lr": args.learning_rate,
        # "eps": args.eps,
        # "rho": args.rho,
        "weight_decay": args.weight_decay,
    }

    optimizer = SGD(model.parameters(), **optimizer_params)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    # scheduler = ReduceLROnPlateau(optimizer, patience=2, threshold=1e-3)

    criterion = torch.nn.CTCLoss(
        blank=language_model.mapping[char_blank], zero_infinity=False
    )
    # criterion = nn.MSELoss()
    # criterion = torch.nn.NLLLoss()

    loader_training = DataLoader(
        training,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_training_params,
    )
    loader_validation = DataLoader(
        validation,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_validation_params,
    )

    best_loss = 1.0

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print("Checkpoint: loading '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        print(
            "Checkpoint: loaded '{}' at epoch {}".format(
                args.checkpoint, checkpoint["epoch"]
            )
        )
    else:
        print("Checkpoint: not found")

        save_checkpoint(
            {
                "epoch": args.start_epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            False,
            args.checkpoint,
        )

    with tqdm(total=args.epochs, unit_scale=1, disable=not args.progress_bar) as pbar:

        for epoch in range(args.start_epoch, args.epochs):

            train_one_epoch(
                model,
                criterion,
                optimizer,
                scheduler,
                loader_training,
                device,
                pbar=pbar,
                non_blocking=non_blocking,
            )

            if SIGNAL_RECEIVED:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    False,
                    args.checkpoint,
                )
            if not epoch % args.print_freq or epoch == args.epochs - 1:

                sum_loss = evaluate(
                    model,
                    criterion,
                    loader_validation,
                    greedy_decode,
                    language_model,
                    device,
                    non_blocking=non_blocking,
                )

                is_best = sum_loss < best_loss
                best_loss = min(sum_loss, best_loss)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    is_best,
                    args.checkpoint,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
