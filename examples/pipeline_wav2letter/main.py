import argparse
import logging

from engine import spawn_main


def parse_args():
    parser = argparse.ArgumentParser(description="Train wav2letter for End-to-End ASR.")

    parser.add_argument(
        "--model-input-type",
        default="mfcc",
        choices=["waveform", "mfcc"],
        help="input type for model",
    )
    parser.add_argument(
        "--hidden-channels",
        default=2000,
        type=int,
        help="number of hidden channels in wav2letter",
    )
    parser.add_argument(
        "--freq-mask", default=0, type=int, help="maximal width of frequency mask",
    )
    parser.add_argument(
        "--time-mask", default=0, type=int, help="maximal width of time mask",
    )
    parser.add_argument(
        "--win-length", default=400, type=int, help="width of spectrogram window",
    )
    parser.add_argument(
        "--hop-length", default=160, type=int, help="hop length in for spectrogram",
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="number of data loading workers",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="path to output file, standard output if not specified",
    )
    parser.add_argument(
        "--checkpoint", type=str, metavar="PATH", help="path to checkpoint file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from checkpoint file provided by --checkpoint option",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, help="manual epoch number"
    )
    parser.add_argument(
        "--print-freq", default=10, type=int, help="print frequency in epochs",
    )
    parser.add_argument(
        "--reduce-lr-valid",
        action="store_true",
        help="reduce learning rate based on validation loss",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="normalize model input"
    )
    parser.add_argument(
        "--progress-bar", action="store_true", help="use progress bar while training"
    )
    parser.add_argument(
        "--decoder",
        default="greedy",
        choices=["greedy", "greedyiter", "viterbi"],
        help="ctc decoder to use",
    )
    parser.add_argument("--batch-size", default=128, type=int, help="mini-batch size")
    parser.add_argument(
        "--bins", default=13, type=int, help="number of bins in transforms",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="probability of given weights to be zeroed",
    )
    parser.add_argument(
        "--optimizer",
        metavar="OPT",
        default="adadelta",
        choices=["sgd", "adadelta", "adam", "adamw"],
        help="optimizer to use",
    )
    parser.add_argument(
        "--scheduler",
        default="reduceonplateau",
        choices=["exponential", "reduceonplateau"],
        help="optimizer to use",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.6,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        metavar="GAMMA",
        help="learning rate exponential decay constant",
    )
    parser.add_argument(
        "--momentum", default=0.8, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay", default=1e-5, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--eps",
        metavar="EPS",
        type=float,
        default=1e-8,
        help="epsilon parameter for Adadelta",
    )
    parser.add_argument(
        "--rho",
        metavar="RHO",
        type=float,
        default=0.95,
        help="rho parameter for Adadelta",
    )
    parser.add_argument(
        "--clip-grad", metavar="NORM", type=float, help="value to clip gradient at",
    )
    parser.add_argument(
        "--dataset-root", type=str, help="specify dataset root folder",
    )
    parser.add_argument(
        "--dataset-folder-in-archive",
        type=str,
        help="specify dataset folder in archive",
    )
    parser.add_argument(
        "--dataset-train",
        default=["train-clean-100"],
        nargs="+",
        type=str,
        help="select which part of librispeech to train with",
    )
    parser.add_argument(
        "--dataset-valid",
        default=["dev-clean"],
        nargs="+",
        type=str,
        help="select which part of librispeech to validate with",
    )
    parser.add_argument(
        "--distributed", action="store_true", help="enable DistributedDataParallel"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--world-size", type=int, default=8, help="the world size to initiate DPP"
    )
    parser.add_argument(
        "--speechcommands",
        action="store_true",
        help="select speechcommands instead of librispeech",
    )

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    spawn_main(args)
