"""Generate opus file for testing load functions"""

import argparse
import subprocess

import scipy.io.wavfile
import torch


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate opus files for test")
    parser.add_argument("--num-channels", required=True, type=int)
    parser.add_argument(
        "--compression-level", required=True, type=int, choices=list(range(11))
    )
    parser.add_argument("--bitrate", default="96k")
    return parser.parse_args()


def convert_to_opus(src_path, dst_path, *, bitrate, compression_level):
    """Convert audio file with `ffmpeg` command."""
    command = ["ffmpeg", "-y", "-i", src_path, "-c:a", "libopus", "-b:a", bitrate]
    if compression_level is not None:
        command += ["-compression_level", str(compression_level)]
    command += [dst_path]
    print(" ".join(command))
    subprocess.run(command, check=True)


def _generate(num_channels, compression_level, bitrate):
    org_path = "original.wav"
    ops_path = f"{bitrate}_{compression_level}_{num_channels}ch.opus"

    # Note: ffmpeg forces sample rate 48k Hz for opus https://stackoverflow.com/a/39186779
    # 1. generate original wav
    data = (
        torch.linspace(-32768, 32767, 32768, dtype=torch.int16)
        .repeat([num_channels, 1])
        .t()
    )
    scipy.io.wavfile.write(org_path, 48000, data.numpy())
    # 2. convert to opus
    convert_to_opus(
        org_path, ops_path, bitrate=bitrate, compression_level=compression_level
    )


def _main():
    args = _parse_args()
    _generate(args.num_channels, args.compression_level, args.bitrate)


if __name__ == "__main__":
    _main()
