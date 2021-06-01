#!/usr/bin/env python3
"""Parse a directory contains Librispeech dataset.

Recursively search for "*.trans.txt" file in the given directory and print out

`<ID>\\t<AUDIO_PATH>\\t<TRANSCRIPTION>`

example: python parse_librispeech.py LibriSpeech/test-clean

    1089-134691-0000\t/LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac\tHE COULD WAIT NO LONGER
    ...

Dataset can be obtained from https://www.openslr.org/12
"""
import argparse
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory where `*.trans.txt` files are searched.'
    )
    return parser.parse_args()


def _parse_transcript(path):
    with open(path) as trans_fileobj:
        for line in trans_fileobj:
            line = line.strip()
            if line:
                yield line.split(' ', maxsplit=1)


def _parse_directory(root_dir: Path):
    for trans_file in root_dir.glob('**/*.trans.txt'):
        trans_dir = trans_file.parent
        for id_, transcription in _parse_transcript(trans_file):
            audio_path = trans_dir / f'{id_}.flac'
            yield id_, audio_path, transcription


def _main():
    args = _parse_args()
    for id_, path, transcription in _parse_directory(args.input_dir):
        print(f'{id_}\t{path}\t{transcription}')


if __name__ == '__main__':
    _main()
