#!/usr/bin/env python
"""Parse a directory contains VoxForge dataset.

Recursively search for "PROMPTS" file in the given directory and print out

`<ID>\\t<AUDIO_PATH>\\t<TRANSCRIPTION>`

example: python parse_voxforge.py voxforge/de/Helge-20150608-aku

    de5-001\t/datasets/voxforge/de/guenter-20140214-afn/wav/de5-001.wav\tES SOLL ETWA FÜNFZIGTAUSEND VERSCHIEDENE SORTEN GEBEN
    ...

Dataset can be obtained from http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/
"""
import os
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


def _parse_prompts(path):
    base_dir = path.parent.parent
    with open(path) as trans_fileobj:
        for line in trans_fileobj:
            line = line.strip()
            if not line:
                continue

            id_, transcript = line.split(' ', maxsplit=1)
            if not transcript:
                continue

            transcript = transcript.upper()
            filename = id_.split('/')[-1]
            audio_path = base_dir / 'wav' / f'{filename}.wav'
            if os.path.exists(audio_path):
                yield id_, audio_path, transcript


def _parse_directory(root_dir: Path):
    for prompt_file in root_dir.glob('**/PROMPTS'):
        try:
            yield from _parse_prompts(prompt_file)
        except UnicodeDecodeError:
            pass


def _main():
    args = _parse_args()
    for id_, path, transcription in _parse_directory(args.input_dir):
        print(f'{id_}\t{path}\t{transcription}')


if __name__ == '__main__':
    _main()
