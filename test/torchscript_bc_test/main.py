#!/usr/bin/env python3
"""Generate torchscript object of specific torhcaudio version.

This requires that the corresponding torchaudio (and torch) is installed.
"""
import os
import sys
import argparse


_BASE_OBJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["generate", "validate"],
        required=True,
        help=(
            '"generate" generates Torchscript objects of the specific torchaudio '
            "in the given directory. "
            '"validate" validates if the objects in the givcen directory are compatible '
            "with the current torhcaudio."
        ),
    )
    parser.add_argument(
        "--version", choices=["0.6.0"], required=True, help="torchaudio version."
    )
    parser.add_argument(
        "--base-obj-dir",
        default=_BASE_OBJ_DIR,
        help="Directory where objects are saved/loaded.",
    )
    return parser.parse_args()


def _generate(version, output_dir):
    if version == "0.6.0":
        import ver_060

        ver_060.generate(output_dir)
    else:
        raise ValueError(f"Unexpected torchaudio version: {version}")


def _validate(version, input_dir):
    if version == "0.6.0":
        import ver_060

        ver_060.validate(input_dir)
    else:
        raise ValueError(f"Unexpected torchaudio version: {version}")


def _get_obj_dir(base_dir, version):
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(base_dir, f"{version}-py{py_version}")


def _main():
    args = _parse_args()
    obj_dir = _get_obj_dir(args.base_obj_dir, args.version)
    if args.mode == "generate":
        _generate(args.version, obj_dir)
    elif args.mode == "validate":
        _validate(args.version, obj_dir)
    else:
        raise ValueError(f"Unexpected mode: {args.mode}")


if __name__ == "__main__":
    _main()
