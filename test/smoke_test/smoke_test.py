#!/usr/bin/env python3
"""Run smoke tests"""
import argparse
import logging


def base_smoke_test():
    import torchaudio  # noqa: F401
    import torchaudio.datasets  # noqa: F401
    import torchaudio.functional  # noqa: F401
    import torchaudio.models  # noqa: F401
    import torchaudio.transforms  # noqa: F401
    import torchaudio.utils  # noqa: F401

def _run_smoke_test():
    base_smoke_test()
    print("Smoke test passed.")


def main(args=None) -> None:
    options = _parse_args(args)

    if options.debug:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    _chdir()
    _run_smoke_test()


def _parse_args(args):
    parser = argparse.ArgumentParser()

    # Warning: Please note this option should not be widely used, only use it when absolutely necessary
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    return parser.parse_args(args)


def _chdir():
    # smoke test should not be performed on the root directory of checked out source code.
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    assert "torchaudio" not in os.listdir(os.getcwd())


if __name__ == "__main__":
    main()
