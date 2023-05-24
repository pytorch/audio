"""Run smoke tests"""
import argparse
import logging


def base_smoke_test():
    import torchaudio  # noqa: F401
    import torchaudio.compliance.kaldi  # noqa: F401
    import torchaudio.datasets  # noqa: F401
    import torchaudio.functional  # noqa: F401
    import torchaudio.models  # noqa: F401
    import torchaudio.pipelines  # noqa: F401
    import torchaudio.sox_effects  # noqa: F401
    import torchaudio.transforms  # noqa: F401
    import torchaudio.utils  # noqa: F401


def ffmpeg_test():
    from torchaudio.io import StreamReader  # noqa: F401


def main() -> None:
    options = _parse_args()

    if options.debug:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    base_smoke_test()
    if options.ffmpeg:
        ffmpeg_test()
    print("Smoke test passed.")


def _parse_args():
    parser = argparse.ArgumentParser()

    # Warning: Please note this option should not be widely used, only use it when absolutely necessary
    parser.add_argument("--no-ffmpeg", dest="ffmpeg", action="store_false")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
