"""Run smoke tests"""

import torchaudio  # noqa: F401
import torchaudio.compliance.kaldi  # noqa: F401
import torchaudio.datasets  # noqa: F401
import torchaudio.functional  # noqa: F401
import torchaudio.models  # noqa: F401
import torchaudio.pipelines  # noqa: F401
import torchaudio.sox_effects  # noqa: F401
import torchaudio.transforms  # noqa: F401
import torchaudio.utils  # noqa: F401


def streamreader_test():
    from torchaudio.io import StreamReader  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streamreader", action="store_true")
    parser.add_argument("--no-streamreader", dest="streamreader", action="store_false")
    parser.set_defaults(streamreader=True)

    options = parser.parse_args()
    if options.streamreader:
        streamreader_test()


if __name__ == "__main__":
    main()
