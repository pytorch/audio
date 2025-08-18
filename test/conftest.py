import sys
from pathlib import Path

# Note: [TorchCodec test dependency mocking hack]
# We are adding the `test/` directory to the system path. This causes the
# `tests/torchcodec` folder to be importable, and in particular, this makes it
# possible to mock torchcodec utilities. E.g. executing:
#
# ```
# from torchcodec.decoders import AudioDecoder
# ```
# directly or indirectly when running the tests will effectively be loading the
# mocked `AudioDecoder` implemented in `test/torchcodec/decoders.py`, which
# relies on scipy instead of relying on torchcodec.
#
# So whenever `torchaudio.load()` is called from within the tests, it's the
# mocked scipy `AudioDecoder` that gets used.  Ultimately, this allows us *not*
# to add torchcodec as a test dependency of torchaudio: we can just rely on
# scipy.
#
# This is VERY hacky and ideally we should implement a more robust way to mock
# torchcodec.
sys.path.append(str(Path(__file__).parent.resolve()))
