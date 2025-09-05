import torchaudio_unittest.common_utils.wav_utils as wav_utils
from types import SimpleNamespace

# See corresponding [TorchCodec test dependency mocking hack] note in
# conftest.py

class AudioEncoder:
    def __init__(self, data, sample_rate):
        self.data = data
        self.metadata = SimpleNamespace(sample_rate=sample_rate)

    def to_file(self, uri, bit_rate=None):
        return wav_utils.save_wav(uri, self.data, self.metadata.sample_rate)
