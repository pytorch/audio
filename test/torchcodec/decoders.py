import torchaudio_unittest.common_utils.wav_utils as wav_utils
from types import SimpleNamespace

# See corresponding [TorchCodec test dependency mocking hack] note in
# conftest.py

class AudioDecoder:
    def __init__(self, uri):
        self.uri = uri
        data, sample_rate = wav_utils.load_wav(self.uri)
        self.metadata = SimpleNamespace(sample_rate=sample_rate)
        self.data = data

    def get_all_samples(self):
        return SimpleNamespace(data=self.data)
