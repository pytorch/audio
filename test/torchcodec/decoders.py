import test.torchaudio_unittest.common_utils.wav_utils as wav_utils

class AudioDecoder:
    def __init__(self, uri):
        self.uri = uri

    def get_all_samples(self):
        return wav_utils.load_wav(self.uri)


class AudioEncoder:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate

    def to_file(self, uri, bit_rate=None):
        return wav_utils.save_wav(uri, self.data, self.sample_rate)
