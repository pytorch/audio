import torchaudio_unittest.common_utils.wav_utils as wav_utils

class AudioEncoder:
    def __init__(self, data, sample_rate):
        print("BEING CALLED")
        self.data = data
        self.sample_rate = sample_rate

    def to_file(self, uri, bit_rate=None):
        return wav_utils.save_wav(uri, self.data, self.sample_rate)
