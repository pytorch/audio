from torchaudio_unittest.common_utils import sox_utils


def get_encoding(ext, dtype):
    exts = {
        'mp3',
        'flac',
        'vorbis',
    }
    encodings = {
        'float32': 'PCM_F',
        'int32': 'PCM_S',
        'int16': 'PCM_S',
        'uint8': 'PCM_U',
    }
    return ext.upper() if ext in exts else encodings[dtype]


def get_bits_per_sample(ext, dtype):
    bits_per_samples = {
        'flac': 24,
        'mp3': 0,
        'vorbis': 0,
    }
    return bits_per_samples.get(ext, sox_utils.get_bit_depth(dtype))
