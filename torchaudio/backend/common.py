class AudioMetaData:
    """Return type of ``torchaudio.info`` function.

    This class is used by :ref:`"sox_io" backend<sox_io_backend>` and
    :ref:`"soundfile" backend with the new interface<soundfile_backend>`.

    :ivar int sample_rate: Sample rate
    :ivar int num_frames: The number of frames
    :ivar int num_channels: The number of channels
    :ivar int bits_per_sample: The number of bits per sample. This is 0 for lossy formats,
        or when it cannot be accurately inferred.
    :ivar str encoding: Audio encoding.
    """
    def __init__(
            self,
            sample_rate: int,
            num_frames: int,
            num_channels: int,
            bits_per_sample: int,
            encoding: str,
    ):
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.bits_per_sample = bits_per_sample
        self.encoding = encoding
