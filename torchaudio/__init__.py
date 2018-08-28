import os.path

import torch
import _torch_sox

from torchaudio import transforms, datasets, sox_effects


def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError('Expected a tensor, got %s' % type(src))
    if src.is_cuda:
        raise TypeError('Expected a CPU based tensor, got %s' % type(src))


def load(filepath,
         out=None,
         normalization=True,
         channels_first=True,
         num_frames=-1,
         offset=0,
         signalinfo=None,
         encodinginfo=None,
         filetype=None):
    """Loads an audio file from disk into a Tensor

    Args:
        filepath (string): path to audio file
        out (Tensor, optional): an output Tensor to use instead of creating one
        normalization (bool, number, or function, optional): If boolean `True`, then output is divided by `1 << 31`
                                                             (assumes 16-bit depth audio, and normalizes to `[0, 1]`.
                                                             If `number`, then output is divided by that number
                                                             If `function`, then the output is passed as a parameter
                                                             to the given function, then the output is divided by
                                                             the result.
        num_frames (int, optional): number of frames to load.  -1 to load everything after the offset.
        offset (int, optional): number of frames from the start of the file to begin data loading.
        signalinfo (sox_signalinfo_t, optional): a sox_signalinfo_t type, which could be helpful if the
                                                 audio type cannot be automatically determine
        encodinginfo (sox_encodinginfo_t, optional): a sox_encodinginfo_t type, which could be set if the
                                                     audio type cannot be automatically determined
        filetype (str, optional): a filetype or extension to be set if sox cannot determine it automatically

    Returns: tuple(Tensor, int)
       - Tensor: output Tensor of size `[L x C]` where L is the number of audio frames, C is the number of channels
       - int: the sample-rate of the audio (as listed in the metadata of the file)

    Example::

        >>> data, sample_rate = torchaudio.load('foo.mp3')
        >>> print(data.size())
        torch.Size([278756, 2])
        >>> print(sample_rate)
        44100
        >>> data_volume_normalized, _ = torchaudio.load('foo.mp3', normalization=lambda x: torch.abs(x).max())
        >>> print(data_volume_normalized.abs().max())
        1.

    """
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    # initialize output tensor
    if out is not None:
        check_input(out)
    else:
        out = torch.FloatTensor()

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if offset < 0:
        raise ValueError("Expected positive offset value")

    sample_rate = _torch_sox.read_audio_file(filepath,
                                             out,
                                             channels_first,
                                             num_frames,
                                             offset,
                                             signalinfo,
                                             encodinginfo,
                                             filetype)

    # normalize if needed
    _audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath, src, sample_rate, precision=16, channels_first=True):
    si = sox_signalinfo_t()
    ch_idx = 0 if channels_first else 1
    si.rate = sample_rate
    si.channels = 1 if src.dim() == 1 else src.size(ch_idx)
    si.length = src.numel()
    si.precision = precision
    return save_encinfo(filepath, src, channels_first, si)


def save_encinfo(filepath, src, channels_first=True, signalinfo=None, encodinginfo=None, filetype=None):
    """Saves a Tensor with audio signal to disk as a standard format like mp3, wav, etc.

    Args:
        filepath (string): path to audio file
        src (Tensor): an input 2D Tensor of shape `[L x C]` where L is
                      the number of audio frames, C is the number of channels
        signalinfo (sox_signalinfo_t): a sox_signalinfo_t type, which could be helpful if the
                                       audio type cannot be automatically determine
        encodinginfo (sox_encodinginfo_t, optional): a sox_encodinginfo_t type, which could be set if the
                                                     audio type cannot be automatically determined
        filetype (str, optional): a filetype or extension to be set if sox cannot determine it automatically

    Example::

        >>> data, sample_rate = torchaudio.load('foo.mp3')
        >>> torchaudio.save('foo.wav', data, sample_rate)

    """
    ch_idx = 0 if channels_first else 1
    len_idx = 1 if channels_first else 0

    # check if save directory exists
    abs_dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(abs_dirpath):
        raise OSError("Directory does not exist: {}".format(abs_dirpath))
    # check that src is a CPU tensor
    check_input(src)
    # Check/Fix shape of source data
    if src.dim() == 1:
        # 1d tensors as assumed to be mono signals
        src.unsqueeze_(ch_idx)
    elif src.dim() > 2 or src.size(ch_idx) > src.size(len_idx):
        # assumes num_samples > num_channels
        raise ValueError(
            "Expected format (L x C), C < L, but found {}".format(src.size()))
    # sox stores the sample rate as a float, though practically sample rates are almost always integers
    # convert integers to floats
    if not isinstance(signalinfo.rate, float):
        if float(signalinfo.rate) == signalinfo.rate:
            signalinfo.rate = float(signalinfo.rate)
        else:
            raise TypeError('Sample rate should be a float or int')
    # check if the bit precision (i.e. bits per sample) is an integer
    if not isinstance(signalinfo.precision, int):
        if int(signalinfo.precision) == signalinfo.precision:
            signalinfo.precision = int(signalinfo.precision)
        else:
            raise TypeError('Bit precision should be an integer')
    # programs such as librosa normalize the signal, unnormalize if detected
    if src.min() >= -1.0 and src.max() <= 1.0:
        src = src * (1 << 31)
        src = src.long()
    # set filetype and allow for files with no extensions
    extension = os.path.splitext(filepath)[1]
    filetype = extension[1:] if len(extension) > 0 else filetype
    # transpose from C x L -> L x C
    if channels_first:
        src = src.transpose(1, 0)
    # save data to file
    src = src.contiguous()
    _torch_sox.write_audio_file(filepath, src, signalinfo, encodinginfo, filetype)


def info(filepath):
    """Gets metadata from an audio file without loading the signal.

     Args:
        filepath (string): path to audio file

     Returns: tuple(si, ei)
       - si (sox_signalinfo_t): signal info as a python object
       - ei (sox_encodinginfo_t): encoding info as a python object

     Example::
         >>> si, ei = torchaudio.info('foo.wav')
         >>> rate, channels, encoding = si.rate, si.channels, ei.encoding
     """
    return _torch_sox.get_info(filepath)


def effect_names():
    """Gets list of valid sox effect names

    Returns: list[str]

    Example::
        >>> EFFECT_NAMES = torchaudio.effect_names()
    """
    return _torch_sox.get_effect_names()


def SoxEffect():
    """Create a object to hold sox effect and options to pass between python and c++

    Returns: SoxEffects(object)
      - ename (str), name of effect
      - eopts (list[str]), list of effect options
    """
    return _torch_sox.SoxEffect()


def sox_signalinfo_t():
    """Create a sox_signalinfo_t object.  This object can be used to set the sample
       rate, number of channels, length, bit precision and headroom multiplier
       primarily for effects

    Returns: sox_signalinfo_t(object)
      - rate (float), sample rate as a float, practically will likely be an integer float
      - channel (int), number of audio channels
      - precision (int), bit precision
      - length (int), length of audio, 0 for unspecified and -1 for unknown
      - mult (float, optional), headroom multiplier for effects and None for no multiplier
    """
    return _torch_sox.sox_signalinfo_t()


def sox_encodinginfo_t():
    """Create a sox_encodinginfo_t object.  This object can be used to set the encoding
       type, bit precision, compression factor, reverse bytes, reverse nibbles,
       reverse bits and endianness.  This can be used in an effects chain to encode the
       final output or to save a file with a specific encoding.  For example, one could
       use the sox ulaw encoding to do 8-bit ulaw encoding.  Note in a tensor output
       the result will be a 32-bit number, but number of unique values will be determined by
       the bit precision.

    Returns: sox_encodinginfo_t(object)
      - encoding (sox_encoding_t), output encoding
      - bits_per_sample (int), bit precision, same as `precision` in sox_signalinfo_t
      - compression (float), compression for lossy formats, 0.0 for default compression
      - reverse_bytes (sox_option_t), reverse bytes, use sox_option_default
      - reverse_nibbles (sox_option_t), reverse nibbles, use sox_option_default
      - reverse_bits (sox_option_t), reverse bytes, use sox_option_default
      - opposite_endian (sox_bool), change endianness, use sox_false
    """
    ei = _torch_sox.sox_encodinginfo_t()
    sdo = get_sox_option_t(2)  # sox_default_option
    ei.reverse_bytes = sdo
    ei.reverse_nibbles = sdo
    ei.reverse_bits = sdo
    return ei


def get_sox_encoding_t(i=None):
    """Get enum of sox_encoding_t for sox encodings.

    Args:
        i (int, optional): choose type or get a dict with all possible options
                           use .__members__ to see all options when not specified
    Returns:
        sox_encoding_t: a sox_encoding_t type for output encoding
    """
    if i is None:
        # one can see all possible values using the .__members__ attribute
        return _torch_sox.sox_encoding_t
    else:
        return _torch_sox.sox_encoding_t(i)


def get_sox_option_t(i=2):
    """Get enum of sox_option_t for sox encodinginfo options.

    Args:
        i (int, optional): choose type or get a dict with all possible options
                           use .__members__ to see all options when not specified.
                           Defaults to sox_option_default.
    Returns:
        sox_option_t: a sox_option_t type
    """
    if i is None:
        return _torch_sox.sox_option_t
    else:
        return _torch_sox.sox_option_t(i)


def get_sox_bool(i=0):
    """Get enum of sox_bool for sox encodinginfo options.

    Args:
        i (int, optional): choose type or get a dict with all possible options
                           use .__members__ to see all options when not specified.
                           Defaults to sox_false.
    Returns:
        sox_bool: a sox_bool type
    """
    if i is None:
        return _torch_sox.sox_bool
    else:
        return _torch_sox.sox_bool(i)


def initialize_sox():
    """Initialize sox for effects chain.  Not required for simple loading.  Importantly,
       only initialize this once and do not shutdown until you have done effect chain
       calls even when loading multiple files.
    """
    return _torch_sox.initialize_sox()


def shutdown_sox():
    """Showdown sox for effects chain.  Not required for simple loading.  Importantly,
       only call once.  Attempting to re-initialize sox will result seg faults.
    """
    return _torch_sox.shutdown_sox()


def _audio_normalization(signal, normalization):
    # assumes signed 32-bit depth, which is what sox uses internally

    if not normalization:
        return

    if isinstance(normalization, bool):
        normalization = 1 << 31

    if isinstance(normalization, (float, int)):
        # normalize with custom value
        a = normalization
        signal /= a
    elif callable(normalization):
        a = normalization(signal)
        signal /= a
