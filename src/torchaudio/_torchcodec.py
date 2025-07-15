"""TorchCodec integration for TorchAudio."""

import os
from typing import BinaryIO, Optional, Tuple, Union

import torch


def load_with_torchcodec(
    uri: Union[BinaryIO, str, os.PathLike],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio data from source using TorchCodec's AudioDecoder.
    
    .. note::
        
        This function supports the same API as ``torchaudio.load()``, and relies
        on TorchCodec's decoding capabilities under the hood. It is provided for
        convenience, but we do recommend that you port your code to natively use
        ``torchcodec``'s ``AudioDecoder`` class for better performance:
        https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.
        In TorchAudio 2.9, ``torchaudio.load()`` will be relying on
        ``load_with_torchcodec``. Note that some parameters of
        ``torchaudio.load()``, like ``normalize``, ``buffer_size``, and
        ``backend``, are ignored by ``load_with_torchcodec``.
    
    
    Args:
        uri (path-like object or file-like object):
            Source of audio data. The following types are accepted:
            
            * ``path-like``: File path or URL.
            * ``file-like``: Object with ``read(size: int) -> bytes`` method.
            
        frame_offset (int, optional):
            Number of samples to skip before start reading data.
        num_frames (int, optional):
            Maximum number of samples to read. ``-1`` reads all the remaining samples,
            starting from ``frame_offset``.
        normalize (bool, optional):
            TorchCodec always returns normalized float32 samples. This parameter
            is ignored and a warning is issued if set to False.
            Default: ``True``.
        channels_first (bool, optional):
            When True, the returned Tensor has dimension `[channel, time]`.
            Otherwise, the returned Tensor's dimension is `[time, channel]`.
        format (str or None, optional):
            Format hint for the decoder. May not be supported by all TorchCodec
            decoders. (Default: ``None``)
        buffer_size (int, optional):
            Not used by TorchCodec AudioDecoder. Provided for API compatibility.
        backend (str or None, optional):
            Not used by TorchCodec AudioDecoder. Provided for API compatibility.
            
    Returns:
        (torch.Tensor, int): Resulting Tensor and sample rate.
        Always returns float32 tensors. If ``channels_first=True``, shape is
        `[channel, time]`, otherwise `[time, channel]`.
        
    Raises:
        ImportError: If torchcodec is not available.
        ValueError: If unsupported parameters are used.
        RuntimeError: If TorchCodec fails to decode the audio.
        
    Note:
        - TorchCodec always returns normalized float32 samples, so the ``normalize``
          parameter has no effect.
        - The ``buffer_size`` and ``backend`` parameters are ignored.
        - Not all audio formats supported by torchaudio backends may be supported
          by TorchCodec.
    """
    # Import torchcodec here to provide clear error if not available
    try:
        from torchcodec.decoders import AudioDecoder
    except ImportError as e:
        raise ImportError(
            "TorchCodec is required for load_with_torchcodec. "
            "Please install torchcodec to use this function."
        ) from e
    
    # Parameter validation and warnings
    if not normalize:
        import warnings
        warnings.warn(
            "TorchCodec AudioDecoder always returns normalized float32 samples. "
            "The 'normalize=False' parameter is ignored.",
            UserWarning,
            stacklevel=2
        )
    
    if buffer_size != 4096:
        import warnings
        warnings.warn(
            "The 'buffer_size' parameter is not used by TorchCodec AudioDecoder.",
            UserWarning,
            stacklevel=2
        )
        
    if backend is not None:
        import warnings
        warnings.warn(
            "The 'backend' parameter is not used by TorchCodec AudioDecoder.",
            UserWarning,
            stacklevel=2
        )
    
    if format is not None:
        import warnings
        warnings.warn(
            "The 'format' parameter is not supported by TorchCodec AudioDecoder.",
            UserWarning,
            stacklevel=2
        )
    
    # Create AudioDecoder
    try:
        decoder = AudioDecoder(uri)
    except Exception as e:
        raise RuntimeError(f"Failed to create AudioDecoder for {uri}: {e}") from e
    
    # Get sample rate from metadata
    sample_rate = decoder.metadata.sample_rate
    if sample_rate is None:
        raise RuntimeError("Unable to determine sample rate from audio metadata")
    
    # Decode the entire file first, then subsample manually
    # This is the simplest approach since torchcodec uses time-based indexing
    try:
        audio_samples = decoder.get_all_samples()
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio samples: {e}") from e
        
    data = audio_samples.data
    
    # Apply frame_offset and num_frames (which are actually sample offsets)
    if frame_offset > 0:
        if frame_offset >= data.shape[1]:
            # Return empty tensor if offset is beyond available data
            empty_shape = (data.shape[0], 0) if channels_first else (0, data.shape[0])
            return torch.zeros(empty_shape, dtype=torch.float32), sample_rate
        data = data[:, frame_offset:]
    
    if num_frames == 0:
        # Return empty tensor if num_frames is 0
        empty_shape = (data.shape[0], 0) if channels_first else (0, data.shape[0])
        return torch.zeros(empty_shape, dtype=torch.float32), sample_rate
    elif num_frames > 0:
        data = data[:, :num_frames]
    
    # TorchCodec returns data in [channel, time] format by default
    # Handle channels_first parameter
    if not channels_first:
        data = data.transpose(0, 1)  # [channel, time] -> [time, channel]
    
    return data, sample_rate