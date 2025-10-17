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

        This function supports the same API as :func:`~torchaudio.load`, and
        relies on TorchCodec's decoding capabilities under the hood. It is
        provided for convenience, but we do recommend that you port your code to
        natively use ``torchcodec``'s ``AudioDecoder`` class for better
        performance:
        https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.
        As of TorchAudio 2.9, :func:`~torchaudio.load` relies on
        :func:`~torchaudio.load_with_torchcodec`. Note that some parameters of
        :func:`~torchaudio.load`, like ``normalize``, ``buffer_size``, and
        ``backend``, are ignored by :func:`~torchaudio.load_with_torchcodec`.
        To install torchcodec, follow the instructions at https://github.com/pytorch/torchcodec#installing-torchcodec.


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
            "TorchCodec is required for load_with_torchcodec. " "Please install torchcodec to use this function."
        ) from e

    # Parameter validation and warnings
    if not normalize:
        import warnings

        warnings.warn(
            "TorchCodec AudioDecoder always returns normalized float32 samples. "
            "The 'normalize=False' parameter is ignored.",
            UserWarning,
            stacklevel=2,
        )

    if buffer_size != 4096:
        import warnings

        warnings.warn("The 'buffer_size' parameter is not used by TorchCodec AudioDecoder.", UserWarning, stacklevel=2)

    if backend is not None:
        import warnings

        warnings.warn("The 'backend' parameter is not used by TorchCodec AudioDecoder.", UserWarning, stacklevel=2)

    if format is not None:
        import warnings

        warnings.warn("The 'format' parameter is not supported by TorchCodec AudioDecoder.", UserWarning, stacklevel=2)

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


def save_with_torchcodec(
    uri: Union[str, os.PathLike],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
    compression: Optional[Union[float, int]] = None,
) -> None:
    """Save audio data to file using TorchCodec's AudioEncoder.

    .. note::

        This function supports the same API as :func:`~torchaudio.save`, and
        relies on TorchCodec's encoding capabilities under the hood. It is
        provided for convenience, but we do recommend that you port your code to
        natively use ``torchcodec``'s ``AudioEncoder`` class for better
        performance:
        https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.encoders.AudioEncoder.
        As of TorchAudio 2.9, :func:`~torchaudio.save` relies on
        :func:`~torchaudio.save_with_torchcodec`. Note that some parameters of
        :func:`~torchaudio.save`, like ``format``, ``encoding``,
        ``bits_per_sample``, ``buffer_size``, and ``backend``, are ignored by
        are ignored by :func:`~torchaudio.save_with_torchcodec`.
        To install torchcodec, follow the instructions at https://github.com/pytorch/torchcodec#installing-torchcodec.

    This function provides a TorchCodec-based alternative to torchaudio.save
    with the same API. TorchCodec's AudioEncoder provides efficient encoding
    with FFmpeg under the hood.

    Args:
        uri (path-like object):
            Path to save the audio file. The file extension determines the format.

        src (torch.Tensor):
            Audio data to save. Must be a 1D or 2D tensor with float32 values
            in the range [-1, 1]. If 2D, shape should be [channel, time] when
            channels_first=True, or [time, channel] when channels_first=False.

        sample_rate (int):
            Sample rate of the audio data.

        channels_first (bool, optional):
            Indicates whether the input tensor has channels as the first dimension.
            If True, expects [channel, time]. If False, expects [time, channel].
            Default: True.

        format (str or None, optional):
            Audio format hint. Not used by TorchCodec (format is determined by
            file extension). A warning is issued if provided.
            Default: None.

        encoding (str or None, optional):
            Audio encoding. Not fully supported by TorchCodec AudioEncoder.
            A warning is issued if provided. Default: None.

        bits_per_sample (int or None, optional):
            Bits per sample. Not directly supported by TorchCodec AudioEncoder.
            A warning is issued if provided. Default: None.

        buffer_size (int, optional):
            Not used by TorchCodec AudioEncoder. Provided for API compatibility.
            A warning is issued if not default value. Default: 4096.

        backend (str or None, optional):
            Not used by TorchCodec AudioEncoder. Provided for API compatibility.
            A warning is issued if provided. Default: None.

        compression (float, int or None, optional):
            Compression level or bit rate. Maps to bit_rate parameter in
            TorchCodec AudioEncoder. Default: None.

    Raises:
        ImportError: If torchcodec is not available.
        ValueError: If input parameters are invalid.
        RuntimeError: If TorchCodec fails to encode the audio.

    Note:
        - TorchCodec AudioEncoder expects float32 samples in [-1, 1] range.
        - Some parameters (format, encoding, bits_per_sample, buffer_size, backend)
          are not used by TorchCodec but are provided for API compatibility.
        - The output format is determined by the file extension in the uri.
        - TorchCodec uses FFmpeg under the hood for encoding.
    """
    # Import torchcodec here to provide clear error if not available
    try:
        from torchcodec.encoders import AudioEncoder
    except ImportError as e:
        raise ImportError(
            "TorchCodec is required for save_with_torchcodec. " "Please install torchcodec to use this function."
        ) from e

    # Parameter validation and warnings
    if format is not None:
        import warnings

        warnings.warn(
            "The 'format' parameter is not used by TorchCodec AudioEncoder. "
            "Format is determined by the file extension.",
            UserWarning,
            stacklevel=2,
        )

    if encoding is not None:
        import warnings

        warnings.warn(
            "The 'encoding' parameter is not fully supported by TorchCodec AudioEncoder.", UserWarning, stacklevel=2
        )

    if bits_per_sample is not None:
        import warnings

        warnings.warn(
            "The 'bits_per_sample' parameter is not directly supported by TorchCodec AudioEncoder.",
            UserWarning,
            stacklevel=2,
        )

    if buffer_size != 4096:
        import warnings

        warnings.warn("The 'buffer_size' parameter is not used by TorchCodec AudioEncoder.", UserWarning, stacklevel=2)

    if backend is not None:
        import warnings

        warnings.warn("The 'backend' parameter is not used by TorchCodec AudioEncoder.", UserWarning, stacklevel=2)

    # Input validation
    if not isinstance(src, torch.Tensor):
        raise ValueError(f"Expected src to be a torch.Tensor, got {type(src)}")

    if src.dtype != torch.float32:
        src = src.float()

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    # Handle tensor shape and channels_first
    if src.ndim == 1:
        # Convert to 2D: [1, time] for channels_first=True
        if channels_first:
            data = src.unsqueeze(0)  # [1, time]
        else:
            # For channels_first=False, input is [time] -> reshape to [time, 1] -> transpose to [1, time]
            data = src.unsqueeze(1).transpose(0, 1)  # [time, 1] -> [1, time]
    elif src.ndim == 2:
        if channels_first:
            data = src  # Already [channel, time]
        else:
            data = src.transpose(0, 1)  # [time, channel] -> [channel, time]
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {src.ndim}D tensor")

    # Create AudioEncoder
    try:
        encoder = AudioEncoder(data, sample_rate=sample_rate)
    except Exception as e:
        raise RuntimeError(f"Failed to create AudioEncoder: {e}") from e

    # Determine bit_rate from compression parameter
    bit_rate = None
    if compression is not None:
        if isinstance(compression, (int, float)):
            bit_rate = int(compression)
        else:
            import warnings

            warnings.warn(
                f"Unsupported compression type {type(compression)}. "
                "TorchCodec AudioEncoder expects int or float for bit_rate.",
                UserWarning,
                stacklevel=2,
            )

    # Save to file
    try:
        encoder.to_file(uri, bit_rate=bit_rate)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio to {uri}: {e}") from e
