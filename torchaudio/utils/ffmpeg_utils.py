"""Module to change the configuration of FFmpeg libraries (such as libavformat).

It affects functionalities in :py:mod:`torchaudio.io` (and indirectly :py:func:`torchaudio.load`).
"""
from typing import Dict, List, Tuple

import torchaudio


@torchaudio._extension.fail_if_no_ffmpeg
def get_versions() -> Dict[str, Tuple[int]]:
    """Get the versions of FFmpeg libraries

    Returns:
        dict: mapping from library names to version string,
            i.e. `"libavutil": (56, 22, 100)`.
    """
    return torchaudio._extension._FFMPEG_EXT.get_versions()


@torchaudio._extension.fail_if_no_ffmpeg
def get_log_level() -> int:
    """Get the log level of FFmpeg.

    See :py:func:`set_log_level` for the detailo.
    """
    return torchaudio._extension._FFMPEG_EXT.get_log_level()


@torchaudio._extension.fail_if_no_ffmpeg
def set_log_level(level: int):
    """Set the log level of FFmpeg (libavformat etc)

    Arguments:
        level (int): Log level. The larger, the more verbose.

            The following values are common values, the corresponding ``ffmpeg``'s
            ``-loglevel`` option value and desription.

                * ``-8`` (``quiet``):
                  Print no output.
                * ``0`` (``panic``):
                  Something went really wrong and we will crash now.
                * ``8`` (``fatal``):
                  Something went wrong and recovery is not possible.
                  For example, no header was found for a format which depends
                  on headers or an illegal combination of parameters is used.
                * ``16`` (``error``):
                  Something went wrong and cannot losslessly be recovered.
                  However, not all future data is affected.
                * ``24`` (``warning``):
                  Something somehow does not look correct.
                  This may or may not lead to problems.
                * ``32`` (``info``):
                  Standard information.
                * ``40`` (``verbose``):
                  Detailed information.
                * ``48`` (``debug``):
                  Stuff which is only useful for libav* developers.
                * ``56`` (``trace``):
                  Extremely verbose debugging, useful for libav* development.

    """
    torchaudio._extension._FFMPEG_EXT.set_log_level(level)


@torchaudio._extension.fail_if_no_ffmpeg
def get_demuxers() -> Dict[str, str]:
    """Get the available demuxers.

    Returns:
        Dict[str, str]: Mapping from demuxer (format) short name to long name.

    Example
        >>> for k, v in get_demuxers().items():
        >>>     print(f"{k}: {v}")
        ... aa: Audible AA format files
        ... aac: raw ADTS AAC (Advanced Audio Coding)
        ... aax: CRI AAX
        ... ac3: raw AC-3
    """
    return torchaudio._extension._FFMPEG_EXT.get_demuxers()


@torchaudio._extension.fail_if_no_ffmpeg
def get_muxers() -> Dict[str, str]:
    """Get the available muxers.

    Returns:
        Dict[str, str]: Mapping from muxer (format) short name to long name.

    Example
        >>> for k, v in get_muxers().items():
        >>>     print(f"{k}: {v}")
        ... a64: a64 - video for Commodore 64
        ... ac3: raw AC-3
        ... adts: ADTS AAC (Advanced Audio Coding)
        ... adx: CRI ADX
        ... aiff: Audio IFF
    """
    return torchaudio._extension._FFMPEG_EXT.get_muxers()


@torchaudio._extension.fail_if_no_ffmpeg
def get_audio_decoders() -> Dict[str, str]:
    """Get the available audio decoders.

    Returns:
        Dict[str, str]: Mapping from decoder short name to long name.

    Example
        >>> for k, v in get_audio_decoders().items():
        >>>     print(f"{k}: {v}")
        ... a64: a64 - video for Commodore 64
        ... ac3: raw AC-3
        ... adts: ADTS AAC (Advanced Audio Coding)
        ... adx: CRI ADX
        ... aiff: Audio IFF
    """
    return torchaudio._extension._FFMPEG_EXT.get_audio_decoders()


@torchaudio._extension.fail_if_no_ffmpeg
def get_audio_encoders() -> Dict[str, str]:
    """Get the available audio encoders.

    Returns:
        Dict[str, str]: Mapping from encoder short name to long name.

    Example
        >>> for k, v in get_audio_encoders().items():
        >>>     print(f"{k}: {v}")
        ... comfortnoise: RFC 3389 comfort noise generator
        ... s302m: SMPTE 302M
        ... aac: AAC (Advanced Audio Coding)
        ... ac3: ATSC A/52A (AC-3)
        ... ac3_fixed: ATSC A/52A (AC-3)
        ... alac: ALAC (Apple Lossless Audio Codec)
    """
    return torchaudio._extension._FFMPEG_EXT.get_audio_encoders()


@torchaudio._extension.fail_if_no_ffmpeg
def get_video_decoders() -> Dict[str, str]:
    """Get the available video decoders.

    Returns:
        Dict[str, str]: Mapping from decoder short name to long name.

    Example
        >>> for k, v in get_video_decoders().items():
        >>>     print(f"{k}: {v}")
        ... aasc: Autodesk RLE
        ... aic: Apple Intermediate Codec
        ... alias_pix: Alias/Wavefront PIX image
        ... agm: Amuse Graphics Movie
        ... amv: AMV Video
        ... anm: Deluxe Paint Animation
    """
    return torchaudio._extension._FFMPEG_EXT.get_video_decoders()


@torchaudio._extension.fail_if_no_ffmpeg
def get_video_encoders() -> Dict[str, str]:
    """Get the available video encoders.

    Returns:
        Dict[str, str]: Mapping from encoder short name to long name.

    Example
        >>> for k, v in get_audio_encoders().items():
        >>>     print(f"{k}: {v}")
        ... a64multi: Multicolor charset for Commodore 64
        ... a64multi5: Multicolor charset for Commodore 64, extended with 5th color (colram)
        ... alias_pix: Alias/Wavefront PIX image
        ... amv: AMV Video
        ... apng: APNG (Animated Portable Network Graphics) image
        ... asv1: ASUS V1
        ... asv2: ASUS V2
    """
    return torchaudio._extension._FFMPEG_EXT.get_video_encoders()


@torchaudio._extension.fail_if_no_ffmpeg
def get_input_devices() -> Dict[str, str]:
    """Get the available input devices.

    Returns:
        Dict[str, str]: Mapping from device short name to long name.

    Example
        >>> for k, v in get_input_devices().items():
        >>>     print(f"{k}: {v}")
        ... avfoundation: AVFoundation input device
        ... lavfi: Libavfilter virtual input device
    """
    return torchaudio._extension._FFMPEG_EXT.get_input_devices()


@torchaudio._extension.fail_if_no_ffmpeg
def get_output_devices() -> Dict[str, str]:
    """Get the available output devices.

    Returns:
        Dict[str, str]: Mapping from device short name to long name.

    Example
        >>> for k, v in get_output_devices().items():
        >>>     print(f"{k}: {v}")
        ... audiotoolbox: AudioToolbox output device
    """
    return torchaudio._extension._FFMPEG_EXT.get_output_devices()


@torchaudio._extension.fail_if_no_ffmpeg
def get_input_protocols() -> List[str]:
    """Get the supported input protocols.

    Returns:
        List[str]: The names of supported input protocols

    Example
        >>> print(get_input_protocols())
        ... ['file', 'ftp', 'hls', 'http','https', 'pipe', 'rtmp', 'tcp', 'tls', 'udp', 'unix']
    """
    return torchaudio._extension._FFMPEG_EXT.get_input_protocols()


@torchaudio._extension.fail_if_no_ffmpeg
def get_output_protocols() -> List[str]:
    """Get the supported output protocols.

    Returns:
        list of str: The names of supported output protocols

    Example
        >>> print(get_output_protocols())
        ... ['file', 'ftp', 'http', 'https', 'md5', 'pipe', 'prompeg', 'rtmp', 'tee', 'tcp', 'tls', 'udp', 'unix']
    """
    return torchaudio._extension._FFMPEG_EXT.get_output_protocols()


@torchaudio._extension.fail_if_no_ffmpeg
def get_build_config() -> str:
    """Get the FFmpeg build configuration

    Returns:
        str: Build configuration string.

    Example
        >>> print(get_build_config())
        --prefix=/Users/runner/miniforge3 --cc=arm64-apple-darwin20.0.0-clang --enable-gpl --enable-hardcoded-tables --enable-libfreetype --enable-libopenh264 --enable-neon --enable-libx264 --enable-libx265 --enable-libaom --enable-libsvtav1 --enable-libxml2 --enable-libvpx --enable-pic --enable-pthreads --enable-shared --disable-static --enable-version3 --enable-zlib --enable-libmp3lame --pkg-config=/Users/runner/miniforge3/conda-bld/ffmpeg_1646229390493/_build_env/bin/pkg-config --enable-cross-compile --arch=arm64 --target-os=darwin --cross-prefix=arm64-apple-darwin20.0.0- --host-cc=/Users/runner/miniforge3/conda-bld/ffmpeg_1646229390493/_build_env/bin/x86_64-apple-darwin13.4.0-clang  # noqa
    """
    return torchaudio._extension._FFMPEG_EXT.get_build_config()


@torchaudio._extension.fail_if_no_ffmpeg
def clear_cuda_context_cache():
    """Clear the CUDA context used by CUDA Hardware accelerated video decoding"""
    torchaudio._extension._FFMPEG_EXT.clear_cuda_context_cache()
