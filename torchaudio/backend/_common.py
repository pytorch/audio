def __getattr__(name: str):
    import warnings

    if name == "AudioMetaData":
        warnings.warn(
            "`torchaudio.backend.common.AudioMetaData` has been moved to "
            "`torchaudio.AudioMetaData`. Please update the import path.",
            stacklevel=2,
        )
        from torchaudio._backend.common import AudioMetaData

        return AudioMetaData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
