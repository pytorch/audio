def __getattr__(name: str):
    if name == "AudioMetaData":
        import warnings

        warnings.warn(
            "`torchaudio.backend.common.AudioMetaData` has been moved to "
            "`torchaudio.AudioMetaData`. Please update the import path.",
            stacklevel=2,
        )
        from torchaudio import AudioMetaData

        return AudioMetaData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
