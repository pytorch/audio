def __getattr__(name: str):
    if name == "Streamer":
        import warnings

        from torchaudio.io import StreamReader

        warnings.warn(
            f"{__name__}.{name} has been moved to torchaudio.io.StreamReader. Please use torchaudio.io.StreamReader",
            DeprecationWarning,
        )

        global Streamer
        Streamer = StreamReader
        return Streamer
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return ["Streamer"]
