def __getattr__(name: str):
    import warnings

    warnings.warn(
        "Torchaudio's I/O functions now support per-call backend dispatch. "
        "Importing backend implementation directly is no longer guaranteed to work. "
        "Please use `backend` keyword with load/save/info function, instead of "
        "calling the underlying implementation directly.",
        stacklevel=2,
    )

    from torchaudio._backend import soundfile_backend

    return getattr(soundfile_backend, name)
