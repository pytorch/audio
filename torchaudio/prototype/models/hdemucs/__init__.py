import torchaudio

functions = ["HDemucs", "hdemucs_high", "hdemucs_medium", "hdemucs_low"]


def __getattr__(name: str):
    if name in functions:
        import warnings

        warnings.warn(
            f"{__name__}.{name} has been moved to torchaudio.models.hdemucs",
            DeprecationWarning,
        )

        return getattr(torchaudio.models, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return functions
