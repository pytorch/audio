import torchaudio


def __getattr__(name: str):
    if name in ["HDemucs", "hdemucs_high", "hdemucs_medium", "hdemucs_low"]:
        import warnings

        warnings.warn(
            f"{__name__}.{name} has been moved to torchaudio.models.hdemucs",
            DeprecationWarning,
        )

        demucs = eval(f"torchaudio.models.{name}")
        return demucs

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return ["HDemucs", "hdemucs_high", "hdemucs_medium", "hdemucs_low"]
