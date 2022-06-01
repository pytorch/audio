def __getattr__(name: str):
    if name in ["ctc_decoder", "lexicon_decoder"]:
        import warnings

        from torchaudio.models.decoder import ctc_decoder

        warnings.warn(
            f"{__name__}.{name} has been moved to torchaudio.models.decoder.ctc_decoder",
            DeprecationWarning,
        )

        if name == "lexicon_decoder":
            global lexicon_decoder
            lexicon_decoder = ctc_decoder
            return lexicon_decoder
        else:
            return ctc_decoder
    elif name == "download_pretrained_files":
        import warnings

        from torchaudio.models.decoder import download_pretrained_files

        return download_pretrained_files

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return ["ctc_decoder", "lexicon_decoder", "download_pretrained_files"]
