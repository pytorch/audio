def is_ctc_decoder_available():
    try:
        import torchaudio.prototype.ctc_decoder  # noqa: F401

        return True
    except ImportError:
        return False
