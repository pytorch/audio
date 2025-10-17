import inspect

from torchaudio._internal.module_utils import dropping_class_support, dropping_support

_CTC_DECODERS = [
    "CTCHypothesis",
    "CTCDecoder",
    "CTCDecoderLM",
    "CTCDecoderLMState",
    "ctc_decoder",
    "download_pretrained_files",
]
_CUDA_CTC_DECODERS = [
    "CUCTCDecoder",
    "CUCTCHypothesis",
    "cuda_ctc_decoder",
]


def __getattr__(name: str):
    if name in _CTC_DECODERS:
        try:
            from . import _ctc_decoder
        except Exception as err:
            raise RuntimeError(
                "CTC Decoder suit requires flashlight-text package and optionally KenLM. Please install them."
            ) from err

        item = getattr(_ctc_decoder, name)
        globals()[name] = item
        return item
    elif name in _CUDA_CTC_DECODERS:
        try:
            from . import _cuda_ctc_decoder
        except AttributeError as err:
            raise RuntimeError(
                "To use CUCTC decoder, please set BUILD_CUDA_CTC_DECODER=1 when building from source."
            ) from err

        # TODO: when all unsupported classes are removed, replace the
        # following if-else block with
        #  item = getattr(_cuda_ctc_decoder, name)
        orig_item = getattr(_cuda_ctc_decoder, name)
        if inspect.isclass(orig_item) or (
            # workaround a failure to detect type instances
            # after sphinx autodoc mocking, required for
            # building docs
            getattr(orig_item, "__sphinx_mock__", False)
            and inspect.isclass(orig_item.__class__)
        ):
            item = dropping_class_support(orig_item)
        else:
            item = dropping_support(orig_item)

        globals()[name] = item
        return item
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__)


__all__ = _CTC_DECODERS + _CUDA_CTC_DECODERS
