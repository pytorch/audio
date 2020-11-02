import itertools
from unittest import skipIf

from parameterized import parameterized
from torchaudio._internal.module_utils import is_module_available


def name_func(func, _, params):
    return f'{func.__name__}_{"_".join(str(arg) for arg in params.args)}'


def dtype2subtype(dtype):
    return {
        "float64": "DOUBLE",
        "float32": "FLOAT",
        "int32": "PCM_32",
        "int16": "PCM_16",
        "uint8": "PCM_U8",
        "int8": "PCM_S8",
    }[dtype]


def skipIfFormatNotSupported(fmt):
    fmts = []
    if is_module_available("soundfile"):
        import soundfile

        fmts = soundfile.available_formats()
        return skipIf(fmt not in fmts, f'"{fmt}" is not supported by sondfile')
    return skipIf(True, '"soundfile" not available.')


def parameterize(*params):
    return parameterized.expand(list(itertools.product(*params)), name_func=name_func)
