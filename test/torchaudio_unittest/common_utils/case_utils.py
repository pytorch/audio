import functools
import os.path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from itertools import zip_longest

import torch
import torchaudio
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
from torchaudio._internal.module_utils import is_module_available

from .backend_utils import set_audio_backend


class TempDirMixin:
    """Mixin to provide easy access to temp dir"""

    temp_dir_ = None

    @classmethod
    def get_base_temp_dir(cls):
        # If TORCHAUDIO_TEST_TEMP_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = "TORCHAUDIO_TEST_TEMP_DIR"
        if key in os.environ:
            return os.environ[key]
        if cls.temp_dir_ is None:
            cls.temp_dir_ = tempfile.TemporaryDirectory()
        return cls.temp_dir_.name

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir_ is not None:
            try:
                cls.temp_dir_.cleanup()
                cls.temp_dir_ = None
            except PermissionError:
                # On Windows there is a know issue with `shutil.rmtree`,
                # which fails intermittenly.
                #
                # https://github.com/python/cpython/issues/74168
                #
                # We observed this on CircleCI, where Windows job raises
                # PermissionError.
                #
                # Following the above thread, we ignore it.
                pass
        super().tearDownClass()

    def get_temp_path(self, *paths):
        temp_dir = os.path.join(self.get_base_temp_dir(), self.id())
        path = os.path.join(temp_dir, *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


class HttpServerMixin(TempDirMixin):
    """Mixin that serves temporary directory as web server

    This class creates temporary directory and serve the directory as HTTP service.
    The server is up through the execution of all the test suite defined under the subclass.
    """

    _proc = None
    _port = 8000

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._proc = subprocess.Popen(
            ["python", "-m", "http.server", f"{cls._port}"], cwd=cls.get_base_temp_dir(), stderr=subprocess.DEVNULL
        )  # Disable server-side error log because it is confusing
        time.sleep(2.0)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._proc.kill()

    def get_url(self, *route):
        return f'http://localhost:{self._port}/{self.id()}/{"/".join(route)}'


class TestBaseMixin:
    """Mixin to provide consistent way to define device/dtype/backend aware TestCase"""

    dtype = None
    device = None
    backend = None

    def setUp(self):
        super().setUp()
        set_audio_backend(self.backend)
        torch.random.manual_seed(2434)

    @property
    def complex_dtype(self):
        if self.dtype in ["float32", "float", torch.float, torch.float32]:
            return torch.cfloat
        if self.dtype in ["float64", "double", torch.double, torch.float64]:
            return torch.cdouble
        raise ValueError(f"No corresponding complex dtype for {self.dtype}")


class TorchaudioTestCase(TestBaseMixin, PytorchTestCase):
    pass


def is_ffmpeg_available():
    return torchaudio._extension._FFMPEG_INITIALIZED


_IS_CTC_DECODER_AVAILABLE = None


def is_ctc_decoder_available():
    global _IS_CTC_DECODER_AVAILABLE
    if _IS_CTC_DECODER_AVAILABLE is None:
        try:
            from torchaudio.models.decoder import CTCDecoder  # noqa: F401

            _IS_CTC_DECODER_AVAILABLE = True
        except Exception:
            _IS_CTC_DECODER_AVAILABLE = False
    return _IS_CTC_DECODER_AVAILABLE


def _eval_env(var, default):
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        # fmt: off
        raise RuntimeError(
            f"Unexpected environment variable value `{var}={val}`. "
            f"Expected one of {trues + falses}")
        # fmt: on
    return False


def _fail(reason):
    def deco(test_item):
        if isinstance(test_item, type):
            # whole class is decorated
            def _f(self, *_args, **_kwargs):
                raise RuntimeError(reason)

            test_item.setUp = _f
            return test_item

        # A method is decorated
        @functools.wraps(test_item)
        def f(*_args, **_kwargs):
            raise RuntimeError(reason)

        return f

    return deco


def _pass(test_item):
    return test_item


_IN_CI = _eval_env("CI", default=False)


def _skipIf(condition, reason, key):
    if not condition:
        return _pass

    # In CI, default to fail, so as to prevent accidental skip.
    # In other env, default to skip
    var = f"TORCHAUDIO_TEST_ALLOW_SKIP_IF_{key}"
    skip_allowed = _eval_env(var, default=not _IN_CI)
    if skip_allowed:
        return unittest.skip(reason)
    return _fail(f"{reason} But the test cannot be skipped. (CI={_IN_CI}, {var}={skip_allowed}.)")


def skipIfNoExec(cmd):
    return _skipIf(
        shutil.which(cmd) is None,
        f"`{cmd}` is not available.",
        key=f"NO_CMD_{cmd.upper().replace('-', '_')}",
    )


def skipIfNoModule(module, display_name=None):
    return _skipIf(
        not is_module_available(module),
        f'"{display_name or module}" is not available.',
        key=f"NO_MOD_{module.replace('.', '_')}",
    )


skipIfNoCuda = _skipIf(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
    key="NO_CUDA",
)
skipIfNoSox = _skipIf(
    not torchaudio._extension._SOX_INITIALIZED,
    reason="Sox features are not available.",
    key="NO_SOX",
)
skipIfNoKaldi = _skipIf(
    not torchaudio._extension._IS_KALDI_AVAILABLE,
    reason="Kaldi features are not available.",
    key="NO_KALDI",
)
skipIfNoCtcDecoder = _skipIf(
    not is_ctc_decoder_available(),
    reason="CTC decoder not available.",
    key="NO_CTC_DECODER",
)
skipIfRocm = _skipIf(
    _eval_env("TORCHAUDIO_TEST_WITH_ROCM", default=False),
    reason="The test doesn't currently work on the ROCm stack.",
    key="ON_ROCM",
)
skipIfNoQengine = _skipIf(
    "fbgemm" not in torch.backends.quantized.supported_engines,
    reason="`fbgemm` is not available.",
    key="NO_QUANTIZATION",
)
skipIfNoFFmpeg = _skipIf(
    not is_ffmpeg_available(),
    reason="ffmpeg features are not available.",
    key="NO_FFMPEG",
)
skipIfPy310 = _skipIf(
    sys.version_info >= (3, 10, 0),
    reason=(
        "Test is known to fail for Python 3.10, disabling for now"
        "See: https://github.com/pytorch/audio/pull/2224#issuecomment-1048329450"
    ),
    key="ON_PYTHON_310",
)


def zip_equal(*iterables):
    """With the regular Python `zip` function, if one iterable is longer than the other,
    the remainder portions are ignored.This is resolved in Python 3.10 where we can use
    `strict=True` in the `zip` function
    From https://github.com/pytorch/text/blob/c047efeba813ac943cb8046a49e858a8b529d577/test/torchtext_unittest/common/case_utils.py#L45-L54  # noqa: E501
    """
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo
