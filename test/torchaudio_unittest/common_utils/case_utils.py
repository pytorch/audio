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
import torio
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
from torchaudio._internal.module_utils import eval_env, is_module_available


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
    _port = 12345

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
    """Mixin to provide consistent way to define device/dtype aware TestCase"""

    dtype = None
    device = None

    def setUp(self):
        super().setUp()
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


_IS_FFMPEG_AVAILABLE = torio._extension.lazy_import_ffmpeg_ext().is_available()
_IS_CTC_DECODER_AVAILABLE = None
_IS_CUDA_CTC_DECODER_AVAILABLE = None


def is_ctc_decoder_available():
    global _IS_CTC_DECODER_AVAILABLE
    if _IS_CTC_DECODER_AVAILABLE is None:
        try:
            from torchaudio.models.decoder import CTCDecoder  # noqa: F401

            _IS_CTC_DECODER_AVAILABLE = True
        except Exception:
            _IS_CTC_DECODER_AVAILABLE = False
    return _IS_CTC_DECODER_AVAILABLE


def is_cuda_ctc_decoder_available():
    global _IS_CUDA_CTC_DECODER_AVAILABLE
    if _IS_CUDA_CTC_DECODER_AVAILABLE is None:
        try:
            from torchaudio.models.decoder import CUCTCDecoder  # noqa: F401

            _IS_CUDA_CTC_DECODER_AVAILABLE = True
        except Exception:
            _IS_CUDA_CTC_DECODER_AVAILABLE = False
    return _IS_CUDA_CTC_DECODER_AVAILABLE


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


_IN_CI = eval_env("CI", default=False)


def _skipIf(condition, reason, key):
    if not condition:
        return _pass

    # In CI, default to fail, so as to prevent accidental skip.
    # In other env, default to skip
    var = f"TORCHAUDIO_TEST_ALLOW_SKIP_IF_{key}"
    skip_allowed = eval_env(var, default=not _IN_CI)
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
# Skip test if CUDA memory is not enough
# TODO: detect the real CUDA memory size and allow call site to configure how much the test needs
skipIfCudaSmallMemory = _skipIf(
    "CI" in os.environ and torch.cuda.is_available(),  # temporary
    reason="CUDA does not have enough memory.",
    key="CUDA_SMALL_MEMORY",
)

skipIfNoRIR = _skipIf(
    not torchaudio._extension._IS_RIR_AVAILABLE,
    reason="RIR features are not available.",
    key="NO_RIR",
)
skipIfNoCtcDecoder = _skipIf(
    not is_ctc_decoder_available(),
    reason="CTC decoder not available.",
    key="NO_CTC_DECODER",
)
skipIfNoCuCtcDecoder = _skipIf(
    not is_cuda_ctc_decoder_available(),
    reason="CUCTC decoder not available.",
    key="NO_CUCTC_DECODER",
)
skipIfRocm = _skipIf(
    eval_env("TORCHAUDIO_TEST_WITH_ROCM", default=False),
    reason="The test doesn't currently work on the ROCm stack.",
    key="ON_ROCM",
)
skipIfNoQengine = _skipIf(
    "fbgemm" not in torch.backends.quantized.supported_engines,
    reason="`fbgemm` is not available.",
    key="NO_QUANTIZATION",
)
skipIfNoFFmpeg = _skipIf(
    not _IS_FFMPEG_AVAILABLE,
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
skipIfNoAudioDevice = _skipIf(
    not _IS_FFMPEG_AVAILABLE,
    reason="No output audio device is available.",
    key="NO_AUDIO_OUT_DEVICE",
)
skipIfNoMacOS = _skipIf(
    sys.platform != "darwin",
    reason="This feature is only available for MacOS.",
    key="NO_MACOS",
)
disabledInCI = _skipIf(
    "CI" in os.environ,
    reason="Tests are failing on CI consistently. Disabled while investigating.",
    key="TEMPORARY_DISABLED",
)


def skipIfNoHWAccel(name):
    key = "NO_HW_ACCEL"
    return _skipIf(True, reason="ffmpeg features are not available.", key=key)

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
