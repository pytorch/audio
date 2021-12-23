import os.path
import shutil
import subprocess
import tempfile
import time
import unittest

import torch
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
from torchaudio._internal.module_utils import is_module_available, is_sox_available, is_kaldi_available

from .backend_utils import set_audio_backend
from .ctc_decoder_utils import is_ctc_decoder_available


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
        super().tearDownClass()
        if cls.temp_dir_ is not None:
            cls.temp_dir_.cleanup()
            cls.temp_dir_ = None

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

    @property
    def complex_dtype(self):
        if self.dtype in ["float32", "float", torch.float, torch.float32]:
            return torch.cfloat
        if self.dtype in ["float64", "double", torch.double, torch.float64]:
            return torch.cdouble
        raise ValueError(f"No corresponding complex dtype for {self.dtype}")


class TorchaudioTestCase(TestBaseMixin, PytorchTestCase):
    pass


def skipIfNoExec(cmd):
    return unittest.skipIf(shutil.which(cmd) is None, f"`{cmd}` is not available")


def skipIfNoModule(module, display_name=None):
    display_name = display_name or module
    return unittest.skipIf(not is_module_available(module), f'"{display_name}" is not available')


def skipIfNoCuda(test_item):
    if torch.cuda.is_available():
        return test_item
    force_cuda_test = os.environ.get("TORCHAUDIO_TEST_FORCE_CUDA", "0")
    if force_cuda_test not in ["0", "1"]:
        raise ValueError('"TORCHAUDIO_TEST_FORCE_CUDA" must be either "0" or "1".')
    if force_cuda_test == "1":
        raise RuntimeError('"TORCHAUDIO_TEST_FORCE_CUDA" is set but CUDA is not available.')
    return unittest.skip("CUDA is not available.")(test_item)


skipIfNoSox = unittest.skipIf(not is_sox_available(), reason="Sox not available")
skipIfNoKaldi = unittest.skipIf(not is_kaldi_available(), reason="Kaldi not available")
skipIfNoCtcDecoder = unittest.skipIf(not is_ctc_decoder_available(), reason="CTC decoder not available")
skipIfRocm = unittest.skipIf(
    os.getenv("TORCHAUDIO_TEST_WITH_ROCM", "0") == "1", reason="test doesn't currently work on the ROCm stack"
)
skipIfNoQengine = unittest.skipIf(
    "fbgemm" not in torch.backends.quantized.supported_engines, reason="`fbgemm` is not available."
)
