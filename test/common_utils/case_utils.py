import shutil
import os.path
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
import torchaudio
from torchaudio._internal.module_utils import is_module_available

from .backend_utils import set_audio_backend


class TempDirMixin:
    """Mixin to provide easy access to temp dir"""
    temp_dir_ = None
    base_temp_dir = None
    temp_dir = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # If TORCHAUDIO_TEST_TEMP_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = 'TORCHAUDIO_TEST_TEMP_DIR'
        if key in os.environ:
            cls.base_temp_dir = os.environ[key]
        else:
            cls.temp_dir_ = tempfile.TemporaryDirectory()
            cls.base_temp_dir = cls.temp_dir_.name

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if isinstance(cls.temp_dir_, tempfile.TemporaryDirectory):
            cls.temp_dir_.cleanup()

    def setUp(self):
        super().setUp()
        self.temp_dir = os.path.join(self.base_temp_dir, self.id())

    def get_temp_path(self, *paths):
        path = os.path.join(self.temp_dir, *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


class TestBaseMixin:
    """Mixin to provide consistent way to define device/dtype/backend aware TestCase"""
    dtype = None
    device = None
    backend = None

    def setUp(self):
        super().setUp()
        set_audio_backend(self.backend)


class TorchaudioTestCase(TestBaseMixin, PytorchTestCase):
    pass


def skipIfNoExec(cmd):
    return unittest.skipIf(shutil.which(cmd) is None, f'`{cmd}` is not available')


def skipIfNoModule(module, display_name=None):
    display_name = display_name or module
    return unittest.skipIf(not is_module_available(module), f'"{display_name}" is not available')


skipIfNoSoxBackend = unittest.skipIf(
    'sox' not in torchaudio.list_audio_backends(), 'Sox backend not available')
skipIfNoCuda = unittest.skipIf(not torch.cuda.is_available(), reason='CUDA not available')
skipIfNoExtension = skipIfNoModule('torchaudio._torchaudio', 'torchaudio C++ extension')
