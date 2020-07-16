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

    @property
    def base_temp_dir(self):
        # If TORCHAUDIO_TEST_TEMP_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = 'TORCHAUDIO_TEST_TEMP_DIR'
        if key in os.environ:
            return os.environ[key]
        if self.__class__.temp_dir_ is None:
            self.__class__.temp_dir_ = tempfile.TemporaryDirectory()
        return self.__class__.temp_dir_.name

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.temp_dir_ is not None:
            cls.temp_dir_.cleanup()
            cls.temp_dir_ = None

    def get_temp_path(self, *paths):
        temp_dir = os.path.join(self.base_temp_dir, self.id())
        path = os.path.join(temp_dir, *paths)
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
