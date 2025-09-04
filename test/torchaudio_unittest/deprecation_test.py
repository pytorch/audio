import pytest

import torch

from torchaudio._internal.module_utils import UNSUPPORTED


@pytest.mark.parametrize("func", UNSUPPORTED)
def test_deprecations(func):
    with pytest.warns(UserWarning, match="deprecated"):
        try:
            func()
        except Exception as e:
            assert isinstance(e, (TypeError, RuntimeError, ValueError, ImportError))
