import pytest

import torch

from torchaudio._internal.module_utils import UNSUPPORTED
from torchaudio.prototype.functional import exp_sigmoid

# Importing prototype modules is needed to trigger the registration of the
# corresponding APIs in the UNSUPPORTED register.
from torchaudio.prototype import datasets, functional, models, pipelines, transforms


@pytest.mark.parametrize("func", UNSUPPORTED)
def test_deprecations(func):
    with pytest.warns(UserWarning, match="deprecated"):
        try:
            func()
        except Exception as e:
            assert isinstance(e, (TypeError, RuntimeError, ValueError, ImportError))


# It's not great, but the deprecation decorator we're using breaks torchscript
# This test just illustrates this behavior. Ideally, we wouldn't break
# torchscript users. But oh well, torchscript is supposed to have been
# deprecated for years.
@pytest.mark.parametrize("scripted", (True, False))
def test_torchscript_fails(scripted):
    f = exp_sigmoid
    if scripted:
        pytest.xfail("Deprecation decorator breaks torchscript")
        f = torch.jit.script(f)
    f(torch.rand(2, 1000))
