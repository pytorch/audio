import pytest

import torch

from torchaudio._internal.module_utils import UNSUPPORTED
from torchaudio.sox_effects import apply_effects_tensor

# Importing prototype modules is needed to trigger the registration of the
# corresponding APIs in the UNSUPPORTED register.
from torchaudio.prototype import datasets, functional, models, pipelines, transforms

@pytest.mark.parametrize("func", UNSUPPORTED)
def test_deprecations(func):
    with pytest.warns(UserWarning, match="deprecated"):
        try:
            func()
        except Exception as e:
	        # Type, Runtime or Value Error  error because we call func() without proper parameters.
             # The deprecation warning is still properly raised, since it is emitted before
             # the underlying (deprecated) function is called.
            assert isinstance(e, (TypeError, RuntimeError, ValueError))


@pytest.mark.parametrize("scripted", (True, False))
def test_torchscript_fails(scripted):
    f = apply_effects_tensor
    if scripted:
        pytest.xfail("Deprecation decorator breaks torchscript")
        f = torch.jit.script(f)
    _, out_sample_rate = f(torch.rand(2, 1000), sample_rate=16_000, effects=[["rate", "8000"]])
    assert out_sample_rate == 8000
