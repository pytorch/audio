import pytest

from torchaudio._internal.module_utils import UNSUPPORTED

@pytest.mark.parametrize("func", UNSUPPORTED)
def test_deprecations(func):
    with pytest.warns(UserWarning, match="deprecated"):
        try:
            func()
        except Exception as e:
	     # Type or Runtime error because we call func() without proper parameters.
             # The deprecation warning is still properly raised, since it is emitted before
             # the underlying (deprecated) function is called.
            assert isinstance(e, (TypeError, RuntimeError))
