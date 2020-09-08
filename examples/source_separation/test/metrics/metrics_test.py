import torch
from torch.testing._internal.common_utils import TestCase
import pytest

try:
    from . import reference
except ImportError:
    raise RuntimeError(
        "Reference implementation is not found. Download "
        "https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py"
        " and place it as `reference.py`"
    )

from utils import metrics


_Case = TestCase()


def _assert_equal(*args, **kwargs):
    _Case.assertEqual(*args, **kwargs)


@pytest.mark.parametrize("batch_size", [1, 2, 32])
def test_sdr(batch_size):
    """sdr produces the same result as the reference implementation

    https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py#L34-L56
    """
    num_frames = 256

    estimation = torch.rand(batch_size, num_frames)
    origin = torch.rand(batch_size, num_frames)

    sdr_ref = reference.calc_sdr_torch(estimation, origin)
    sdr = metrics.sdr(estimation.unsqueeze(1), origin.unsqueeze(1)).squeeze(1)

    _assert_equal(sdr, sdr_ref)


@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("num_sources", [2, 3, 4, 5])
def test_sdr_pit(batch_size, num_sources):
    """sdr_pit produces the same result as the reference implementation

    https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py#L107-L153
    """
    num_frames = 256

    estimation = torch.randn(batch_size, num_sources, num_frames)
    origin = torch.randn(batch_size, num_sources, num_frames)

    estimation -= estimation.mean(axis=2, keepdim=True)
    origin -= origin.mean(axis=2, keepdim=True)

    batch_sdr_ref = reference.batch_SDR_torch(estimation, origin)
    batch_sdr = metrics.sdr_pit(estimation, origin)

    _assert_equal(batch_sdr, batch_sdr_ref)
