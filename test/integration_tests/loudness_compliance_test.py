"""Test suite for compliance with the ITU-R BS.1770-4 recommendation"""
import zipfile

import pytest

import torch
import torchaudio
from torchaudio.utils import load_torchcodec
import torchaudio.functional as F


# Test files linked in https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BS.2217-2-2016-PDF-E.pdf
@pytest.mark.parametrize(
    "filename,url,expected",
    [
        (
            "1770-2_Comp_RelGateTest",
            "http://www.itu.int/dms_pub/itu-r/oth/11/02/R11020000010030ZIPM.zip",
            -10.0,
        ),
        (
            "1770-2_Comp_AbsGateTest",
            "http://www.itu.int/dms_pub/itu-r/oth/11/02/R11020000010029ZIPM.zip",
            -69.5,
        ),
        (
            "1770-2_Comp_24LKFS_500Hz_2ch",
            "http://www.itu.int/dms_pub/itu-r/oth/11/02/R11020000010018ZIPM.zip",
            -24.0,
        ),
        (
            "1770-2 Conf Mono Voice+Music-24LKFS",
            "http://www.itu.int/dms_pub/itu-r/oth/11/02/R11020000010038ZIPM.zip",
            -24.0,
        ),
    ],
)
def test_loudness(tmp_path, filename, url, expected):
    zippath = tmp_path / filename
    torch.hub.download_url_to_file(url, zippath, progress=False)
    with zipfile.ZipFile(zippath) as file:
        file.extractall(zippath.parent)

    waveform, sample_rate = load_torchcodec(zippath.with_suffix(".wav"))
    loudness = F.loudness(waveform, sample_rate)
    expected = torch.tensor(expected, dtype=loudness.dtype, device=loudness.device)
    assert torch.allclose(loudness, expected, rtol=0.01, atol=0.1)
