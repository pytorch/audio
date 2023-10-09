import logging
import os
import tarfile
import zipfile
from typing import Any, List, Optional

import torchaudio

_LG = logging.getLogger(__name__)


def _extract_tar(from_path: str, to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)
    with tarfile.open(from_path, "r") as tar:
        files = []
        for file_ in tar:  # type: Any
            file_path = os.path.join(to_path, file_.name)
            if file_.isfile():
                files.append(file_path)
                if os.path.exists(file_path):
                    _LG.info("%s already extracted.", file_path)
                    if not overwrite:
                        continue
            tar.extract(file_, to_path)
        return files


def _extract_zip(from_path: str, to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as zfile:
        files = zfile.namelist()
        for file_ in files:
            file_path = os.path.join(to_path, file_)
            if os.path.exists(file_path):
                _LG.info("%s already extracted.", file_path)
                if not overwrite:
                    continue
            zfile.extract(file_, to_path)
    return files


def _load_waveform(
    root: str,
    filename: str,
    exp_sample_rate: int,
):
    path = os.path.join(root, filename)
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(f"sample rate should be {exp_sample_rate}, but got {sample_rate}")
    return waveform
