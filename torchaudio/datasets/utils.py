import logging
import os
import tarfile
import zipfile
from typing import Any, List, Optional

import torchaudio


def extract_archive(from_path: str, to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    """Extract archive.
    Args:
        from_path (str): the path of the archive.
        to_path (str or None, optional): the root path of the extraced files (directory of from_path)
            (Default: ``None``)
        overwrite (bool, optional): overwrite existing files (Default: ``False``)

    Returns:
        List[str]: List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchaudio.datasets.utils.download_from_url(url, from_path)
        >>> torchaudio.datasets.utils.extract_archive(from_path, to_path)
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    try:
        with tarfile.open(from_path, "r") as tar:
            logging.info("Opened tar file {}.".format(from_path))
            files = []
            for file_ in tar:  # type: Any
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            return files
    except tarfile.ReadError:
        pass

    try:
        with zipfile.ZipFile(from_path, "r") as zfile:
            logging.info("Opened zip file {}.".format(from_path))
            files = zfile.namelist()
            for file_ in files:
                file_path = os.path.join(to_path, file_)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        return files
    except zipfile.BadZipFile:
        pass

    raise NotImplementedError("We currently only support tar.gz, tgz, and zip achives.")


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
